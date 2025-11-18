#![recursion_limit = "256"]

use std::{
    path::PathBuf,
    sync::{Arc, Mutex},
};

use bevy::asset::RenderAssetUsages;
use bevy::{
    color::palettes::css::GOLD,
    diagnostic::{
        Diagnostic, DiagnosticPath, Diagnostics, DiagnosticsStore, FrameTimeDiagnosticsPlugin,
        RegisterDiagnostic,
    },
    ecs::world::CommandQueue,
    prelude::*,
    render::{
        RenderPlugin,
        render_resource::{Extent3d, TextureDimension, TextureFormat, TextureUsages},
        settings::{RenderCreation, WgpuFeatures, WgpuSettings},
    },
    tasks::{AsyncComputeTaskPool, Task, block_on, futures_lite::future},
    ui::widget::ImageNode,
};
use bevy_args::{Deserialize, Parser, Serialize, parse_args};
use bevy_burn::{BevyBurnBridgePlugin, BevyBurnHandle, BindingDirection, BurnDevice, TransferKind};
use bevy_burn_depth::{platform::camera::receive_image, process_frame};
use burn::prelude::*;
use burn_depth::model::depth_anything3::{DepthAnything3, DepthAnything3Config};
use burn_wgpu::Wgpu;
use image::RgbImage;

const DEFAULT_CHECKPOINT: &str = "assets/model/da3_small.mpk";
const MAX_IN_FLIGHT_TASKS: usize = 1;

#[derive(Resource, Clone, Debug, Serialize, Deserialize, Parser, Reflect)]
#[reflect(Resource)]
#[command(about = "bevy_burn_depth", version, long_about = None)]
pub struct BevyBurnDepthConfig {
    #[arg(long, default_value = "true")]
    pub press_esc_to_close: bool,

    #[arg(long, default_value = "true")]
    pub show_fps: bool,

    #[arg(long, default_value = DEFAULT_CHECKPOINT)]
    pub checkpoint: PathBuf,

    #[arg(long)]
    pub image_path: Option<PathBuf>,

    #[arg(long, default_value = "true")]
    pub normalize_relative_depth: bool,
}

impl Default for BevyBurnDepthConfig {
    fn default() -> Self {
        Self {
            press_esc_to_close: true,
            show_fps: true,
            checkpoint: PathBuf::from(DEFAULT_CHECKPOINT),
            image_path: None,
            normalize_relative_depth: true,
        }
    }
}

#[cfg(feature = "native")]
mod io {
    use std::path::Path;

    use burn::{
        prelude::*,
        record::{HalfPrecisionSettings, NamedMpkFileRecorder},
    };
    use burn_depth::model::depth_anything3::{
        DepthAnything3, DepthAnything3Config, with_model_load_stack,
    };

    pub async fn load_model<B: Backend>(
        config: DepthAnything3Config,
        checkpoint: &Path,
        device: &B::Device,
    ) -> DepthAnything3<B> {
        let recorder = NamedMpkFileRecorder::<HalfPrecisionSettings>::new();
        with_model_load_stack(|| {
            DepthAnything3::new(device, config).load_file(checkpoint, &recorder, device)
        })
        .expect("failed to load Depth Anything 3 checkpoint")
    }
}

#[cfg(feature = "web")]
mod io {
    use burn::{
        prelude::*,
        record::{HalfPrecisionSettings, NamedMpkBytesRecorder, Recorder},
    };
    use burn_depth::model::depth_anything3::{
        DepthAnything3, DepthAnything3Config, with_model_load_stack,
    };
    use js_sys::Uint8Array;
    use wasm_bindgen::JsCast;
    use wasm_bindgen_futures::JsFuture;
    use web_sys::{Request, RequestInit, RequestMode, Response, window};

    pub async fn load_model<B: Backend>(
        config: DepthAnything3Config,
        checkpoint: &str,
        device: &B::Device,
    ) -> DepthAnything3<B> {
        let opts = RequestInit::new();
        opts.set_method("GET");
        opts.set_mode(RequestMode::Cors);

        let request = Request::new_with_str_and_init(checkpoint, &opts)
            .unwrap_or_else(|_| panic!("failed to construct request for {checkpoint}"));

        let window = window().expect("missing browser window");
        let response = JsFuture::from(window.fetch_with_request(&request))
            .await
            .expect("failed to fetch checkpoint");
        let response: Response = response.dyn_into().expect("invalid response type");

        let buffer = JsFuture::from(
            response
                .array_buffer()
                .expect("failed to retrieve checkpoint buffer"),
        )
        .await
        .expect("failed to read checkpoint buffer");
        let bytes = Uint8Array::new(&buffer);

        let mut data = vec![0; bytes.length() as usize];
        bytes.copy_to(&mut data);

        let record = NamedMpkBytesRecorder::<HalfPrecisionSettings>::default()
            .load(data, device)
            .expect("failed to decode checkpoint");

        with_model_load_stack(|| {
            let model = DepthAnything3::new(device, config);
            model.load_record(record)
        })
    }
}

#[derive(Resource)]
struct DepthModelState {
    checkpoint: PathBuf,
    config: DepthAnything3Config,
    preferred_resolution: Option<usize>,
    model: Option<Arc<Mutex<DepthAnything3<Wgpu>>>>,
    load_task: Option<Task<DepthModelLoadResult>>,
    normalize_relative_depth: bool,
}

impl DepthModelState {
    fn new(
        checkpoint: PathBuf,
        config: DepthAnything3Config,
        normalize_relative_depth: bool,
    ) -> Self {
        Self {
            checkpoint,
            preferred_resolution: Some(config.image_size),
            config,
            model: None,
            load_task: None,
            normalize_relative_depth,
        }
    }
}

struct DepthModelLoadResult {
    model: DepthAnything3<Wgpu>,
    resolution: usize,
}

#[derive(Resource)]
struct DepthTexture {
    image: Handle<Image>,
    entity: Option<Entity>,
    width: u32,
    height: u32,
}

impl Default for DepthTexture {
    fn default() -> Self {
        Self {
            image: Handle::default(),
            entity: None,
            width: 1,
            height: 1,
        }
    }
}

#[derive(Resource, Default, Clone)]
struct StaticFrame(Option<Arc<RgbImage>>);

#[derive(Component)]
struct ProcessDepth(Task<CommandQueue>);

fn process_frames(
    mut commands: Commands,
    depth_model: Res<DepthModelState>,
    depth_texture: Res<DepthTexture>,
    static_frame: Res<StaticFrame>,
    active_tasks: Query<&ProcessDepth>,
    burn_device: Option<Res<BurnDevice>>,
) {
    let Some(model) = depth_model.model.as_ref() else {
        return;
    };

    let Some(image_entity) = depth_texture.entity else {
        return;
    };

    if active_tasks.iter().count() >= MAX_IN_FLIGHT_TASKS {
        return;
    }

    let Some(burn_device) = burn_device else {
        return;
    };

    if !burn_device.is_ready() {
        return;
    }

    let normalize_relative_depth = depth_model.normalize_relative_depth;
    let device = burn_device.device().unwrap().clone();
    let model = model.clone();

    let frame_source = if let Some(frame) = static_frame.0.as_ref() {
        Some((**frame).clone())
    } else {
        receive_image()
    };

    if let Some(frame) = frame_source {
        let thread_pool = AsyncComputeTaskPool::get();
        let task_entity = commands.spawn_empty().id();

        let patch_size = depth_model.config.patch_size;
        let preferred_resolution = depth_model.preferred_resolution;
        let task = thread_pool.spawn({
            let target = image_entity;
            async move {
                let tensor = process_frame(
                    frame,
                    model.clone(),
                    device.clone(),
                    patch_size,
                    preferred_resolution,
                    normalize_relative_depth,
                )
                .await;
                let [tensor_height, tensor_width, _] = tensor.dims();

                let mut queue = CommandQueue::default();
                queue.push(move |world: &mut World| {
                    let mut image_handle = None;
                    if let Ok(mut entity) = world.get_entity_mut(target) {
                        if let Some(mut handle) = entity.get_mut::<BevyBurnHandle<Wgpu>>() {
                            image_handle = Some(handle.bevy_image.clone());
                            handle.tensor = tensor.clone();
                            handle.upload = true;
                        }
                    }

                    if let Some(handle) = image_handle {
                        if let Some(mut images) = world.get_resource_mut::<Assets<Image>>() {
                            let desired = Extent3d {
                                width: tensor_width as u32,
                                height: tensor_height as u32,
                                depth_or_array_layers: 1,
                            };

                            let needs_resize = images
                                .get(handle.id())
                                .map(|image| {
                                    image.texture_descriptor.size.width != desired.width
                                        || image.texture_descriptor.size.height != desired.height
                                })
                                .unwrap_or(true);

                            if needs_resize {
                                let (fill_bytes, texture_format, _, texture_usage) =
                                    depth_image_setup();
                                let mut replacement = Image::new_fill(
                                    desired,
                                    TextureDimension::D2,
                                    fill_bytes,
                                    texture_format,
                                    RenderAssetUsages::RENDER_WORLD,
                                );
                                replacement.texture_descriptor.usage |= texture_usage;

                                let _ = images.insert(handle.id(), replacement);
                            }
                        }
                    }

                    if let Some(mut texture_meta) = world.get_resource_mut::<DepthTexture>() {
                        texture_meta.width = tensor_width as u32;
                        texture_meta.height = tensor_height as u32;
                    }

                    if let Ok(mut tracker) = world.get_entity_mut(task_entity) {
                        tracker.remove::<ProcessDepth>();
                        tracker.despawn();
                    }
                });

                queue
            }
        });

        commands.entity(task_entity).insert(ProcessDepth(task));
    }
}

fn begin_depth_model_load(
    mut depth_model: ResMut<DepthModelState>,
    burn_device: Option<Res<BurnDevice>>,
) {
    if depth_model.model.is_some() || depth_model.load_task.is_some() {
        return;
    }

    let Some(burn_device) = burn_device else {
        return;
    };

    if !burn_device.is_ready() {
        return;
    }

    let checkpoint = depth_model.checkpoint.clone();
    let config = depth_model.config.clone();
    let device = burn_device.device().unwrap().clone();

    log("loading depth model...");
    log(&format!("checkpoint: {}", checkpoint.display()));

    depth_model.load_task = Some(spawn_depth_model_load_task(config, checkpoint, device));
}

#[cfg(feature = "native")]
fn spawn_depth_model_load_task(
    config: DepthAnything3Config,
    checkpoint: PathBuf,
    device: <Wgpu as Backend>::Device,
) -> Task<DepthModelLoadResult> {
    AsyncComputeTaskPool::get().spawn(async move {
        log("begin load_model task (native)...");
        let depth = io::load_model::<Wgpu>(config, checkpoint.as_path(), &device).await;
        log("load_model task finished.");
        let resolution = depth.img_size();
        DepthModelLoadResult {
            model: depth,
            resolution,
        }
    })
}

#[cfg(feature = "web")]
fn spawn_depth_model_load_task(
    config: DepthAnything3Config,
    checkpoint: PathBuf,
    device: <Wgpu as Backend>::Device,
) -> Task<DepthModelLoadResult> {
    let checkpoint = normalize_web_checkpoint(&checkpoint);
    AsyncComputeTaskPool::get().spawn(async move {
        let depth = io::load_model::<Wgpu>(config, &checkpoint, &device).await;
        let resolution = depth.img_size();
        DepthModelLoadResult {
            model: depth,
            resolution,
        }
    })
}

#[cfg(feature = "web")]
fn normalize_web_checkpoint(path: &Path) -> String {
    let normalized = path.to_string_lossy().replace('\\', "/");
    if normalized.starts_with("./")
        || normalized.starts_with('/')
        || normalized.starts_with("http://")
        || normalized.starts_with("https://")
        || normalized.starts_with("data:")
    {
        normalized
    } else {
        format!("./{normalized}")
    }
}

fn finish_depth_model_load(mut depth_model: ResMut<DepthModelState>) {
    let Some(task) = depth_model.load_task.as_mut() else {
        return;
    };

    if let Some(result) = block_on(future::poll_once(task)) {
        log(&format!(
            "depth model ready (inference resolution: {}px)",
            result.resolution
        ));
        depth_model.model = Some(Arc::new(Mutex::new(result.model)));
        depth_model.load_task = None;
    }
}

fn handle_tasks(
    mut commands: Commands,
    mut diagnostics: Diagnostics,
    mut last_frame: Local<Time<Real>>,
    mut active_tasks: Query<&mut ProcessDepth>,
) {
    for mut task in &mut active_tasks {
        if let Some(mut queue) = block_on(future::poll_once(&mut task.0)) {
            if let Some(last_instant) = last_frame.last_update() {
                let delta_seconds = last_instant.elapsed().as_secs_f64();
                if delta_seconds > 0.0 {
                    diagnostics.add_measurement(&INFERENCE_FPS, || 1.0 / delta_seconds);
                }
            }
            last_frame.update();

            commands.append(&mut queue);
        }
    }
}

fn depth_image_setup() -> (&'static [u8], TextureFormat, TransferKind, TextureUsages) {
    (
        &[0u8; 16],
        TextureFormat::Rgba32Float,
        TransferKind::Gpu,
        TextureUsages::COPY_SRC
            | TextureUsages::COPY_DST
            | TextureUsages::TEXTURE_BINDING
            | TextureUsages::STORAGE_BINDING,
    )
}

fn setup_ui(
    mut commands: Commands,
    mut depth_texture: ResMut<DepthTexture>,
    mut images: ResMut<Assets<Image>>,
    burn_device: Option<Res<BurnDevice>>,
) {
    if depth_texture.entity.is_some() {
        return;
    }

    let Some(burn_device) = burn_device else {
        return;
    };

    if !burn_device.is_ready() {
        return;
    }

    let size = Extent3d {
        width: depth_texture.width.max(1),
        height: depth_texture.height.max(1),
        depth_or_array_layers: 1,
    };

    let (fill_bytes, texture_format, transfer_kind, texture_usage) = depth_image_setup();
    let mut image = Image::new_fill(
        size,
        TextureDimension::D2,
        fill_bytes,
        texture_format,
        RenderAssetUsages::RENDER_WORLD,
    );
    image.texture_descriptor.usage |= texture_usage;
    depth_texture.image = images.add(image);

    let mut image_entity = None;
    commands
        .spawn(Node {
            display: Display::Grid,
            width: Val::Percent(100.0),
            height: Val::Percent(100.0),
            grid_template_columns: RepeatedGridTrack::flex(1, 1.0),
            grid_template_rows: RepeatedGridTrack::flex(1, 1.0),
            ..default()
        })
        .with_children(|builder| {
            let entity = builder
                .spawn((
                    ImageNode::new(depth_texture.image.clone()).with_mode(NodeImageMode::Stretch),
                    BevyBurnHandle::<Wgpu> {
                        bevy_image: depth_texture.image.clone(),
                        tensor: Tensor::<Wgpu, 3>::zeros(
                            [
                                depth_texture.height.max(1) as usize,
                                depth_texture.width.max(1) as usize,
                                4,
                            ],
                            burn_device.device().unwrap(),
                        ),
                        upload: true,
                        direction: BindingDirection::BurnToBevy,
                        xfer: transfer_kind,
                    },
                ))
                .id();
            image_entity = Some(entity);
        });

    depth_texture.entity = image_entity;

    commands.spawn(Camera2d);
}

pub fn viewer_app(args: BevyBurnDepthConfig) -> App {
    let mut app = App::new();
    app.insert_resource(args.clone());

    let title = "bevy_burn_depth".to_string();

    #[cfg(target_arch = "wasm32")]
    let primary_window = Some(Window {
        canvas: Some("#bevy".to_string()),
        mode: bevy::window::WindowMode::Windowed,
        prevent_default_event_handling: true,
        title: title.clone(),
        #[cfg(feature = "perftest")]
        present_mode: bevy::window::PresentMode::AutoNoVsync,
        #[cfg(not(feature = "perftest"))]
        present_mode: bevy::window::PresentMode::AutoVsync,
        ..default()
    });

    #[cfg(not(target_arch = "wasm32"))]
    let primary_window = Some(Window {
        mode: bevy::window::WindowMode::Windowed,
        prevent_default_event_handling: false,
        resolution: bevy::window::WindowResolution::new(1024, 1024),
        title,
        #[cfg(feature = "perftest")]
        present_mode: bevy::window::PresentMode::AutoNoVsync,
        #[cfg(not(feature = "perftest"))]
        present_mode: bevy::window::PresentMode::AutoVsync,
        ..default()
    });

    app.insert_resource(ClearColor(Color::srgba(0.0, 0.0, 0.0, 0.0)));

    let default_plugins = DefaultPlugins
        .set(ImagePlugin::default_nearest())
        .set(RenderPlugin {
            render_creation: RenderCreation::Automatic(WgpuSettings {
                features: WgpuFeatures::TEXTURE_ADAPTER_SPECIFIC_FORMAT_FEATURES,
                ..Default::default()
            }),
            ..Default::default()
        })
        .set(WindowPlugin {
            primary_window,
            ..default()
        });

    app.add_plugins(default_plugins);
    app.add_plugins(BevyBurnBridgePlugin::<Wgpu>::default());

    if args.press_esc_to_close {
        app.add_systems(Update, press_esc_close);
    }

    if args.show_fps {
        app.add_plugins(FrameTimeDiagnosticsPlugin::default());
        app.register_diagnostic(Diagnostic::new(INFERENCE_FPS));
        app.add_systems(Startup, fps_display_setup);
        app.add_systems(Update, fps_update_system);
    }

    app
}

fn press_esc_close(keys: Res<ButtonInput<KeyCode>>, mut exit: MessageWriter<AppExit>) {
    if keys.just_pressed(KeyCode::Escape) {
        exit.write(AppExit::Success);
    }
}

const INFERENCE_FPS: DiagnosticPath = DiagnosticPath::const_new("inference_fps");

fn fps_display_setup(mut commands: Commands) {
    commands
        .spawn((
            Text("fps: ".to_string()),
            TextFont {
                font_size: 60.0,
                ..Default::default()
            },
            TextColor(Color::WHITE),
            Node {
                position_type: PositionType::Absolute,
                bottom: Val::Px(5.0),
                left: Val::Px(15.0),
                ..default()
            },
            ZIndex(2),
        ))
        .with_child((
            FpsText,
            TextColor(Color::Srgba(GOLD)),
            TextFont {
                font_size: 60.0,
                ..Default::default()
            },
            TextSpan::default(),
        ));
}

#[derive(Component)]
struct FpsText;

fn fps_update_system(
    diagnostics: Res<DiagnosticsStore>,
    mut query: Query<&mut TextSpan, With<FpsText>>,
) {
    for mut text in &mut query {
        if let Some(fps) = diagnostics.get(&INFERENCE_FPS) {
            if let Some(value) = fps.smoothed() {
                **text = format!("{value:.2}");
            }
        }
    }
}

fn run_app(args: BevyBurnDepthConfig) {
    log("running app...");
    log(&format!("{args:?}"));

    let config = DepthAnything3Config::small();

    let static_frame = args.image_path.as_ref().map(|path| {
        image::open(path)
            .unwrap_or_else(|err| panic!("failed to load image `{}`: {err}", path.display()))
            .to_rgb8()
    });
    let static_frame = static_frame.map(Arc::new);

    let mut depth_texture = DepthTexture::default();
    if let Some(frame) = static_frame.as_ref() {
        depth_texture.width = frame.width();
        depth_texture.height = frame.height();
    }

    let mut app = viewer_app(args.clone());

    app.insert_resource(depth_texture);
    app.insert_resource(StaticFrame(static_frame.clone()));
    app.insert_resource(DepthModelState::new(
        args.checkpoint.clone(),
        config,
        args.normalize_relative_depth,
    ));
    app.add_systems(
        Update,
        (
            setup_ui,
            begin_depth_model_load,
            finish_depth_model_load,
            handle_tasks,
            process_frames,
        )
            .chain(),
    );

    log("launching Bevy application...");
    app.run();

    #[cfg(feature = "native")]
    if let Some(sender) = bevy_burn_depth::platform::camera::APP_RUN_SENDER.get() {
        let _ = sender.send(());
    }
}

pub fn log(_message: &str) {
    #[cfg(debug_assertions)]
    #[cfg(target_arch = "wasm32")]
    {
        web_sys::console::log_1(&_message.into());
    }

    #[cfg(debug_assertions)]
    #[cfg(not(target_arch = "wasm32"))]
    {
        println!("{_message}");
    }
}

fn main() {
    #[cfg(feature = "native")]
    {
        let args = parse_args::<BevyBurnDepthConfig>();
        if args.image_path.is_none() {
            std::thread::spawn(bevy_burn_depth::platform::camera::native_camera_thread);
        }
        run_app(args);
    }

    #[cfg(target_arch = "wasm32")]
    {
        let args = parse_args::<BevyBurnDepthConfig>();

        #[cfg(debug_assertions)]
        console_error_panic_hook::set_once();

        run_app(args);
    }
}
