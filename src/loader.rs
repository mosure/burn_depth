use crate::model::DepthModelKind;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
#[cfg(not(target_arch = "wasm32"))]
use std::time::{Duration, SystemTime, UNIX_EPOCH};
use std::{
    fmt,
    fs::{self, File},
    io::{Read, Write},
    path::{Component, Path, PathBuf},
};

pub const DEFAULT_CDN_BASE_URL: &str = "https://aberration.technology/model/burn_depth";
const DEFAULT_CACHE_DIR: &str = ".burn_depth";
#[cfg(not(target_arch = "wasm32"))]
const DOWNLOAD_MAX_ATTEMPTS: u32 = 4;
#[cfg(not(target_arch = "wasm32"))]
const DOWNLOAD_RETRY_BASE_DELAY_MS: u64 = 500;
#[cfg(not(target_arch = "wasm32"))]
const DOWNLOAD_CONNECT_TIMEOUT_SECS: u64 = 20;
#[cfg(not(target_arch = "wasm32"))]
const DOWNLOAD_READ_TIMEOUT_SECS: u64 = 60;
#[cfg(not(target_arch = "wasm32"))]
const DOWNLOAD_WRITE_TIMEOUT_SECS: u64 = 60;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum DepthPrecision {
    F32,
    F16,
}

#[derive(Clone, Debug)]
pub struct DepthLoadConfig {
    pub model: DepthModelKind,
    pub precision: DepthPrecision,
    pub checkpoint: DepthCheckpointSource,
    pub cache_dir: Option<PathBuf>,
    pub allow_download: bool,
    pub require_gpu: bool,
}

impl DepthLoadConfig {
    pub fn cdn(model: DepthModelKind, precision: DepthPrecision) -> Self {
        Self {
            model,
            precision,
            checkpoint: DepthCheckpointSource::default_cdn(model, precision),
            cache_dir: None,
            allow_download: true,
            require_gpu: true,
        }
    }
}

#[derive(Clone, Debug)]
pub enum DepthCheckpointSource {
    Local(PathBuf),
    PartsManifest(PathBuf),
    Cdn { base_url: String, manifest: String },
}

impl DepthCheckpointSource {
    pub fn default_cdn(model: DepthModelKind, precision: DepthPrecision) -> Self {
        Self::Cdn {
            base_url: default_cdn_base_url(),
            manifest: model.default_cdn_manifest(precision).to_string(),
        }
    }
}

impl DepthModelKind {
    pub fn default_cdn_manifest(self, precision: DepthPrecision) -> &'static str {
        match (self, precision) {
            (DepthModelKind::DepthPro, DepthPrecision::F32) => "depth-pro/depth_pro.bpk.parts.json",
            (DepthModelKind::DepthPro, DepthPrecision::F16) => {
                "depth-pro/depth_pro_f16.bpk.parts.json"
            }
            (
                DepthModelKind::DepthAnything3MetricLarge | DepthModelKind::DepthAnything3,
                DepthPrecision::F32,
            ) => "da3/da3_metric_large.bpk.parts.json",
            (
                DepthModelKind::DepthAnything3MetricLarge | DepthModelKind::DepthAnything3,
                DepthPrecision::F16,
            ) => "da3/da3_metric_large_f16.bpk.parts.json",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DepthLoadStage {
    Manifest,
    CacheHit,
    CacheMiss,
    Part,
    Verify,
    Deserialize,
    ModelReady,
}

#[derive(Clone, Debug)]
pub struct DepthLoadEvent {
    pub stage: DepthLoadStage,
    pub message: String,
    pub current: Option<usize>,
    pub total: Option<usize>,
}

impl DepthLoadEvent {
    pub fn new(stage: DepthLoadStage, message: impl Into<String>) -> Self {
        Self {
            stage,
            message: message.into(),
            current: None,
            total: None,
        }
    }

    pub fn progress(
        stage: DepthLoadStage,
        message: impl Into<String>,
        current: usize,
        total: usize,
    ) -> Self {
        Self {
            stage,
            message: message.into(),
            current: Some(current),
            total: Some(total),
        }
    }
}

#[derive(Debug)]
pub enum DepthLoadError {
    Io(std::io::Error),
    Manifest(serde_json::Error),
    MissingParent(PathBuf),
    HashMismatch {
        path: PathBuf,
        expected: String,
        actual: String,
    },
    LengthMismatch {
        path: PathBuf,
        expected: u64,
        actual: u64,
    },
    Download(String),
    DownloadUnsupported(String),
    InvalidSource(String),
}

impl fmt::Display for DepthLoadError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io(err) => write!(f, "io error: {err}"),
            Self::Manifest(err) => write!(f, "manifest parse error: {err}"),
            Self::MissingParent(path) => {
                write!(
                    f,
                    "path `{}` does not have a parent directory",
                    path.display()
                )
            }
            Self::HashMismatch {
                path,
                expected,
                actual,
            } => write!(
                f,
                "sha256 mismatch for `{}`: expected {expected}, got {actual}",
                path.display()
            ),
            Self::LengthMismatch {
                path,
                expected,
                actual,
            } => write!(
                f,
                "length mismatch for `{}`: expected {expected} bytes, got {actual}",
                path.display()
            ),
            Self::Download(message) => write!(f, "{message}"),
            Self::DownloadUnsupported(message) => write!(f, "{message}"),
            Self::InvalidSource(message) => write!(f, "{message}"),
        }
    }
}

impl std::error::Error for DepthLoadError {}

impl From<std::io::Error> for DepthLoadError {
    fn from(value: std::io::Error) -> Self {
        Self::Io(value)
    }
}

impl From<serde_json::Error> for DepthLoadError {
    fn from(value: serde_json::Error) -> Self {
        Self::Manifest(value)
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct DepthArtifactManifest {
    pub model_id: String,
    pub model_family: String,
    pub precision: DepthPrecision,
    pub source_checkpoint_hash: Option<String>,
    pub source_upstream: Option<String>,
    pub burn_version: String,
    pub importer_version: String,
    pub artifact_sha256: String,
    pub parts: Vec<DepthArtifactPart>,
    pub tensor_count: Option<usize>,
    pub total_bytes: u64,
    pub created_timestamp: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct DepthArtifactPart {
    pub name: String,
    pub byte_length: u64,
    pub sha256: String,
}

pub fn read_parts_manifest(
    path: impl AsRef<Path>,
) -> Result<DepthArtifactManifest, DepthLoadError> {
    let bytes = fs::read(path)?;
    Ok(serde_json::from_slice(&bytes)?)
}

pub fn resolve_checkpoint(
    config: &DepthLoadConfig,
    mut progress: Option<&mut dyn FnMut(DepthLoadEvent)>,
) -> Result<PathBuf, DepthLoadError> {
    match &config.checkpoint {
        DepthCheckpointSource::Local(path) => Ok(path.clone()),
        DepthCheckpointSource::PartsManifest(path) => {
            assemble_parts_manifest(path, config.cache_dir.as_deref(), &mut progress)
        }
        DepthCheckpointSource::Cdn { base_url, manifest } => resolve_cdn_checkpoint(
            base_url,
            manifest,
            config.cache_dir.as_deref(),
            config.allow_download,
            &mut progress,
        ),
    }
}

pub fn default_cdn_base_url() -> String {
    option_env!("BURN_DEPTH_MODEL_BASE_URL")
        .unwrap_or(DEFAULT_CDN_BASE_URL)
        .trim_end_matches('/')
        .to_string()
}

pub fn default_cache_dir() -> PathBuf {
    if let Some(home) = user_home_dir() {
        home.join(DEFAULT_CACHE_DIR)
    } else {
        PathBuf::from(DEFAULT_CACHE_DIR)
    }
}

pub fn cdn_manifest_url(base_url: &str, manifest: &str) -> String {
    if manifest.contains("://") {
        return manifest.to_string();
    }
    join_url(base_url, manifest)
}

pub fn resolve_cdn_checkpoint(
    base_url: &str,
    manifest: &str,
    cache_dir: Option<&Path>,
    allow_download: bool,
    progress: &mut Option<&mut dyn FnMut(DepthLoadEvent)>,
) -> Result<PathBuf, DepthLoadError> {
    let cache_root = cache_dir
        .map(Path::to_path_buf)
        .unwrap_or_else(default_cache_dir);
    let local_manifest_path = cache_root.join(safe_relative_path(manifest));

    if local_manifest_path.exists() {
        match assemble_parts_manifest(&local_manifest_path, None, progress) {
            Ok(path) => return Ok(path),
            Err(err) => {
                emit(
                    progress,
                    DepthLoadEvent::new(
                        DepthLoadStage::CacheMiss,
                        format!("cached CDN artifact incomplete: {err}"),
                    ),
                );
                if !allow_download {
                    return Err(err);
                }
            }
        }
    } else if !allow_download {
        return Err(DepthLoadError::InvalidSource(format!(
            "downloads disabled and cached CDN manifest is missing: {}",
            local_manifest_path.display()
        )));
    }

    if !allow_download {
        return Err(DepthLoadError::InvalidSource(
            "downloads disabled for CDN checkpoint".to_string(),
        ));
    }

    download_cdn_checkpoint(base_url, manifest, &local_manifest_path, progress)?;
    assemble_parts_manifest(&local_manifest_path, None, progress)
}

#[cfg(target_arch = "wasm32")]
fn download_cdn_checkpoint(
    base_url: &str,
    manifest: &str,
    _local_manifest_path: &Path,
    _progress: &mut Option<&mut dyn FnMut(DepthLoadEvent)>,
) -> Result<(), DepthLoadError> {
    Err(DepthLoadError::DownloadUnsupported(format!(
        "CDN checkpoint {}/{} requires async fetch integration on wasm; synchronous XHR is not used",
        base_url.trim_end_matches('/'),
        manifest.trim_start_matches('/')
    )))
}

#[cfg(not(target_arch = "wasm32"))]
fn download_cdn_checkpoint(
    base_url: &str,
    manifest: &str,
    local_manifest_path: &Path,
    progress: &mut Option<&mut dyn FnMut(DepthLoadEvent)>,
) -> Result<(), DepthLoadError> {
    let manifest_url = cdn_manifest_url(base_url, manifest);
    emit(
        progress,
        DepthLoadEvent::new(
            DepthLoadStage::Manifest,
            format!("downloading {manifest_url}"),
        ),
    );
    let manifest_bytes = download_bytes_with_retries(&manifest_url)?;
    let artifact_manifest: DepthArtifactManifest = serde_json::from_slice(&manifest_bytes)?;
    if artifact_manifest.parts.is_empty() {
        return Err(DepthLoadError::InvalidSource(format!(
            "CDN manifest {manifest_url} has no parts"
        )));
    }
    write_file_atomically(local_manifest_path, &manifest_bytes)?;

    for (index, part) in artifact_manifest.parts.iter().enumerate() {
        let part_path = resolve_part_entry_path(local_manifest_path, &part.name)?;
        match verify_file(&part_path, Some(part.byte_length), Some(&part.sha256)) {
            Ok(()) => {
                emit(
                    progress,
                    DepthLoadEvent::progress(
                        DepthLoadStage::CacheHit,
                        format!("using cached {}", part_path.display()),
                        index + 1,
                        artifact_manifest.parts.len(),
                    ),
                );
                continue;
            }
            Err(err) if part_path.exists() => {
                emit(
                    progress,
                    DepthLoadEvent::progress(
                        DepthLoadStage::CacheMiss,
                        format!("discarding corrupt cached part: {err}"),
                        index + 1,
                        artifact_manifest.parts.len(),
                    ),
                );
                let _ = fs::remove_file(&part_path);
            }
            Err(_) => {
                emit(
                    progress,
                    DepthLoadEvent::progress(
                        DepthLoadStage::CacheMiss,
                        format!("missing {}", part_path.display()),
                        index + 1,
                        artifact_manifest.parts.len(),
                    ),
                );
            }
        }

        let part_url = resolve_manifest_entry_url(&manifest_url, &part.name);
        emit(
            progress,
            DepthLoadEvent::progress(
                DepthLoadStage::Part,
                format!("downloading {part_url}"),
                index + 1,
                artifact_manifest.parts.len(),
            ),
        );
        download_part_file(&part_url, &part_path, part)?;
    }

    Ok(())
}

pub fn assemble_parts_manifest(
    manifest_path: impl AsRef<Path>,
    cache_dir: Option<&Path>,
    progress: &mut Option<&mut dyn FnMut(DepthLoadEvent)>,
) -> Result<PathBuf, DepthLoadError> {
    let manifest_path = manifest_path.as_ref();
    emit(
        progress,
        DepthLoadEvent::new(
            DepthLoadStage::Manifest,
            format!("reading {}", manifest_path.display()),
        ),
    );
    let manifest = read_parts_manifest(manifest_path)?;
    let base_dir = manifest_path
        .parent()
        .ok_or_else(|| DepthLoadError::MissingParent(manifest_path.to_path_buf()))?;
    let output_name = artifact_name_from_manifest_path(manifest_path)
        .unwrap_or_else(|| format!("{}.bpk", manifest.model_id));
    let output_dir = cache_dir.unwrap_or(base_dir);
    fs::create_dir_all(output_dir)?;
    let output_path = output_dir.join(output_name);

    if output_path.exists() {
        match verify_file(
            &output_path,
            Some(manifest.total_bytes),
            Some(&manifest.artifact_sha256),
        ) {
            Ok(()) => {
                emit(
                    progress,
                    DepthLoadEvent::new(
                        DepthLoadStage::CacheHit,
                        format!("using cached {}", output_path.display()),
                    ),
                );
                return Ok(output_path);
            }
            Err(err) => {
                emit(
                    progress,
                    DepthLoadEvent::new(
                        DepthLoadStage::CacheMiss,
                        format!("discarding corrupt cache entry: {err}"),
                    ),
                );
                let _ = fs::remove_file(&output_path);
            }
        }
    } else {
        emit(
            progress,
            DepthLoadEvent::new(
                DepthLoadStage::CacheMiss,
                format!("assembling {}", output_path.display()),
            ),
        );
    }

    let tmp_path = output_path.with_extension("bpk.partial");
    let mut tmp = File::create(&tmp_path)?;
    let mut full_hasher = Sha256::new();

    for (index, part) in manifest.parts.iter().enumerate() {
        let part_path = resolve_part_entry_path(manifest_path, &part.name)?;
        emit(
            progress,
            DepthLoadEvent::progress(
                DepthLoadStage::Part,
                format!("reading {}", part_path.display()),
                index + 1,
                manifest.parts.len(),
            ),
        );
        let bytes = fs::read(&part_path)?;
        verify_bytes(&part_path, &bytes, part.byte_length, &part.sha256)?;
        full_hasher.update(&bytes);
        tmp.write_all(&bytes)?;
    }
    tmp.flush()?;
    drop(tmp);

    let actual = hex_sha256(full_hasher.finalize().as_slice());
    if actual != normalize_sha256(&manifest.artifact_sha256) {
        let _ = fs::remove_file(&tmp_path);
        return Err(DepthLoadError::HashMismatch {
            path: output_path,
            expected: normalize_sha256(&manifest.artifact_sha256),
            actual,
        });
    }

    emit(
        progress,
        DepthLoadEvent::new(DepthLoadStage::Verify, "verified assembled artifact"),
    );
    fs::rename(&tmp_path, &output_path)?;
    Ok(output_path)
}

pub fn resolve_part_entry_path(
    manifest_path: impl AsRef<Path>,
    entry: &str,
) -> Result<PathBuf, DepthLoadError> {
    let manifest_path = manifest_path.as_ref();
    let base_dir = manifest_path
        .parent()
        .ok_or_else(|| DepthLoadError::MissingParent(manifest_path.to_path_buf()))?;
    Ok(base_dir.join(safe_relative_path(entry)))
}

pub fn verify_file(
    path: impl AsRef<Path>,
    expected_len: Option<u64>,
    expected_sha256: Option<&str>,
) -> Result<(), DepthLoadError> {
    let path = path.as_ref();
    let mut file = File::open(path)?;
    let mut hasher = Sha256::new();
    let mut len = 0u64;
    let mut buf = [0u8; 64 * 1024];
    loop {
        let read = file.read(&mut buf)?;
        if read == 0 {
            break;
        }
        len += read as u64;
        hasher.update(&buf[..read]);
    }
    if let Some(expected) = expected_len {
        if len != expected {
            return Err(DepthLoadError::LengthMismatch {
                path: path.to_path_buf(),
                expected,
                actual: len,
            });
        }
    }
    if let Some(expected) = expected_sha256 {
        let actual = hex_sha256(hasher.finalize().as_slice());
        let expected = normalize_sha256(expected);
        if actual != expected {
            return Err(DepthLoadError::HashMismatch {
                path: path.to_path_buf(),
                expected,
                actual,
            });
        }
    }
    Ok(())
}

pub fn sha256_file(path: impl AsRef<Path>) -> Result<String, DepthLoadError> {
    let path = path.as_ref();
    let mut file = File::open(path)?;
    let mut hasher = Sha256::new();
    let mut buf = [0u8; 64 * 1024];
    loop {
        let read = file.read(&mut buf)?;
        if read == 0 {
            break;
        }
        hasher.update(&buf[..read]);
    }
    Ok(hex_sha256(hasher.finalize().as_slice()))
}

fn verify_bytes(
    path: &Path,
    bytes: &[u8],
    expected_len: u64,
    expected_sha256: &str,
) -> Result<(), DepthLoadError> {
    let actual_len = bytes.len() as u64;
    if actual_len != expected_len {
        return Err(DepthLoadError::LengthMismatch {
            path: path.to_path_buf(),
            expected: expected_len,
            actual: actual_len,
        });
    }
    let actual = hex_sha256(Sha256::digest(bytes).as_slice());
    let expected = normalize_sha256(expected_sha256);
    if actual != expected {
        return Err(DepthLoadError::HashMismatch {
            path: path.to_path_buf(),
            expected,
            actual,
        });
    }
    Ok(())
}

fn artifact_name_from_manifest_path(path: &Path) -> Option<String> {
    let name = path.file_name()?.to_str()?;
    name.strip_suffix(".parts.json").map(ToOwned::to_owned)
}

fn safe_relative_path(value: &str) -> PathBuf {
    let without_fragment = value.split('#').next().unwrap_or(value);
    let without_query = without_fragment
        .split('?')
        .next()
        .unwrap_or(without_fragment);
    let pathish = if let Some((_, after_scheme)) = without_query.split_once("://") {
        after_scheme
            .split_once('/')
            .map(|(_, path)| path)
            .unwrap_or("model.bpk.parts.json")
    } else {
        without_query.trim_start_matches('/')
    };

    let normalized = pathish.replace('\\', "/");
    let mut out = PathBuf::new();
    for component in Path::new(&normalized).components() {
        if let Component::Normal(value) = component {
            out.push(value);
        }
    }
    if out.as_os_str().is_empty() {
        out.push("model.bpk.parts.json");
    }
    out
}

fn join_url(root: &str, rel: &str) -> String {
    let mut out = root.trim_end_matches('/').to_string();
    out.push('/');
    out.push_str(rel.trim_start_matches('/'));
    out
}

#[cfg(not(target_arch = "wasm32"))]
fn resolve_manifest_entry_url(manifest_url: &str, entry_url: &str) -> String {
    if entry_url.contains("://") || entry_url.starts_with('/') {
        return entry_url.to_string();
    }
    let normalized = entry_url.replace('\\', "/");
    if let Some((parent, _)) = manifest_url.rsplit_once('/') {
        return format!("{}/{}", parent.trim_end_matches('/'), normalized);
    }
    normalized
}

fn user_home_dir() -> Option<PathBuf> {
    if let Some(home) = std::env::var_os("HOME").map(PathBuf::from) {
        return Some(home);
    }
    #[cfg(target_os = "windows")]
    {
        if let Some(profile) = std::env::var_os("USERPROFILE").map(PathBuf::from) {
            return Some(profile);
        }
        let drive = std::env::var_os("HOMEDRIVE");
        let path = std::env::var_os("HOMEPATH");
        if let (Some(drive), Some(path)) = (drive, path) {
            return Some(PathBuf::from(format!(
                "{}{}",
                drive.to_string_lossy(),
                path.to_string_lossy()
            )));
        }
    }
    None
}

#[cfg(not(target_arch = "wasm32"))]
fn download_bytes_with_retries(url: &str) -> Result<Vec<u8>, DepthLoadError> {
    let mut last_error = None;
    for attempt in 1..=DOWNLOAD_MAX_ATTEMPTS {
        match download_bytes_once(url) {
            Ok(bytes) => return Ok(bytes),
            Err(err) => {
                if attempt == DOWNLOAD_MAX_ATTEMPTS {
                    return Err(err);
                }
                last_error = Some(err);
                std::thread::sleep(retry_delay(attempt));
            }
        }
    }
    Err(last_error.unwrap_or_else(|| {
        DepthLoadError::Download(format!("failed downloading {url}: unknown error"))
    }))
}

#[cfg(not(target_arch = "wasm32"))]
fn download_bytes_once(url: &str) -> Result<Vec<u8>, DepthLoadError> {
    let response = http_agent()
        .get(url)
        .call()
        .map_err(|err| DepthLoadError::Download(format_download_error(url, err)))?;
    let mut reader = response.into_reader();
    let mut bytes = Vec::new();
    reader
        .read_to_end(&mut bytes)
        .map_err(|err| DepthLoadError::Download(format!("failed to read {url}: {err}")))?;
    Ok(bytes)
}

#[cfg(not(target_arch = "wasm32"))]
fn download_part_file(
    url: &str,
    destination: &Path,
    part: &DepthArtifactPart,
) -> Result<(), DepthLoadError> {
    if let Some(parent) = destination.parent() {
        fs::create_dir_all(parent)?;
    }
    let partial_path = partial_download_path(destination);
    if partial_path.exists() {
        let _ = fs::remove_file(&partial_path);
    }

    let response = http_agent()
        .get(url)
        .call()
        .map_err(|err| DepthLoadError::Download(format_download_error(url, err)))?;
    let mut reader = response.into_reader();
    let mut writer = File::create(&partial_path)?;
    let mut hasher = Sha256::new();
    let mut len = 0u64;
    let mut buffer = vec![0u8; 1024 * 1024];
    loop {
        let read = reader
            .read(&mut buffer)
            .map_err(|err| DepthLoadError::Download(format!("failed to read {url}: {err}")))?;
        if read == 0 {
            break;
        }
        writer.write_all(&buffer[..read])?;
        hasher.update(&buffer[..read]);
        len += read as u64;
    }
    writer.flush()?;
    drop(writer);

    if len != part.byte_length {
        let _ = fs::remove_file(&partial_path);
        return Err(DepthLoadError::LengthMismatch {
            path: destination.to_path_buf(),
            expected: part.byte_length,
            actual: len,
        });
    }
    let actual = hex_sha256(hasher.finalize().as_slice());
    let expected = normalize_sha256(&part.sha256);
    if actual != expected {
        let _ = fs::remove_file(&partial_path);
        return Err(DepthLoadError::HashMismatch {
            path: destination.to_path_buf(),
            expected,
            actual,
        });
    }
    if destination.exists() {
        fs::remove_file(destination)?;
    }
    fs::rename(partial_path, destination)?;
    Ok(())
}

#[cfg(not(target_arch = "wasm32"))]
fn write_file_atomically(path: &Path, bytes: &[u8]) -> Result<(), DepthLoadError> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    let temp_path = temp_download_path(path);
    fs::write(&temp_path, bytes)?;
    if path.exists() {
        fs::remove_file(path)?;
    }
    fs::rename(temp_path, path)?;
    Ok(())
}

#[cfg(not(target_arch = "wasm32"))]
fn http_agent() -> ureq::Agent {
    ureq::AgentBuilder::new()
        .timeout_connect(Duration::from_secs(DOWNLOAD_CONNECT_TIMEOUT_SECS))
        .timeout_read(Duration::from_secs(DOWNLOAD_READ_TIMEOUT_SECS))
        .timeout_write(Duration::from_secs(DOWNLOAD_WRITE_TIMEOUT_SECS))
        .build()
}

#[cfg(not(target_arch = "wasm32"))]
fn retry_delay(attempt: u32) -> Duration {
    let exponent = attempt.saturating_sub(1).min(6);
    let factor = 1u64 << exponent;
    Duration::from_millis(DOWNLOAD_RETRY_BASE_DELAY_MS.saturating_mul(factor))
}

#[cfg(not(target_arch = "wasm32"))]
fn partial_download_path(path: &Path) -> PathBuf {
    let file_name = path
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or("download.bin");
    path.with_file_name(format!("{file_name}.partial"))
}

#[cfg(not(target_arch = "wasm32"))]
fn temp_download_path(path: &Path) -> PathBuf {
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|value| value.as_nanos())
        .unwrap_or(0);
    let file_name = path
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or("download.bin");
    path.with_file_name(format!("{file_name}.download-{nanos}.tmp"))
}

#[cfg(not(target_arch = "wasm32"))]
fn format_download_error(url: &str, err: ureq::Error) -> String {
    match err {
        ureq::Error::Status(code, response) => {
            format!("HTTP {code} ({}) for {url}", response.status_text())
        }
        ureq::Error::Transport(transport) => {
            format!("transport error while downloading {url}: {transport}")
        }
    }
}

fn normalize_sha256(value: &str) -> String {
    value
        .strip_prefix("sha256:")
        .unwrap_or(value)
        .to_ascii_lowercase()
}

fn hex_sha256(bytes: &[u8]) -> String {
    let mut out = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        use std::fmt::Write as _;
        let _ = write!(&mut out, "{byte:02x}");
    }
    out
}

fn emit(progress: &mut Option<&mut dyn FnMut(DepthLoadEvent)>, event: DepthLoadEvent) {
    if let Some(callback) = progress.as_deref_mut() {
        callback(event);
    }
}
