use crate::model::DepthModelKind;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::{
    fmt,
    fs::{self, File},
    io::{Read, Write},
    path::{Path, PathBuf},
};

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

#[derive(Clone, Debug)]
pub enum DepthCheckpointSource {
    Local(PathBuf),
    PartsManifest(PathBuf),
    Cdn { base_url: String, manifest: String },
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
        DepthCheckpointSource::Cdn { base_url, manifest } => {
            if !config.allow_download {
                return Err(DepthLoadError::InvalidSource(format!(
                    "downloads disabled for CDN checkpoint {base_url}/{manifest}"
                )));
            }
            Err(DepthLoadError::DownloadUnsupported(
                "CDN loading requires a caller-provided native or wasm fetcher; synchronous downloads are intentionally not built into the core loader".to_string(),
            ))
        }
    }
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
        let part_path = base_dir.join(&part.name);
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
