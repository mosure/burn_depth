use std::{
    error::Error,
    fs::{self, File},
    io::{Read, Write},
    path::{Path, PathBuf},
    time::{SystemTime, UNIX_EPOCH},
};

use burn::{
    module::Module,
    record::{FullPrecisionSettings, HalfPrecisionSettings, NamedMpkFileRecorder},
    tensor::backend::Backend,
};
use burn_depth::loader::{DepthArtifactManifest, DepthArtifactPart, DepthPrecision, sha256_file};
use burn_store::{BurnpackStore, HalfPrecisionAdapter, ModuleSnapshot};
use clap::ValueEnum;
use sha2::{Digest, Sha256};

const BURN_VERSION: &str = "0.21.0";

#[derive(Clone, Copy, Debug, ValueEnum)]
pub enum ImportPrecision {
    F32,
    F16,
}

impl From<ImportPrecision> for DepthPrecision {
    fn from(value: ImportPrecision) -> Self {
        match value {
            ImportPrecision::F32 => DepthPrecision::F32,
            ImportPrecision::F16 => DepthPrecision::F16,
        }
    }
}

pub fn save_import_artifact<B, M>(
    model: &M,
    output: &Path,
    precision: ImportPrecision,
) -> Result<(), Box<dyn Error>>
where
    B: Backend,
    M: Clone + Module<B> + ModuleSnapshot<B>,
{
    if let Some(parent) = output.parent() {
        fs::create_dir_all(parent)?;
    }

    if output
        .extension()
        .and_then(|extension| extension.to_str())
        .is_some_and(|extension| extension.eq_ignore_ascii_case("mpk"))
    {
        save_mpk::<B, M>(model, output, precision)?;
    } else {
        save_burnpack::<B, M>(model, output, precision)?;
    }

    Ok(())
}

pub fn maybe_write_shards(
    output: &Path,
    shard_size_mb: Option<u64>,
    model_id: &str,
    model_family: &str,
    precision: ImportPrecision,
    source_checkpoint: &Path,
    source_upstream: Option<&str>,
    tensor_count: usize,
) -> Result<Option<PathBuf>, Box<dyn Error>> {
    let Some(shard_size_mb) = shard_size_mb else {
        return Ok(None);
    };
    if shard_size_mb == 0 {
        return Err("shard size must be greater than zero".into());
    }

    let shard_bytes = shard_size_mb
        .checked_mul(1024 * 1024)
        .ok_or("shard size overflow")?;
    let base_dir = output
        .parent()
        .ok_or_else(|| format!("output path `{}` has no parent", output.display()))?;
    let stem = output
        .file_stem()
        .and_then(|value| value.to_str())
        .ok_or_else(|| format!("output path `{}` has no UTF-8 file stem", output.display()))?;
    let output_name = output
        .file_name()
        .and_then(|value| value.to_str())
        .ok_or_else(|| format!("output path `{}` has no UTF-8 file name", output.display()))?;
    let manifest_path = output.with_file_name(format!("{output_name}.parts.json"));

    let mut input = File::open(output)?;
    let mut parts = Vec::new();
    let mut full_hasher = Sha256::new();
    let mut buffer = vec![0u8; 1024 * 1024];
    let mut part_index = 0usize;

    loop {
        let part_name = format!("{stem}.part-{part_index:05}.bpk");
        let part_path = base_dir.join(&part_name);
        let mut part_file = File::create(&part_path)?;
        let mut part_hasher = Sha256::new();
        let mut part_len = 0u64;

        while part_len < shard_bytes {
            let remaining = (shard_bytes - part_len) as usize;
            let read_size = buffer.len().min(remaining);
            let read = input.read(&mut buffer[..read_size])?;
            if read == 0 {
                break;
            }
            let bytes = &buffer[..read];
            part_file.write_all(bytes)?;
            part_hasher.update(bytes);
            full_hasher.update(bytes);
            part_len += read as u64;
        }

        if part_len == 0 {
            let _ = fs::remove_file(&part_path);
            break;
        }

        part_file.flush()?;
        parts.push(DepthArtifactPart {
            name: part_name,
            byte_length: part_len,
            sha256: hex_sha256(part_hasher.finalize().as_slice()),
        });
        part_index += 1;
    }

    let total_bytes = output.metadata()?.len();
    let manifest = DepthArtifactManifest {
        model_id: model_id.to_string(),
        model_family: model_family.to_string(),
        precision: precision.into(),
        source_checkpoint_hash: if source_checkpoint.exists() {
            Some(sha256_file(source_checkpoint)?)
        } else {
            None
        },
        source_upstream: source_upstream.map(ToOwned::to_owned),
        burn_version: BURN_VERSION.to_string(),
        importer_version: env!("CARGO_PKG_VERSION").to_string(),
        artifact_sha256: hex_sha256(full_hasher.finalize().as_slice()),
        parts,
        tensor_count: Some(tensor_count),
        total_bytes,
        created_timestamp: created_timestamp(),
    };
    fs::write(&manifest_path, serde_json::to_vec_pretty(&manifest)?)?;
    Ok(Some(manifest_path))
}

fn save_burnpack<B, M>(
    model: &M,
    output: &Path,
    precision: ImportPrecision,
) -> Result<(), Box<dyn Error>>
where
    B: Backend,
    M: ModuleSnapshot<B>,
{
    let mut store = BurnpackStore::from_file(output)
        .auto_extension(false)
        .overwrite(true);
    if matches!(precision, ImportPrecision::F16) {
        store = store.with_to_adapter(HalfPrecisionAdapter::new());
    }
    model.save_into(&mut store)?;
    Ok(())
}

fn save_mpk<B, M>(
    model: &M,
    output: &Path,
    precision: ImportPrecision,
) -> Result<(), Box<dyn Error>>
where
    B: Backend,
    M: Clone + Module<B>,
{
    match precision {
        ImportPrecision::F32 => {
            model.clone().save_file(
                output.to_path_buf(),
                &NamedMpkFileRecorder::<FullPrecisionSettings>::new(),
            )?;
        }
        ImportPrecision::F16 => {
            model.clone().save_file(
                output.to_path_buf(),
                &NamedMpkFileRecorder::<HalfPrecisionSettings>::new(),
            )?;
        }
    }
    Ok(())
}

fn created_timestamp() -> String {
    let seconds = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_secs())
        .unwrap_or_default();
    format!("unix:{seconds}")
}

fn hex_sha256(bytes: &[u8]) -> String {
    let mut out = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        use std::fmt::Write as _;
        let _ = write!(&mut out, "{byte:02x}");
    }
    out
}
