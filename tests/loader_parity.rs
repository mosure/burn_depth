use burn_depth::loader::{
    DEFAULT_CDN_BASE_URL, DepthArtifactManifest, DepthArtifactPart, DepthPrecision,
    assemble_parts_manifest, cdn_manifest_url, resolve_checkpoint, sha256_file,
};
use burn_depth::{DepthCheckpointSource, DepthLoadConfig, DepthModelKind};
use sha2::Digest;
use std::{
    collections::BTreeMap,
    fs,
    io::{Read, Write},
    net::TcpListener,
    thread,
    time::{SystemTime, UNIX_EPOCH},
};

#[test]
fn sharded_load_reconstructs_single_artifact() {
    let dir = std::env::temp_dir().join(format!(
        "burn_depth_loader_parity_{}",
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    fs::create_dir_all(&dir).unwrap();

    let artifact = b"burn-depth-sharded-artifact";
    let part0 = &artifact[..10];
    let part1 = &artifact[10..];
    let part0_path = dir.join("fixture.part-00000.bpk");
    let part1_path = dir.join("fixture.part-00001.bpk");
    fs::write(&part0_path, part0).unwrap();
    fs::write(&part1_path, part1).unwrap();

    let single_path = dir.join("fixture.bpk.single");
    fs::write(&single_path, artifact).unwrap();
    let part0_sha = sha256_file(&part0_path).unwrap();
    let part1_sha = sha256_file(&part1_path).unwrap();
    let artifact_sha = sha256_file(&single_path).unwrap();

    let manifest = DepthArtifactManifest {
        model_id: "fixture".to_string(),
        model_family: "test".to_string(),
        precision: DepthPrecision::F32,
        source_checkpoint_hash: Some("fixture-source".to_string()),
        source_upstream: Some("fixture-upstream".to_string()),
        burn_version: "0.21.0".to_string(),
        importer_version: env!("CARGO_PKG_VERSION").to_string(),
        artifact_sha256: artifact_sha,
        parts: vec![
            DepthArtifactPart {
                name: "fixture.part-00000.bpk".to_string(),
                byte_length: part0.len() as u64,
                sha256: part0_sha,
            },
            DepthArtifactPart {
                name: "fixture.part-00001.bpk".to_string(),
                byte_length: part1.len() as u64,
                sha256: part1_sha,
            },
        ],
        tensor_count: Some(0),
        total_bytes: artifact.len() as u64,
        created_timestamp: "1970-01-01T00:00:00Z".to_string(),
    };
    let manifest_path = dir.join("fixture.bpk.parts.json");
    fs::write(
        &manifest_path,
        serde_json::to_vec_pretty(&manifest).unwrap(),
    )
    .unwrap();

    let resolved = assemble_parts_manifest(&manifest_path, Some(&dir), &mut None).unwrap();
    assert_eq!(resolved, dir.join("fixture.bpk"));
    assert_eq!(fs::read(resolved).unwrap(), artifact);

    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn default_cdn_urls_match_artifact_layout() {
    assert_eq!(
        DepthModelKind::DepthPro.default_cdn_manifest(DepthPrecision::F32),
        "depth-pro/depth_pro.bpk.parts.json"
    );
    assert_eq!(
        DepthModelKind::DepthPro.default_cdn_manifest(DepthPrecision::F16),
        "depth-pro/depth_pro_f16.bpk.parts.json"
    );
    assert_eq!(
        DepthModelKind::DepthAnything3MetricLarge.default_cdn_manifest(DepthPrecision::F32),
        "da3/da3_metric_large.bpk.parts.json"
    );
    assert_eq!(
        DepthModelKind::DepthAnything3MetricLarge.default_cdn_manifest(DepthPrecision::F16),
        "da3/da3_metric_large_f16.bpk.parts.json"
    );

    let source = DepthCheckpointSource::default_cdn(
        DepthModelKind::DepthAnything3MetricLarge,
        DepthPrecision::F16,
    );
    let DepthCheckpointSource::Cdn { base_url, manifest } = source else {
        panic!("default checkpoint should be CDN-backed");
    };
    assert_eq!(base_url, DEFAULT_CDN_BASE_URL);
    assert_eq!(
        cdn_manifest_url(&base_url, &manifest),
        "https://aberration.technology/model/burn_depth/da3/da3_metric_large_f16.bpk.parts.json"
    );
}

#[test]
#[cfg(not(target_arch = "wasm32"))]
fn cdn_load_downloads_parts_and_reuses_cache() {
    let dir = std::env::temp_dir().join(format!(
        "burn_depth_cdn_loader_{}",
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    let cache_dir = dir.join("cache");
    fs::create_dir_all(&cache_dir).unwrap();

    let artifact = b"burn-depth-cdn-sharded-artifact";
    let part0 = &artifact[..12];
    let part1 = &artifact[12..];
    let manifest_rel = "da3/fixture.bpk.parts.json";
    let manifest = DepthArtifactManifest {
        model_id: "fixture".to_string(),
        model_family: "test".to_string(),
        precision: DepthPrecision::F32,
        source_checkpoint_hash: None,
        source_upstream: None,
        burn_version: "0.21.0".to_string(),
        importer_version: env!("CARGO_PKG_VERSION").to_string(),
        artifact_sha256: sha256_bytes(artifact),
        parts: vec![
            DepthArtifactPart {
                name: "fixture.part-00000.bpk".to_string(),
                byte_length: part0.len() as u64,
                sha256: sha256_bytes(part0),
            },
            DepthArtifactPart {
                name: "fixture.part-00001.bpk".to_string(),
                byte_length: part1.len() as u64,
                sha256: sha256_bytes(part1),
            },
        ],
        tensor_count: Some(0),
        total_bytes: artifact.len() as u64,
        created_timestamp: "1970-01-01T00:00:00Z".to_string(),
    };

    let (base_url, server) = start_http_fixture(
        [
            (
                "/da3/fixture.bpk.parts.json",
                serde_json::to_vec_pretty(&manifest).unwrap(),
            ),
            ("/da3/fixture.part-00000.bpk", part0.to_vec()),
            ("/da3/fixture.part-00001.bpk", part1.to_vec()),
        ],
        3,
    );

    let config = DepthLoadConfig {
        model: DepthModelKind::DepthAnything3MetricLarge,
        precision: DepthPrecision::F32,
        checkpoint: DepthCheckpointSource::Cdn {
            base_url,
            manifest: manifest_rel.to_string(),
        },
        cache_dir: Some(cache_dir.clone()),
        allow_download: true,
        require_gpu: false,
    };
    let resolved = resolve_checkpoint(&config, None).unwrap();
    assert_eq!(resolved, cache_dir.join("da3/fixture.bpk"));
    assert_eq!(fs::read(&resolved).unwrap(), artifact);
    server.join().unwrap();

    let cached_config = DepthLoadConfig {
        model: DepthModelKind::DepthAnything3MetricLarge,
        precision: DepthPrecision::F32,
        checkpoint: DepthCheckpointSource::Cdn {
            base_url: "http://127.0.0.1:9".to_string(),
            manifest: manifest_rel.to_string(),
        },
        cache_dir: Some(cache_dir.clone()),
        allow_download: false,
        require_gpu: false,
    };
    let cached = resolve_checkpoint(&cached_config, None).unwrap();
    assert_eq!(cached, resolved);

    fs::remove_dir_all(dir).unwrap();
}

#[cfg(not(target_arch = "wasm32"))]
fn start_http_fixture(
    files: impl IntoIterator<Item = (&'static str, Vec<u8>)>,
    request_count: usize,
) -> (String, thread::JoinHandle<()>) {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let base_url = format!("http://{}", listener.local_addr().unwrap());
    let files = files
        .into_iter()
        .map(|(path, bytes)| (path.to_string(), bytes))
        .collect::<BTreeMap<_, _>>();

    let handle = thread::spawn(move || {
        for _ in 0..request_count {
            let Ok((mut stream, _)) = listener.accept() else {
                break;
            };
            let mut request = [0u8; 4096];
            let read = stream.read(&mut request).unwrap_or(0);
            let request = String::from_utf8_lossy(&request[..read]);
            let path = request
                .lines()
                .next()
                .and_then(|line| line.split_whitespace().nth(1))
                .unwrap_or("/");
            if let Some(bytes) = files.get(path) {
                let header = format!(
                    "HTTP/1.1 200 OK\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
                    bytes.len()
                );
                stream.write_all(header.as_bytes()).unwrap();
                stream.write_all(bytes).unwrap();
            } else {
                stream
                    .write_all(
                        b"HTTP/1.1 404 Not Found\r\nContent-Length: 0\r\nConnection: close\r\n\r\n",
                    )
                    .unwrap();
            }
        }
    });

    (base_url, handle)
}

fn sha256_bytes(bytes: &[u8]) -> String {
    let digest = sha2::Sha256::digest(bytes);
    let mut out = String::with_capacity(digest.len() * 2);
    for byte in digest {
        use std::fmt::Write as _;
        let _ = write!(&mut out, "{byte:02x}");
    }
    out
}
