use burn_depth::loader::{
    DepthArtifactManifest, DepthArtifactPart, DepthPrecision, assemble_parts_manifest, sha256_file,
};
use std::{
    fs,
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
    fs::write(&manifest_path, serde_json::to_vec_pretty(&manifest).unwrap()).unwrap();

    let resolved = assemble_parts_manifest(&manifest_path, Some(&dir), &mut None).unwrap();
    assert_eq!(resolved, dir.join("fixture.bpk"));
    assert_eq!(fs::read(resolved).unwrap(), artifact);

    fs::remove_dir_all(dir).unwrap();
}
