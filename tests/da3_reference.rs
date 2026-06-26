use std::path::Path;

#[test]
fn da3_reference_assets_are_discoverable() {
    if std::env::var_os("BURN_DEPTH_WGPU_CORRECTNESS").is_none()
        && std::env::var_os("BURN_DEPTH_REFERENCE").is_none()
    {
        return;
    }

    let checkpoint = Path::new("models/da3/da3_metric_large.bpk");
    let legacy_checkpoint = Path::new("assets/model/da3_metric_large.mpk");
    let reference = Path::new("assets/image/test_da3_reference.safetensors");

    if !(checkpoint.exists() || legacy_checkpoint.exists()) || !reference.exists() {
        eprintln!("da3_reference skipped: model/reference assets are not present in this checkout");
        return;
    }

    assert!(
        checkpoint.exists() || legacy_checkpoint.exists(),
        "DA3 checkpoint missing"
    );
    assert!(reference.exists(), "DA3 reference tensor missing");
}
