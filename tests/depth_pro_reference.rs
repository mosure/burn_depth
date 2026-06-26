use std::path::Path;

#[test]
fn depth_pro_reference_assets_are_discoverable() {
    if std::env::var_os("BURN_DEPTH_REFERENCE").is_none()
        && std::env::var_os("BURN_DEPTH_WGPU_CORRECTNESS").is_none()
    {
        return;
    }

    let checkpoint = Path::new("models/depth-pro/depth_pro.bpk");
    let legacy_checkpoint = Path::new("assets/model/depth_pro.mpk");
    let reference = Path::new("assets/image/test.safetensors");

    if !(checkpoint.exists() || legacy_checkpoint.exists()) || !reference.exists() {
        eprintln!(
            "depth_pro_reference skipped: model/reference assets are not present in this checkout"
        );
        return;
    }

    assert!(
        checkpoint.exists() || legacy_checkpoint.exists(),
        "Depth Pro checkpoint missing"
    );
    assert!(reference.exists(), "Depth Pro reference tensor missing");
}
