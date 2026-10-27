from pathlib import Path


ROOT = Path(__file__).parents[2]
STATIC = ROOT / "src" / "robo_visualize" / "web" / "static"


def test_core_static_document_has_viewer_surfaces() -> None:
    html = (STATIC / "index.html").read_text()
    assert 'id="scene"' in html
    assert 'id="dataset-select"' in html
    assert 'id="episode-select"' in html
    assert 'id="timeline"' in html
    assert 'id="play-button"' in html
    assert "Three.js" not in html


def test_core_static_app_has_cddn_and_stl_loaders() -> None:
    app = (STATIC / "app.js").read_text()
    assert "ColladaLoader" in app
    assert "STLLoader" in app
    assert "/api/config" in app
    assert "/api/episode/" in app
    assert "colladaMeshInfo" in app
    assert "THREE.ArrowHelper" in app
    assert "sid-approach-manifold" in app
    assert "predicted-recovery-direction" in app
    assert "...indices.map" in app
    assert "...signals.map" in app


def test_pyproject_declares_instance_scripts() -> None:
    text = (ROOT / "pyproject.toml").read_text()
    assert "robo-visualize-robotwin-web" in text
    assert "robo-visualize-so101-web" in text
    assert "robo-visualize-so101-render" in text
    assert "robo-visualize-piperx-web" in text


def test_so101_assets_are_checked_in() -> None:
    urdf = ROOT / "src" / "robo_visualize" / "arms" / "so101" / "assets" / "so101_new_calib.urdf"
    assert urdf.is_file()
    stl_dir = urdf.parent / "assets"
    assert len(list(stl_dir.glob("*.stl"))) >= 10


def test_piperx_assets_have_complete_verified_sources() -> None:
    import hashlib
    import json
    package = ROOT / "src/robo_visualize"
    manifest = json.loads((package / "asset-manifest.json").read_text())
    entries = manifest["piperx"]
    assert len(entries) == 12
    assert len({entry["path"] for entry in entries}) == len(entries)
    for entry in entries:
        assert len(entry["sha256"]) == 64
        if "bundled" in entry:
            assert hashlib.sha256((package / entry["bundled"]).read_bytes()).hexdigest() == entry["sha256"]
        else:
            assert entry["url"].startswith("https://raw.githubusercontent.com/")
            assert len(entry["url"].split("/")[5]) == 40
