import hashlib
import io
import json
from pathlib import Path

import pytest
from robo_visualize import assets


def test_download_verifies_content_reuses_cache_and_invalidates_bad_receipt(tmp_path, monkeypatch):
    package = tmp_path / 'package'
    package.mkdir()
    payload = b'verified mesh data'
    manifest = {'piperx': [{'path': 'meshes/test.STL', 'url': 'https://example.test/mesh',
                           'sha256': hashlib.sha256(payload).hexdigest()}]}
    (package / 'asset-manifest.json').write_text(json.dumps(manifest))
    monkeypatch.setattr(assets, 'files', lambda name: package)
    calls = []
    def download(url, timeout):
        calls.append(url)
        return io.BytesIO(payload)
    monkeypatch.setattr(assets, 'urlopen', download)
    output = tmp_path / 'output'
    assets.install(output)
    assets.install(output)
    assert len(calls) == 1
    assert (output / 'verified-assets.json').is_file()
    (output / 'meshes/test.STL').write_bytes(b'corrupt')
    with pytest.raises(ValueError, match='SHA-256 mismatch'):
        assets.install(output)
    assert not (output / 'verified-assets.json').exists()


def test_bad_download_never_publishes_asset_or_receipt(tmp_path, monkeypatch):
    package = tmp_path / 'package'
    package.mkdir()
    manifest = {'piperx': [{'path': 'mesh.STL', 'url': 'https://example.test/mesh',
                           'sha256': hashlib.sha256(b'expected').hexdigest()}]}
    (package / 'asset-manifest.json').write_text(json.dumps(manifest))
    monkeypatch.setattr(assets, 'files', lambda name: package)
    monkeypatch.setattr(assets, 'urlopen', lambda url, timeout: io.BytesIO(b'wrong'))
    output = tmp_path / 'output'
    with pytest.raises(ValueError, match='SHA-256 mismatch'):
        assets.install(output)
    assert not (output / 'mesh.STL').exists()
    assert not (output / 'verified-assets.json').exists()
