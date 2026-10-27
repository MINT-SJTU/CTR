"""Install revision-pinned PiperX geometry with SHA-256 verification."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from importlib.resources import files
from pathlib import Path
from tempfile import TemporaryDirectory
from urllib.request import urlopen


def assets_root() -> Path:
    return Path(os.environ.get('CTR_PIPERX_ASSETS', '~/.cache/ctr/piperx')).expanduser().resolve()


def install(output: Path) -> dict:
    root = output.expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    receipt = root / 'verified-assets.json'
    receipt.unlink(missing_ok=True)
    package = files('robo_visualize')
    manifest = json.loads((package / 'asset-manifest.json').read_text())
    for entry in manifest['piperx']:
        destination = root / entry['path']
        if not destination.resolve().is_relative_to(root):
            raise ValueError(f"asset path escapes destination: {entry['path']}")
        if destination.exists():
            content = destination.read_bytes()
        elif 'bundled' in entry:
            content = (package / entry['bundled']).read_bytes()
        else:
            with urlopen(entry['url'], timeout=60) as response:
                content = response.read()
        if hashlib.sha256(content).hexdigest() != entry['sha256']:
            raise ValueError(f"asset SHA-256 mismatch: {entry['path']}")
        if not destination.exists():
            destination.parent.mkdir(parents=True, exist_ok=True)
            with TemporaryDirectory(dir=destination.parent) as temporary:
                staged = Path(temporary) / 'asset'
                staged.write_bytes(content)
                staged.replace(destination)
    report = {'schema': 1, 'files': manifest['piperx']}
    receipt.write_text(json.dumps(report, indent=2) + '\n')
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=assets_root())
    args = parser.parse_args()
    report = install(args.output)
    print(json.dumps({'root': str(args.output.expanduser().resolve()), 'verified_files': len(report['files'])}))


if __name__ == '__main__':
    main()
