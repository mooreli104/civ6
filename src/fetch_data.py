"""Stage 1 - FETCH.

Downloads the Civilization VI roguelike game package (`civ6-roguelike-ai`) straight
from the public npm registry. The package ships the full TypeScript source of a
hex-grid, Civ-6-style city-builder designed for AI agents: terrain yields, district
adjacency rules, improvements, a six-tree tech system and a headless game engine
with a `getValidActions()` / `act()` API.

That package is both our *data source* (the game's rule tables) and our
*environment* (the engine we roll episodes out in).

Output:
    data/raw/<pkg>-<version>.tgz     the exact artifact that was downloaded
    vendor/civ6-roguelike-ai/        extracted sources (used by the Node bridge)
    data/raw/fetch_manifest.json     provenance: url, version, sha256, file list
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import tarfile
import time
from pathlib import Path

import requests

REGISTRY = "https://registry.npmjs.org"
PACKAGE = "civ6-roguelike-ai"

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
VENDOR_DIR = ROOT / "vendor"


def _get(url: str, *, retries: int = 4, stream: bool = False) -> requests.Response:
    """GET with exponential backoff, because registries hiccup."""
    delay = 2.0
    last: Exception | None = None
    for attempt in range(retries):
        try:
            resp = requests.get(url, timeout=60, stream=stream)
            resp.raise_for_status()
            return resp
        except Exception as exc:  # noqa: BLE001 - retry on anything transient
            last = exc
            if attempt == retries - 1:
                break
            print(f"  ! {type(exc).__name__}: {exc} - retrying in {delay:.0f}s")
            time.sleep(delay)
            delay *= 2
    raise RuntimeError(f"failed to fetch {url}") from last


def fetch_package(package: str = PACKAGE, version: str | None = None) -> dict:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    VENDOR_DIR.mkdir(parents=True, exist_ok=True)

    print(f"[fetch] querying {REGISTRY}/{package}")
    meta = _get(f"{REGISTRY}/{package}").json()
    version = version or meta["dist-tags"]["latest"]
    if version not in meta["versions"]:
        raise SystemExit(f"version {version} not published; have {list(meta['versions'])}")

    dist = meta["versions"][version]["dist"]
    tarball_url = dist["tarball"]
    print(f"[fetch] {package}@{version} -> {tarball_url}")

    tgz_path = RAW_DIR / f"{package}-{version}.tgz"
    resp = _get(tarball_url, stream=True)
    with tgz_path.open("wb") as fh:
        for chunk in resp.iter_content(chunk_size=1 << 16):
            fh.write(chunk)

    sha256 = hashlib.sha256(tgz_path.read_bytes()).hexdigest()
    print(f"[fetch] {tgz_path.name} {tgz_path.stat().st_size / 1024:.1f} KiB sha256={sha256[:16]}...")

    target = VENDOR_DIR / package
    if target.exists():
        shutil.rmtree(target)
    staging = VENDOR_DIR / f".staging-{package}"
    if staging.exists():
        shutil.rmtree(staging)

    with tarfile.open(tgz_path) as tar:
        members = [m for m in tar.getmembers() if _safe_member(m.name)]
        tar.extractall(staging, members=members)  # noqa: S202 - members filtered above
    # npm tarballs nest everything under "package/"
    shutil.move(str(staging / "package"), str(target))
    shutil.rmtree(staging, ignore_errors=True)

    files = sorted(str(p.relative_to(target)) for p in target.rglob("*") if p.is_file())
    print(f"[fetch] extracted {len(files)} files to {target.relative_to(ROOT)}/")

    manifest = {
        "package": package,
        "version": version,
        "registry": REGISTRY,
        "tarball_url": tarball_url,
        "tarball_sha256": sha256,
        "npm_shasum": dist.get("shasum"),
        "license": meta["versions"][version].get("license"),
        "description": meta["versions"][version].get("description"),
        "fetched_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "extracted_to": str(target.relative_to(ROOT)),
        "files": files,
    }
    manifest_path = RAW_DIR / "fetch_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"[fetch] wrote {manifest_path.relative_to(ROOT)}")
    return manifest


def _safe_member(name: str) -> bool:
    """Reject absolute paths and traversal in tar entries."""
    p = Path(name)
    return not p.is_absolute() and ".." not in p.parts


def main() -> None:
    ap = argparse.ArgumentParser(description="Fetch the Civ 6 roguelike game package from npm.")
    ap.add_argument("--package", default=PACKAGE)
    ap.add_argument("--version", default=None, help="defaults to the registry's latest tag")
    args = ap.parse_args()
    fetch_package(args.package, args.version)


if __name__ == "__main__":
    main()
