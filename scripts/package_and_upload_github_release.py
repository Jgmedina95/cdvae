#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import tarfile
from pathlib import Path
from urllib.parse import quote

import requests


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Package a trained CDVAE run directory and upload it to a GitHub release."
    )
    parser.add_argument("run_dir", type=Path, help="Hydra run directory containing checkpoints.")
    parser.add_argument("--repo", required=True, help="GitHub repo in owner/name format.")
    parser.add_argument("--tag", required=True, help="Release tag to create or update.")
    parser.add_argument(
        "--release-name",
        default=None,
        help="Optional release name. Defaults to the tag.",
    )
    parser.add_argument(
        "--asset-name",
        default=None,
        help="Optional asset filename. Defaults to <run_dir_name>.tar.gz.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Optional explicit checkpoint path to package.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Optional output directory for the packaged tarball.",
    )
    parser.add_argument(
        "--notes",
        default="Automated CDVAE model artifact upload.",
        help="Release body if a new release needs to be created.",
    )
    return parser.parse_args()


def find_checkpoint(run_dir: Path, asset_name: str | None = None) -> Path:
    if asset_name is not None and asset_name.endswith(("-last.tar.gz", "-latest.tar.gz")):
        last_ckpt = run_dir / "last.ckpt"
        if last_ckpt.exists():
            return last_ckpt

    checkpoints = sorted(run_dir.glob("*.ckpt"))
    if not checkpoints:
        raise FileNotFoundError(f"No checkpoint files found in {run_dir}")

    def checkpoint_sort_key(path: Path) -> tuple[int, int, float]:
        match = re.fullmatch(r"epoch=(\d+)-step=(\d+)(?:-v\d+)?\.ckpt", path.name)
        if match is None:
            return (-1, -1, path.stat().st_mtime)
        epoch = int(match.group(1))
        step = int(match.group(2))
        return (epoch, step, path.stat().st_mtime)

    best_checkpoint = max(checkpoints, key=checkpoint_sort_key)
    sort_key = checkpoint_sort_key(best_checkpoint)
    if sort_key[:2] == (-1, -1):
        raise FileNotFoundError(f"No epoch checkpoints found in {run_dir}")
    return best_checkpoint


def build_asset(
    run_dir: Path,
    output_dir: Path | None,
    asset_name: str | None,
    checkpoint: Path | None,
) -> Path:
    run_dir = run_dir.resolve()
    output_dir = (output_dir or run_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    asset_name = asset_name or f"{run_dir.name}.tar.gz"
    asset_path = output_dir / asset_name

    if checkpoint is not None:
        checkpoint_path = checkpoint.resolve()
    else:
        checkpoint_path = find_checkpoint(run_dir, asset_name=asset_name)

    files_to_include = [
        checkpoint_path,
        run_dir / "hparams.yaml",
        run_dir / "lattice_scaler.pt",
        run_dir / "prop_scaler.pt",
    ]

    missing_files = [path for path in files_to_include if not path.exists()]
    if missing_files:
        missing_str = ", ".join(str(path) for path in missing_files)
        raise FileNotFoundError(f"Missing expected artifact files: {missing_str}")

    with tarfile.open(asset_path, "w:gz") as archive:
        for file_path in files_to_include:
            archive.add(file_path, arcname=file_path.name)

    return asset_path


def github_headers(token: str) -> dict[str, str]:
    return {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }


def resolve_github_token() -> str | None:
    env_token = os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN")
    if env_token:
        return env_token

    try:
        result = subprocess.run(
            ["gh", "auth", "token"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None

    token = result.stdout.strip()
    return token or None


def get_or_create_release(repo: str, tag: str, release_name: str, notes: str, token: str) -> dict:
    api_root = f"https://api.github.com/repos/{repo}"
    headers = github_headers(token)

    response = requests.get(f"{api_root}/releases/tags/{quote(tag)}", headers=headers, timeout=30)
    if response.status_code == 200:
        return response.json()
    if response.status_code != 404:
        raise RuntimeError(f"Failed to fetch release: {response.status_code} {response.text}")

    create_response = requests.post(
        f"{api_root}/releases",
        headers=headers,
        json={
            "tag_name": tag,
            "name": release_name,
            "body": notes,
            "draft": False,
            "prerelease": False,
        },
        timeout=30,
    )
    if create_response.status_code not in {200, 201}:
        raise RuntimeError(
            f"Failed to create release: {create_response.status_code} {create_response.text}"
        )
    return create_response.json()


def delete_existing_asset(upload_url: str, token: str, asset_name: str) -> None:
    list_url = upload_url.split("{")[0].replace("uploads.github.com", "api.github.com")
    response = requests.get(list_url, headers=github_headers(token), timeout=30)
    if response.status_code != 200:
        raise RuntimeError(f"Failed to list release assets: {response.status_code} {response.text}")

    for asset in response.json():
        if asset.get("name") != asset_name:
            continue
        delete_response = requests.delete(asset["url"], headers=github_headers(token), timeout=30)
        if delete_response.status_code != 204:
            raise RuntimeError(
                f"Failed to delete existing asset: {delete_response.status_code} {delete_response.text}"
            )


def upload_asset(upload_url: str, asset_path: Path, token: str) -> dict:
    delete_existing_asset(upload_url, token, asset_path.name)
    target_url = f"{upload_url.split('{')[0]}?name={quote(asset_path.name)}"

    with asset_path.open("rb") as handle:
        response = requests.post(
            target_url,
            headers={
                "Authorization": f"Bearer {token}",
                "Content-Type": "application/gzip",
                "Accept": "application/vnd.github+json",
            },
            data=handle,
            timeout=300,
        )

    if response.status_code not in {200, 201}:
        raise RuntimeError(f"Failed to upload asset: {response.status_code} {response.text}")
    return response.json()


def main() -> int:
    args = parse_args()
    asset_path = build_asset(args.run_dir, args.output_dir, args.asset_name, args.checkpoint)

    print(json.dumps({"asset_path": str(asset_path)}, indent=2))

    token = resolve_github_token()
    if not token:
        print("No GITHUB_TOKEN or GH_TOKEN found; packaged artifact locally and skipped upload.")
        return 0

    release = get_or_create_release(
        repo=args.repo,
        tag=args.tag,
        release_name=args.release_name or args.tag,
        notes=args.notes,
        token=token,
    )
    uploaded_asset = upload_asset(release["upload_url"], asset_path, token)
    print(json.dumps({"browser_download_url": uploaded_asset["browser_download_url"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())