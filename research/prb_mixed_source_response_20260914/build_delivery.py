"""Build and verify the portable mixed-source validation evidence bundle."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
import zlib
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
TOP_FILES = (
    "README.md", "design_plan.md", "frozen_design.json", "noise_design.json",
    "mixed_vertex_response.py", "run_large_grid.py", "analyze_inference.py",
    "noise_sensitivity.py", "test_mixed_vertex_response.py",
    "build_delivery.py",
)
TREE_ROOTS = (
    "pilot/Bplus_pilot",
    "pilot_analysis/Bplus_pilot",
    "production/Bplus_coarse", "production/Bplus_final",
    "production/B0_coarse", "production/B0_final",
    "production_analysis/Bplus_coarse",
    "production_analysis/Bplus_final",
    "production_analysis/B0_final",
    "noise_sensitivity/run_v1",
)
FIXED_ZIP_TIME = (2026, 9, 15, 0, 0, 0)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def inventory():
    files = [ROOT / name for name in TOP_FILES]
    for relative in TREE_ROOTS:
        tree = ROOT / relative
        if not tree.is_dir():
            raise FileNotFoundError(tree)
        files.extend(path for path in tree.rglob("*") if path.is_file())
    if any(not path.is_file() for path in files):
        missing = [str(path) for path in files if not path.is_file()]
        raise FileNotFoundError(missing)
    files = sorted(set(files), key=lambda path: path.relative_to(ROOT).as_posix())
    forbidden = ("__pycache__", ".pyc", ".zip")
    if any(any(item in part for item in forbidden)
           for path in files for part in path.relative_to(ROOT).parts):
        raise ValueError("Cache/archive entered delivery inventory")
    return files


def build(output, manifest_path):
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.parent.resolve() != (ROOT / "delivery").resolve():
        raise ValueError("Delivery archive must stay under this study's delivery directory")
    files = inventory()
    temporary = output.with_suffix(output.suffix + ".tmp")
    members = []
    with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED,
                         compresslevel=9) as archive:
        for path in files:
            relative = path.relative_to(ROOT).as_posix()
            data = path.read_bytes()
            info = zipfile.ZipInfo(relative, date_time=FIXED_ZIP_TIME)
            info.compress_type = zipfile.ZIP_DEFLATED
            info.create_system = 3
            info.external_attr = 0o100644 << 16
            archive.writestr(info, data, compress_type=zipfile.ZIP_DEFLATED,
                             compresslevel=9)
            members.append({
                "path": relative, "bytes": len(data),
                "sha256": digest(data),
                "crc32": f"{zlib.crc32(data) & 0xffffffff:08x}"})
    os.replace(temporary, output)
    expected_names = [item["path"] for item in members]
    member_hashes_ok = True
    member_crc_ok = True
    with zipfile.ZipFile(output, "r") as archive:
        bad_member = archive.testzip()
        actual_names = archive.namelist()
        info = {item.filename: item for item in archive.infolist()}
        for item in members:
            data = archive.read(item["path"])
            member_hashes_ok &= digest(data) == item["sha256"]
            member_crc_ok &= f"{info[item['path']].CRC:08x}" == item["crc32"]
    archive_bytes = output.read_bytes()
    report = {
        "status": "DELIVERY_BUNDLE_VERIFIED" if (
            bad_member is None and actual_names == expected_names
            and member_hashes_ok and member_crc_ok) else
            "DELIVERY_BUNDLE_VALIDATION_FAILED",
        "archive": output.name,
        "archive_sha256": digest(archive_bytes),
        "archive_bytes": len(archive_bytes),
        "fixed_member_timestamp": "2026-09-15T00:00:00",
        "member_count": len(members),
        "members": members,
        "verification": {
            "zip_testzip_bad_member": bad_member,
            "member_order_exact": actual_names == expected_names,
            "member_sha256_all_match": member_hashes_ok,
            "member_crc32_all_match": member_crc_ok,
            "zip_or_cache_members_excluded": True}}
    temporary_manifest = manifest_path.with_suffix(manifest_path.suffix + ".tmp")
    temporary_manifest.write_text(
        json.dumps(report, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    os.replace(temporary_manifest, manifest_path)
    if report["status"] != "DELIVERY_BUNDLE_VERIFIED":
        raise SystemExit(1)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", default="prb_mixed_source_validation_01")
    args = parser.parse_args()
    if not args.name.replace("_", "").isalnum():
        raise ValueError("Use an alphanumeric/underscore bundle name")
    delivery = ROOT / "delivery"
    report = build(delivery / f"{args.name}.zip",
                   delivery / f"{args.name}_manifest.json")
    print(json.dumps({key: report[key] for key in (
        "status", "archive", "archive_sha256", "archive_bytes",
        "member_count")}, indent=2), flush=True)
