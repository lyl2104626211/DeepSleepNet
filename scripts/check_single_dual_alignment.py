from __future__ import annotations

import argparse
import json
from pathlib import Path


def read_json(path: Path):
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def load_manifest_index(manifest_path: Path) -> dict[tuple[str, int], dict]:
    records = read_json(manifest_path)
    return {
        (str(record["subject_id"]), int(record["epoch_index"])): record
        for record in records
    }


def compare_manifests(single_manifest: Path, dual_manifest: Path) -> list[str]:
    single = load_manifest_index(single_manifest)
    dual = load_manifest_index(dual_manifest)
    issues: list[str] = []

    single_keys = set(single)
    dual_keys = set(dual)
    missing_in_single = sorted(dual_keys - single_keys)
    missing_in_dual = sorted(single_keys - dual_keys)
    if missing_in_single:
        issues.append(f"dual has {len(missing_in_single)} epochs missing in single")
        issues.extend(f"  missing in single: {item}" for item in missing_in_single[:10])
    if missing_in_dual:
        issues.append(f"single has {len(missing_in_dual)} epochs missing in dual")
        issues.extend(f"  missing in dual: {item}" for item in missing_in_dual[:10])

    for key in sorted(single_keys & dual_keys):
        single_record = single[key]
        dual_record = dual[key]
        for field in ("label", "start_second", "n_samples", "participant_id"):
            if single_record.get(field) != dual_record.get(field):
                issues.append(
                    f"{key} field mismatch {field}: "
                    f"single={single_record.get(field)!r}, dual={dual_record.get(field)!r}"
                )
                if len(issues) >= 50:
                    return issues

    return issues


def compare_splits(single_dir: Path, dual_dir: Path, n_folds: int) -> list[str]:
    issues: list[str] = []
    for fold_index in range(n_folds):
        fold = f"fold_{fold_index:02d}"
        single_path = single_dir / f"{fold}.json"
        dual_path = dual_dir / f"{fold}.json"
        if not single_path.exists() or not dual_path.exists():
            issues.append(f"{fold} split missing: single={single_path.exists()}, dual={dual_path.exists()}")
            continue

        single_split = read_json(single_path)
        dual_split = read_json(dual_path)
        for field in ("train_subjects", "val_subjects", "test_subjects"):
            if single_split.get(field) != dual_split.get(field):
                issues.append(f"{fold} {field} mismatch")
                issues.append(f"  single: {single_split.get(field)}")
                issues.append(f"  dual:   {dual_split.get(field)}")
    return issues


def summarize_manifest(manifest_path: Path) -> tuple[int, int]:
    records = read_json(manifest_path)
    subject_ids = {record["subject_id"] for record in records}
    return len(records), len(subject_ids)


def main() -> None:
    parser = argparse.ArgumentParser(description="Check single-channel v2 and EEG+EOG data alignment.")
    parser.add_argument("--single-dir", default="data/processed/sleep_edf_paper_fpz_cz_v2")
    parser.add_argument("--dual-dir", default="data/processed/sleep_edf_paper_fpz_cz_eeg_eog")
    parser.add_argument("--n-folds", type=int, default=20)
    args = parser.parse_args()

    single_dir = Path(args.single_dir)
    dual_dir = Path(args.dual_dir)
    single_manifest = single_dir / "manifest.json"
    dual_manifest = dual_dir / "manifest.json"

    manifest_issues = compare_manifests(single_manifest, dual_manifest)
    split_issues = compare_splits(single_dir, dual_dir, args.n_folds)

    single_records, single_subjects = summarize_manifest(single_manifest)
    dual_records, dual_subjects = summarize_manifest(dual_manifest)
    print(f"single: records={single_records}, subjects={single_subjects}, manifest={single_manifest}")
    print(f"dual:   records={dual_records}, subjects={dual_subjects}, manifest={dual_manifest}")

    if manifest_issues or split_issues:
        print("alignment check failed")
        for issue in manifest_issues + split_issues:
            print(f"- {issue}")
        raise SystemExit(1)

    print("alignment check passed")


if __name__ == "__main__":
    main()
