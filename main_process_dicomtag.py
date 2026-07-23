"""Extract DICOM metadata recursively and save one row per DICOM to CSV."""

from __future__ import annotations

import csv
import os
from pathlib import Path
from typing import Any

import pandas as pd
import pydicom
from pydicom.errors import InvalidDicomError
from pydicom.multival import MultiValue
from tqdm import tqdm


METADATA_TAGS = {
    "Study Date": (0x0008, 0x0020),
    "Manufacturer": (0x0008, 0x0070),
    "Manufacturer's Model Name": (0x0008, 0x1090), 
    "Patient's Sex": (0x0010, 0x0040),
    "Patient's Size": (0x0010, 0x1020),
    "Patient's Weight": (0x0010, 0x1030),
    "Effective Duration": (0x0018, 0x0072),
    "Frame rate": (0x0018, 0x1063),
    "Actual Frame Duration": (0x0018, 0x1242),
    "Number of Frames": (0x0028, 0x0008),
    "Rows": (0x0028, 0x0010),
    "Columns": (0x0028, 0x0011),
    "Ultrasound Color Data Present": (0x0028, 0x0014),
}

OUTPUT_COLUMNS = [
    "filepath",
    "filename",
    "Study Date",
    "Manufacturer",
    "Manufacturer's Model Name",
    "Patient's Sex",
    "Patient's Size",
    "Patient's Weight",
    "Effective Duration",
    "Frame rate",
    "Actual Frame Duration",
    "Number of Frames",
    "Rows",
    "Columns",
    "Ultrasound Color Data Present",
    "Region Spatial Format",
    "Modality",
]


def get_value(ds: pydicom.dataset.Dataset, tag: tuple[int, int], default=None):
    """Return a CSV-friendly DICOM tag value, or ``default`` if absent."""
    element = ds.get(tag)
    if element is None:
        return default

    value = element.value
    if value is None:
        return default
    if isinstance(value, (MultiValue, list, tuple)):
        return "\\".join(str(item) for item in value)
    return value


def get_ultrasound_region_spatial_format(
    ds: pydicom.dataset.Dataset, default=None
):
    """Get the maximum Spatial Format from the Ultrasound Region sequence."""
    sequence = ds.get((0x0018, 0x6011))
    if sequence is None or not sequence.value:
        return default

    values = []
    for item in sequence.value:
        value = get_value(item, (0x0018, 0x6012))
        try:
            values.append(float(value))
        except (TypeError, ValueError):
            continue

    return max(values) if values else default


def infer_modality(row: pd.Series) -> str:
    """Infer the echo modality from region, frame count, and color data."""
    region = pd.to_numeric(row["Region Spatial Format"], errors="coerce")
    n_frames = pd.to_numeric(row["Number of Frames"], errors="coerce")
    color_present = pd.to_numeric(
        row["Ultrasound Color Data Present"], errors="coerce"
    )

    if pd.isna(region):
        if pd.isna(n_frames):
            return "N/A"
        return "Cine N/S"

    if region == 1:
        if pd.isna(n_frames):
            return "Image B-mode"
        if n_frames > 0:
            if color_present == 0:
                return "Cine B-mode"
            if color_present == 1:
                return "Cine Color Doppler"
    elif region == 2:
        return "M-mode"
    elif region == 3:
        return "Doppler"
    elif region == 4:
        return "Waveform"
    elif region == 5:
        return "Graphics"

    return "N/A"


def extract_metadata(file_path: Path) -> dict[str, Any] | None:
    """Read one file without pixel data; return None when it is not DICOM."""
    try:
        ds = pydicom.dcmread(file_path, stop_before_pixels=True, force=False)
    except (InvalidDicomError, PermissionError, IsADirectoryError, OSError):
        return None

    row = {"filepath": str(file_path.resolve())}
    row.update({name: get_value(ds, tag) for name, tag in METADATA_TAGS.items()})
    row["Region Spatial Format"] = get_ultrasound_region_spatial_format(ds)
    row["Modality"] = infer_modality(pd.Series(row))
    return row


def get_relative_filename(file_path: Path, root: Path) -> str:
    """Return the path below ``root`` without the file extension."""
    try:
        relative_path = file_path.resolve().relative_to(root.resolve())
    except ValueError:
        relative_path = Path(file_path.name)
    return str(relative_path.with_suffix(""))


def prepare_output_csv(output_csv: Path, root: Path) -> set[str]:
    """Create the CSV if needed and return normalized paths already recorded."""
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    if not output_csv.exists() or output_csv.stat().st_size == 0:
        with output_csv.open("w", newline="", encoding="utf-8") as csv_file:
            csv.DictWriter(csv_file, fieldnames=OUTPUT_COLUMNS).writeheader()
        return set()

    with output_csv.open("r", newline="", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        legacy_columns = [column for column in OUTPUT_COLUMNS if column != "filename"]
        if reader.fieldnames == legacy_columns:
            rows = list(reader)
        elif reader.fieldnames == OUTPUT_COLUMNS:
            return {
                os.path.normcase(row["filepath"])
                for row in reader
                if row.get("filepath")
            }
        else:
            raise ValueError(
                f"The existing CSV has unexpected columns and cannot be resumed: "
                f"{output_csv}"
            )

    # Upgrade a resumable CSV created by an earlier version of this script.
    temporary_csv = output_csv.with_suffix(output_csv.suffix + ".migration.tmp")
    with temporary_csv.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=OUTPUT_COLUMNS)
        writer.writeheader()
        for row in rows:
            if row.get("filepath"):
                row["filename"] = get_relative_filename(Path(row["filepath"]), root)
            writer.writerow(row)
        csv_file.flush()
        os.fsync(csv_file.fileno())
    os.replace(temporary_csv, output_csv)

    return {
        os.path.normcase(row["filepath"])
        for row in rows
        if row.get("filepath")
    }


def append_csv_row(output_csv: Path, row: dict[str, Any]) -> None:
    """Append and persist one DICOM row immediately."""
    with output_csv.open("a", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=OUTPUT_COLUMNS)
        writer.writerow(row)
        csv_file.flush()
        os.fsync(csv_file.fileno())


def build_dataframe(root: Path, output_csv: Path) -> tuple[int, int, int]:
    """Scan ``root``, appending each new DICOM to a resumable CSV."""
    if not root.is_dir():
        raise NotADirectoryError(f"DICOM directory does not exist: {root}")

    processed_paths = prepare_output_csv(output_csv, root)
    existing_count = len(processed_paths)
    new_count = 0
    skipped_count = 0
    progress = tqdm(
        root.rglob("*"),
        desc="Reading DICOM metadata",
        unit="entry",
        dynamic_ncols=True,
    )
    for file_path in progress:
        if file_path.is_file():
            resolved_path = str(file_path.resolve())
            if os.path.normcase(resolved_path) in processed_paths:
                skipped_count += 1
                progress.set_postfix(
                    saved=new_count, resumed=skipped_count, refresh=False
                )
                continue

            row = extract_metadata(file_path)
            if row is not None:
                row["filename"] = get_relative_filename(file_path, root)
                append_csv_row(output_csv, row)
                processed_paths.add(os.path.normcase(resolved_path))
                new_count += 1
                progress.set_postfix(
                    saved=new_count, resumed=skipped_count, refresh=False
                )

    return existing_count, new_count, skipped_count


def main(
    root_folder: Path,
    data_name: str,
    output_folder: Path = Path("output"),
) -> None:
    output_csv = output_folder / f"{data_name}_dicomtags.csv"

    existing_count, new_count, skipped_count = build_dataframe(
        root_folder, output_csv
    )

    print(f"DICOM files already in CSV: {existing_count}")
    print(f"New DICOM files saved: {new_count}")
    print(f"Previously processed DICOM files skipped: {skipped_count}")
    print(f"CSV saved to: {output_csv.resolve()}")


if __name__ == "__main__":
    root_folder = Path(r"dicom_folder")
    data_name = "DATANAME"
    main(root_folder, data_name)
