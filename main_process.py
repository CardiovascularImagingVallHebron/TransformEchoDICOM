import os

for thread_env_var in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
):
    os.environ.setdefault(thread_env_var, "1")

import multiprocessing
import pandas as pd
from src.utils import extract_frames_from_dicom_parche, new_resize_img, new_orig_img
from src.utils_cone_extract import cone_extract
import pydicom
from tqdm import tqdm
import numpy as np
import cv2
import time

cv2.setNumThreads(1)

DEFAULT_MAX_WORKERS = 24
WORKERS_ENV_VAR = "PREPROCESS_DCM_WORKERS"
BATCH_SIZE_ENV_VAR = "PREPROCESS_DCM_BATCH_SIZE"


def _positive_int_from_env(env_var, default):
    value = os.environ.get(env_var)
    if value is None:
        return default

    try:
        parsed = int(value)
    except ValueError:
        print(f"Invalid {env_var}={value!r}; using {default}")
        return default

    if parsed < 1:
        print(f"Invalid {env_var}={value!r}; using {default}")
        return default

    return parsed


def get_worker_count():
    cpu_count = os.cpu_count() or 1
    default_workers = min(cpu_count, DEFAULT_MAX_WORKERS)
    requested_workers = _positive_int_from_env(WORKERS_ENV_VAR, default_workers)
    return max(1, min(requested_workers, cpu_count))


def get_batch_size(worker_count):
    default_batch_size = max(worker_count * 4, 1)
    return _positive_int_from_env(BATCH_SIZE_ENV_VAR, default_batch_size)


def iter_batches(items, batch_size):
    for start in range(0, len(items), batch_size):
        yield items[start:start + batch_size]


def init_worker():
    cv2.setNumThreads(1)


def iter_dicom_folders(root_folder):
    pending_dirs = [root_folder]

    with tqdm(desc="Reading folders", unit="folder") as pbar:
        while pending_dirs:
            dirpath = pending_dirs.pop()
            pbar.update(1)
            pbar.set_postfix_str(dirpath[-80:])

            try:
                entries = list(os.scandir(dirpath))
            except OSError as e:
                print(f"Could not read folder {dirpath}: {e}")
                continue

            filenames = [entry.name for entry in entries if entry.is_file()]
            dicom_files = [fn for fn in filenames if fn.lower().endswith('.dcm')]
            subdirs = [
                entry.path for entry in entries
                if os.path.isdir(entry.path)
            ]
            pbar.set_postfix_str(
                f"subdirs={len(subdirs)}, dicom={len(dicom_files)}, {dirpath[-60:]}"
            )

            if dicom_files:
                yield dirpath, dicom_files

            pending_dirs.extend(reversed(subdirs))


def ensure_output_folder(root_out_folder):
    try:
        os.makedirs(root_out_folder, exist_ok=True)
        test_path = os.path.join(root_out_folder, '.write_test')
        with open(test_path, 'w') as f:
            f.write('ok')
        os.remove(test_path)
    except OSError as e:
        raise PermissionError(
            f"Cannot write to output folder {root_out_folder}. "
            "Check permissions or use a local output path for testing."
        ) from e


def process_dicom_file(args):
    rel_dir, full_dicom_path, root_out_folder = args

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    x_dim, y_dim = 256, 256
    fps = 15

    # Filename without extension
    filename = os.path.splitext(os.path.basename(full_dicom_path))[0]

    # Replicate directory structure from DICOM inside AVI
    res_path  = os.path.join(root_out_folder, 'vids_resized', rel_dir)
    crop_path = os.path.join(root_out_folder, 'vids_cropped', rel_dir)
    try:
        os.makedirs(res_path, exist_ok=True)
        os.makedirs(crop_path, exist_ok=True)
    except OSError as e:
        print(f"❌ Cannot create output folders in {root_out_folder}: {e}")
        return

    vid_res_path  = os.path.join(res_path,  f'{filename}.avi')
    vid_crop_path = os.path.join(crop_path, f'{filename}.avi')

    if os.path.exists(vid_res_path) and os.path.exists(vid_crop_path):
        print(f"Skipped (already exists): {os.path.join(rel_dir, filename + '.dcm')}")
        return

    try:
        ds = pydicom.dcmread(full_dicom_path)

        if (ds.PhotometricInterpretation not in ['MONOCHROME1', 'MONOCHROME2'] and len(ds.pixel_array.shape) < 4) or \
           (ds.PhotometricInterpretation in ['MONOCHROME1', 'MONOCHROME2'] and len(ds.pixel_array.shape) < 3):
            return

        try:
            frames, frame_ecg_mask = extract_frames_from_dicom_parche(full_dicom_path)
            mask = cone_extract(frames, frame_ecg_mask, 4)
            suma_mask = np.sum(mask)
            total_image = mask.shape[0] * mask.shape[1]
            percentage_mask = (suma_mask / total_image * 100) / 3

            resize_writer = cv2.VideoWriter(vid_res_path, fourcc, fps, (x_dim, y_dim))
            croped_writer = None
            low_mask_record = None

            if percentage_mask > 15:
                for i, frame in enumerate(frames):
                    frame = (frame / 255.).astype(np.float32)

                    res_frame, res_mask, _ = new_resize_img(frame * mask, mask, x_dim, y_dim)
                    res_cone = (res_frame * res_mask * 255).astype(np.uint8)
                    if res_cone.ndim == 2:
                        res_cone = cv2.cvtColor(res_cone, cv2.COLOR_GRAY2BGR)
                    else:
                        res_cone = cv2.cvtColor(res_cone, cv2.COLOR_RGB2BGR)
                    resize_writer.write(res_cone)

                    crop_frame, crop_mask = new_orig_img(frame * mask, mask)
                    crop_cone = (crop_frame * crop_mask * 255).astype(np.uint8)
                    if crop_cone.ndim == 2:
                        crop_cone = cv2.cvtColor(crop_cone, cv2.COLOR_GRAY2BGR)
                    else:
                        crop_cone = cv2.cvtColor(crop_cone, cv2.COLOR_RGB2BGR)

                    if i == 0:
                        croped_writer = cv2.VideoWriter(
                            vid_crop_path, fourcc, fps, (crop_cone.shape[1], crop_cone.shape[0])
                        )
                    croped_writer.write(crop_cone)

                print(f'✅ OK: {os.path.join(rel_dir, filename)}')
            else:
                for i, frame in enumerate(frames):
                    if frame.ndim == 3:
                        h, w, _ = frame.shape
                    else:
                        h, w = frame.shape

                    min_dim = min(h, w)
                    cx, cy = w // 2, h // 2
                    x1 = max(cx - min_dim // 2, 0)
                    y1 = max(cy - min_dim // 2, 0)
                    x2, y2 = x1 + min_dim, y1 + min_dim

                    cropped = frame[y1:y2, x1:x2]
                    resized = cv2.resize(cropped, (x_dim, y_dim), interpolation=cv2.INTER_AREA)

                    if cropped.ndim == 2:
                        cropped_bgr = cv2.cvtColor(cropped, cv2.COLOR_GRAY2BGR)
                    else:
                        cropped_bgr = cv2.cvtColor(cropped, cv2.COLOR_RGB2BGR)

                    if resized.ndim == 2:
                        resized_bgr = cv2.cvtColor(resized, cv2.COLOR_GRAY2BGR)
                    else:
                        resized_bgr = cv2.cvtColor(resized, cv2.COLOR_RGB2BGR)

                    if i == 0:
                        croped_writer = cv2.VideoWriter(
                            vid_crop_path, fourcc, fps, (cropped_bgr.shape[1], cropped_bgr.shape[0])
                        )

                    resize_writer.write(resized_bgr)
                    croped_writer.write(cropped_bgr)

                print(f'⚠️ OK (low mask): {os.path.join(rel_dir, filename)}')
                low_mask_record = (rel_dir, filename)

            resize_writer.release()
            if croped_writer is not None:
                croped_writer.release()
            return low_mask_record

        except Exception as e:
            print(f"❌ Error {e} in: {full_dicom_path}")

    except Exception as e:
        print(f"❌ Error processing {full_dicom_path}: {e}")


def process_patient_folders(root_folder, root_out_folder):
    tasks = []
    low_mask_records = []

    print(f"Scanning DICOM root: {root_folder}")
    print(f"Writing output to: {root_out_folder}")
    ensure_output_folder(root_out_folder)

    # Scan folders with progress and process every folder containing DICOM files.
    for dirpath, dicom_files in iter_dicom_folders(root_folder):
        # Relative path with respect to root_folder
        rel_dir = os.path.relpath(dirpath, root_folder)
        # If it is the root, leave an empty string to avoid creating a "." folder
        if rel_dir == '.':
            rel_dir = ''

        for fn in dicom_files:
            full_path = os.path.join(dirpath, fn)
            tasks.append((rel_dir, full_path, root_out_folder))

        print(f"Found DICOM folder: {dirpath} ({len(dicom_files)} files)")

    total = len(tasks)
    if total == 0:
        print(f"No DICOM files found in: {root_folder}")
        return

    worker_count = get_worker_count()
    batch_size = get_batch_size(worker_count)
    print(
        f"Processing {total} DICOM files with {worker_count} workers "
        f"and batches of {batch_size}. "
        f"Override with {WORKERS_ENV_VAR} and {BATCH_SIZE_ENV_VAR}."
    )

    with tqdm(total=total, desc="Processing DICOM: ") as pbar:
        with multiprocessing.Pool(
            processes=worker_count,
            initializer=init_worker,
        ) as pool:
            for batch in iter_batches(tasks, batch_size):
                for result in pool.imap_unordered(process_dicom_file, batch, chunksize=1):
                    if result is not None:
                        low_mask_records.append(result)
                    pbar.update()

    # Save CSV with low mask
    if len(low_mask_records) > 0:
        df = pd.DataFrame(low_mask_records, columns=['relative_folder', 'filename'])
        csv_path = os.path.join(root_out_folder, 'low_mask_files.csv')
        df.to_csv(csv_path, index=False)
        print(f"CSV saved at: {csv_path}")


if __name__ == '__main__':
    start = time.time()
    root_folder = r'/mnt/nas/storage_dicom_us/data/'
    root_out_folder = r'/scratch/sie/storage_avi_us/data/'
    process_patient_folders(root_folder, root_out_folder)
    print('Total time: ', time.time() - start, 's')
