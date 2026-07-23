# Transform EchoDICOM

![Transform Echo DCM](data/transformEchoDCM.png)

## Requirements

- Python 3.11
- `opencv-python`
- `scikit-image`
- `pandas`
- `sympy`
- `numpy`
- `pydicom`
- `matplotlib`
- `tqdm`

Install the required Python libraries with:

```bash
pip install opencv-python scikit-image pandas sympy numpy pydicom matplotlib tqdm
```

## How to use

1. Open `main_process.py` and configure the input and output paths at the bottom of the file:

   ```python
   root_folder = r'/path_to_your_input_dicom_files/'
   root_out_folder = r'/path_to_output_your_transformed_files/'
   ```

2. Run the preprocessing script:

   ```bash
   python main_process.py
   ```

The input path must contain the DICOM files, but they do not need to be located directly in the root folder. The script recursively walks through `root_folder` and its child directories to find folders containing DICOM files. The output path will contain the mirrored folder structure under `vids_cropped` and `vids_resized`.

The number of workers and batch size can optionally be configured through the `PREPROCESS_DCM_WORKERS` and `PREPROCESS_DCM_BATCH_SIZE` environment variables.

## Extract DICOM metadata

`main_process_dicomtag.py` recursively scans a DICOM directory and creates a
CSV containing one row per valid DICOM file. It reads only the DICOM headers;
pixel data is not loaded.

Configure the dataset path and name at the bottom of the script:

```python
root_folder = Path(r"/path/to/input/dicom/files")
data_name = "dataset_name"
```

Then run:

```bash
python main_process_dicomtag.py
```

The CSV is written to:

```text
output/<data_name>_dicomtags.csv
```

The output includes the absolute `filepath`, a `filename` containing the path
relative to `root_folder` without its extension, the study and acquisition
metadata, image dimensions, ultrasound region format, and the inferred
`Modality`. The inferred modalities include B-mode images and cines, Color
Doppler, M-mode, Doppler, waveform, graphics, and unknown or unspecified cine
formats.

The CSV is incremental and resumable. Each DICOM row is flushed to disk as soon
as it is analyzed. If the script is restarted, paths already stored in the CSV
are skipped. CSV files produced by an earlier version without the `filename`
column are upgraded automatically before processing continues. A `tqdm`
progress bar reports newly saved and resumed files.

This project processes ultrasound DICOM files through two paths:

## Input type

- **Video:** A multi-frame DICOM is saved as a AVI video.
- **Image:** A single-frame DICOM is saved as a PNG image.

## Output type

Each input produces two types of output:

### Cropped (`vids_cropped`)

- **Images:** The PNG is always saved unchanged at its original resolution. This includes 2D ultrasound, M-Mode, spectral Doppler, waveform, graphics, and images without ultrasound-region metadata.
- **Videos with a detected cone:** The AVI keeps the area bounded by the ultrasound cone at its calculated crop resolution.
- **Videos without a valid cone:** Each frame is center-cropped to a square before it is written.

### Resized (`vids_resized`)

- **2D images with a detected cone:** The cone mask is applied, the image is cropped around the cone, and the result is resized to 256 x 256 pixels. The image is saved as a PNG.
- **M-Mode, spectral Doppler, waveform, graphics, images without region metadata, or 2D images where cone detection fails:** The aspect ratio is preserved and the shortest side is resized to 256 pixels. The other side is scaled proportionally. The image is saved as a PNG.
- **Videos with a detected cone:** Every frame is masked and cropped around the cone, then resized to 256 x 256 pixels and saved as an AVI.
- **Videos without a valid cone:** Every frame is center-cropped to a square and resized to 256 x 256 pixels before being saved as an AVI.

## Consideration

- The script can be relaunched after a failure or interruption. It detects output files that have already been created and skips them, allowing processing to continue without starting again from the beginning.

## Citation

In case of use this repository, please cite:

Lopez-Gutierrez, Pere, et al. "**Artificial intelligence for aortic valve calcium score quantification by echocardiography**." MedRxiv.

## Contact

For any questions or inquiries, feel free to reach out to Pere Lopez-Gutierrez, Vall d'Hebron Institut de Recerca, Barcelona, Spain: [pere.lopez@vhir.org](mailto:pere.lopez@vhir.org).
