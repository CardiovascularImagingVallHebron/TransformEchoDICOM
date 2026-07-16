# Transform Echo DCM

This project processes ultrasound DICOM files through two paths:

## Input type

- **Video:** A multi-frame DICOM is saved as a processed video.
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

## Considerations

- The script can be relaunched after a failure or interruption. It detects output files that have already been created and skips them, allowing processing to continue without starting again from the beginning.
- The script receives an input path and an output path. The input path must point to the directory containing the DICOM images. While transforming the files, the script mirrors the input directory structure inside both `vids_cropped` and `vids_resized` in the output directory.

## Credits

Pere Lopez-Gutierrez, Vall d'Hebron Institut de Recerca, Barcelona, Spain
