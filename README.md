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

## Credits

Pere Lopez-Gutierrez, Vall d'Hebron Institut de Recerca, Barcelona, Spain
