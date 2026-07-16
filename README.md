# Echo DICOM preprocessing

This project converts ultrasound DICOM files into processed images or videos. Multi-frame DICOM files are written as AVI videos, while single-frame DICOM files are written as PNG images.

## Single-image processing

For a single-frame DICOM, the original image and the resized image follow different paths:

- `vids_cropped/<relative_path>/<name>.png` contains the image at its original resolution.
- `vids_resized/<relative_path>/<name>.png` contains the processed image.

### Deciding whether to detect and crop the cone

The program first calls `is_2d_ultrasound_image()` to inspect the DICOM ultrasound-region metadata. It reads:

- Ultrasound Regions Sequence: `(0018,6011)`
- Region Spatial Format: `(0018,6012)`

An image is classified as 2D only when the ultrasound-region sequence exists, contains at least one region, and every region has `Region Spatial Format = 1`.

This prevents the cone-processing path from being applied to images whose regions represent another spatial format, such as M-Mode or spectral Doppler. If the metadata is absent, incomplete, or contains a non-2D region, the image is not treated as a 2D cone image.

### Cone detection for a 2D image

When an image is classified as 2D, `cone_extract()` attempts to find its ultrasound cone. Video cone detection normally starts from differences between frames. Because a single image has no temporal differences, its non-background pixels are used as the initial content mask instead.

Cone extraction then:

1. Selects the largest content contour.
2. Detects its edges with Canny edge detection.
3. Finds candidate boundary lines with a probabilistic Hough transform.
4. Groups line segments with similar angles.
5. Intersects the extended boundary lines to estimate the cone apex.
6. Builds a polygonal mask covering the cone.
7. Runs four attempts and keeps the valid mask with the largest area.

If the resulting mask contains pixels, the mask is applied to the image. The cone is cropped around its bounds and resized to exactly 256 x 256 pixels.

If cone detection returns an empty mask, processing falls back to aspect-ratio-preserving resizing. The shortest side becomes 256 pixels, and the other side is scaled proportionally.

### Decision flow

```text
Single-frame DICOM
  |
  +-- Are ultrasound regions present and are all of them spatial format 2D?
       |
       +-- No --> Preserve aspect ratio; resize shortest side to 256
       |
       +-- Yes --> Run cone detection
                    |
                    +-- Valid mask --> Apply mask, crop cone, resize to 256 x 256
                    |
                    +-- Empty mask --> Preserve aspect ratio; resize shortest side to 256
```

## Multi-frame processing

Multi-frame DICOM files continue through the video pipeline. Their frames are used to detect temporal differences, estimate the ultrasound cone, and write resized and cropped AVI outputs under `vids_resized` and `vids_cropped`.
