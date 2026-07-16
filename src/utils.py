import numpy as np
import cv2
import pydicom


def is_2d_ultrasound_image(dicom_path):
    """Return True when every declared ultrasound region is spatial format 2D."""
    ds = pydicom.dcmread(dicom_path, stop_before_pixels=True)
    regions_tag = (0x0018, 0x6011)
    spatial_format_tag = (0x0018, 0x6012)

    if regions_tag not in ds or not ds[regions_tag].value:
        return False

    spatial_formats = []
    for region in ds[regions_tag].value:
        if spatial_format_tag not in region:
            return False
        spatial_formats.append(int(region[spatial_format_tag].value))

    return bool(spatial_formats) and all(value == 1 for value in spatial_formats)


def resize_shortest_side(image, target_short_side=256):
    """Resize an image while preserving its aspect ratio."""
    height, width = image.shape[:2]
    scale = target_short_side / min(height, width)
    new_width = int(width * scale)
    new_height = int(height * scale)
    return cv2.resize(
        image,
        (new_width, new_height),
        interpolation=cv2.INTER_AREA,
    )


def new_resize_img(pixel_array, mask_array, width, height):
    # Calculate mask dimensions and measure the cone distance in pixels and centimeters.
    mask_array = np.array(mask_array, dtype='uint8')
    if mask_array.ndim == 3:
        row, cols, _ = np.where(mask_array > 0)
    else:
        row, cols = np.where(mask_array > 0)
    original_dist_x = np.max(cols) - np.min(cols)
    original_dist_y = np.max(row) - np.min(row)
    center_x = round(original_dist_x/2) + np.min(cols)
    center_y = round(original_dist_y/2) + np.min(row)

    # Resize the mask to the new scaled dimensions.
    if original_dist_x > original_dist_y:
        new_image = img_crop_v2(pixel_array, original_dist_x+6, original_dist_x+6, point=(center_x, center_y))
        new_mask = img_crop_v2(mask_array, original_dist_x+6, original_dist_x+6, point=(center_x, center_y))

        resized_image = cv2.resize(new_image, (width, height), interpolation = cv2.INTER_AREA)
        resized_mask = cv2.resize(new_mask, (width, height), interpolation = cv2.INTER_AREA)
    else:
        new_image = img_crop_v2(pixel_array, original_dist_y+6, original_dist_y+6, point=(center_x, center_y))
        new_mask = img_crop_v2(mask_array, original_dist_y+6, original_dist_y+6, point=(center_x, center_y))

        resized_image = cv2.resize(new_image, (width, height), interpolation = cv2.INTER_AREA)
        resized_mask = cv2.resize(new_mask, (width, height), interpolation = cv2.INTER_AREA)    

    resized_image = resized_image.astype(np.float32)
    resized_mask = resized_mask.astype(np.float32)
    new_pixel_size = None  # Temporary workaround.

    return resized_image, resized_mask, new_pixel_size

def new_orig_img(pixel_array, mask_array):
    # Calculate mask dimensions and measure the cone distance in pixels and centimeters.
    mask_array = np.array(mask_array, dtype='uint8')
    if mask_array.ndim == 3:
        row, cols, _ = np.where(mask_array > 0)
    else:
        row, cols = np.where(mask_array > 0)
    original_dist_x = np.max(cols) - np.min(cols)
    original_dist_y = np.max(row) - np.min(row)
    center_x = round(original_dist_x/2) + np.min(cols)
    center_y = round(original_dist_y/2) + np.min(row)

    # Resize the mask to the new scaled dimensions.
    if original_dist_x > original_dist_y:
        new_image = img_crop_v2(pixel_array, original_dist_x+6, original_dist_x+6, point=(center_x, center_y))
        new_mask = img_crop_v2(mask_array, original_dist_x+6, original_dist_x+6, point=(center_x, center_y))
    else:
        new_image = img_crop_v2(pixel_array, original_dist_y+6, original_dist_y+6, point=(center_x, center_y))
        new_mask = img_crop_v2(mask_array, original_dist_y+6, original_dist_y+6, point=(center_x, center_y))

    new_image = new_image.astype(np.float32)
    new_mask = new_mask.astype(np.float32)
    
    return new_image, new_mask

def img_crop_v2(img: np.ndarray, target_height: int, target_width: int, point: tuple = None) -> np.ndarray:
    """
    Crop or pad a 2D or 3D image to fit a target size.
    If an (x, y) point is provided, use it as the center for cropping or padding.
    """    
    current_height, current_width = img.shape[:2]
    is_color = img.ndim == 3

    # Calculate the required padding.
    pad_width = max(0, target_width - current_width)
    pad_height = max(0, target_height - current_height)

    pad_width_left = pad_width // 2
    pad_width_right = pad_width - pad_width_left
    pad_height_top = pad_height // 2
    pad_height_bottom = pad_height - pad_height_top

    # Apply padding according to whether the image is 2D or 3D.
    if is_color:
        padded_img = np.pad(
            img,
            ((pad_height_top, pad_height_bottom),
             (pad_width_left, pad_width_right),
             (0, 0)),
            mode='constant',
            constant_values=0
        )
    else:
        padded_img = np.pad(
            img,
            ((pad_height_top, pad_height_bottom),
             (pad_width_left, pad_width_right)),
            mode='constant',
            constant_values=0
        )

    # Use the image center when no point is provided.
    if point is None:
        point = (padded_img.shape[1] // 2, padded_img.shape[0] // 2)

    start_x = max(0, point[0] - target_width // 2)
    start_y = max(0, point[1] - target_height // 2)

    # Keep the crop within the image bounds.
    start_x = min(start_x, padded_img.shape[1] - target_width)
    start_y = min(start_y, padded_img.shape[0] - target_height)
    end_x = start_x + target_width
    end_y = start_y + target_height

    # Crop according to whether the image is color or grayscale.
    if is_color:
        cropped_img = padded_img[start_y:end_y, start_x:end_x, :]
    else:
        cropped_img = padded_img[start_y:end_y, start_x:end_x]

    return cropped_img

def extract_frames_from_dicom_parche(dicom_path, gray=False, flip=False):
    """
    Read an input DICOM and return all its frames as NumPy arrays.
    
    @dicom_path: String containing the path to the DICOM file.
    @flip: Boolean; True flips the frames horizontally (default: False).

    @return: Tuple containing (frames, frame_ecg_mask), where:
             - frames is a list of arrays containing the processed frames.
             - frame_ecg_mask is an array (or None when not applicable).
    """
    # Read the DICOM file.
    ds = pydicom.dcmread(dicom_path)

    is_color = ds.PhotometricInterpretation not in ['MONOCHROME1', 'MONOCHROME2']
    
    # Check whether it contains pixel data.
    if not hasattr(ds, "PixelData"):
        print(f"The DICOM file {dicom_path} does not have pixel data.")
        return [], None

    # Get the video data.
    video_data = ds.pixel_array

    # Return single-frame DICOM images without applying the video preprocessing crop.
    is_single_frame = (
        (is_color and video_data.ndim < 4)
        or (not is_color and video_data.ndim < 3)
    )
    if is_single_frame:
        if flip:
            video_data = cv2.flip(video_data, 1)
        return [video_data], None

    video_data = video_data[:, 50:, :]

    frames = []
    frame_ecg_mask = None

    num_frames = video_data.shape[0]
    for i in range(num_frames):
        frame = video_data[i]
        
        if is_color:
            frame_rgb = frame
            # frame_rgb = convert_color_space(frame, ds.PhotometricInterpretation, 'RGB')

            # if (0x0028, 0x0014) in ds and ds[0x0028, 0x0014].value == 1 or gray==False:
            frame_gray = frame_rgb
            # else:
            #     frame_gray = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2GRAY)

            # Create the ECG mask from the last frame.
            if i == (num_frames - 1):
                green_min = np.array([15, 100, 50], np.uint8)
                green_max = np.array([100, 255, 255], np.uint8)
                frame_hsv = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2HSV)
                frame_thresh = cv2.inRange(frame_hsv, green_min, green_max)
                frame_thresh = np.where(frame_thresh > 0, 0, 1)
                rows, cols = frame_thresh.shape
                frame_ecg_mask = np.ones((rows, cols))
                zero_indices = np.where(frame_thresh == 0)
                for r, c in zip(*zero_indices):
                    frame_ecg_mask[max(0, r-3):min(rows, r+4), c] = 0
        else:
            frame_gray = frame

        # Flip horizontally when requested.
        if flip:
            frame_gray = cv2.flip(frame_gray, 1)

        frames.append(frame_gray)

    return frames, frame_ecg_mask
