import numpy as np
import cv2
import os
import pydicom
from pydicom.pixel_data_handlers.util import convert_color_space
import matplotlib.pyplot as plt

import pydicom
import numpy as np
import cv2

def extract_frames_from_dicom(dicom_path, gray=False, flip=False):
    """
    Función que recibe un dicom de entrada y retorna todos sus frames como arrays de numpy.
    
    @dicom_path : string que contiene el path al DICOM
    @flip : boolean, True para voltear los frames horizontalmente (default: False)

    @return : tuple con (frames, frame_ecg_mask) donde:
              - frames es una lista de arrays con los frames procesados
              - frame_ecg_mask es un array (o None si no aplica)
    """
    # Leer el archivo DICOM
    ds = pydicom.dcmread(dicom_path)

    is_color = ds.PhotometricInterpretation not in ['MONOCHROME1', 'MONOCHROME2']
    
    # Verificar si contiene datos de pixeles
    if not hasattr(ds, "PixelData"):
        print(f"The DICOM file {dicom_path} does not have pixel data.")
        return [], None

    # Obtener los datos de video
    video_data = ds.pixel_array

    # Verificar si tiene múltiples frames
    if len(video_data.shape) < 4 and is_color:
        print("DICOM no tiene múltiples frames.")
        return [], None

    frames = []
    frame_ecg_mask = None

    num_frames = video_data.shape[0]
    for i in range(num_frames):
        frame = video_data[i]
        
        if is_color:
            frame_rgb = convert_color_space(frame, ds.PhotometricInterpretation, 'RGB')

            if (0x0028, 0x0014) in ds and ds[0x0028, 0x0014].value == 1 or gray==False:
                frame_gray = frame_rgb
            else:
                frame_gray = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2GRAY)

            # Último frame: crear la máscara ECG
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

        # Voltear horizontalmente si se solicita
        if flip:
            frame_gray = cv2.flip(frame_gray, 1)

        frames.append(frame_gray)

    return frames, frame_ecg_mask

def new_resize_img(pixel_array, mask_array, width, height):
    # Calculos de la máscara para identificar tamaños, se contabiliza la distancia del cono en pixeles y cm
    mask_array = np.array(mask_array, dtype='uint8')
    if mask_array.ndim == 3:
        row, cols, _ = np.where(mask_array > 0)
    else:
        row, cols = np.where(mask_array > 0)
    original_dist_x = np.max(cols) - np.min(cols)
    original_dist_y = np.max(row) - np.min(row)
    center_x = round(original_dist_x/2) + np.min(cols)
    center_y = round(original_dist_y/2) + np.min(row)

    # Redimensión de la mascara a las nuevas dimensiones escaladas
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
    new_pixel_size = None #parchecito

    return resized_image, resized_mask, new_pixel_size


def extract_frames_from_dicom_mayo(dicom_path, gray=False, flip=False):
    """
    Función que recibe un dicom de entrada y retorna todos sus frames como arrays de numpy.
    
    @dicom_path : string que contiene el path al DICOM
    @flip : boolean, True para voltear los frames horizontalmente (default: False)

    @return : tuple con (frames, frame_ecg_mask) donde:
              - frames es una lista de arrays con los frames procesados
              - frame_ecg_mask es un array (o None si no aplica)
    """
    # Leer el archivo DICOM
    ds = pydicom.dcmread(dicom_path)

    is_color = ds.PhotometricInterpretation not in ['MONOCHROME1', 'MONOCHROME2']
    
    # Verificar si contiene datos de pixeles
    if not hasattr(ds, "PixelData"):
        print(f"The DICOM file {dicom_path} does not have pixel data.")
        return [], None

    # Obtener los datos de video
    video_data = ds.pixel_array
    video_data = video_data[:, 50:, :]
    
    # Verificar si tiene múltiples frames
    if len(video_data.shape) < 4 and is_color:
        print("DICOM no tiene múltiples frames.")
        return [], None

    frames = []
    frame_ecg_mask = None

    num_frames = video_data.shape[0]
    for i in range(num_frames):
        frame = video_data[i]
        
        if is_color:
            # frame_rgb = convert_color_space(frame, ds.PhotometricInterpretation, 'RGB')
            frame_rgb = frame
            if (0x0028, 0x0014) in ds and ds[0x0028, 0x0014].value == 1 or gray==False:
                frame_gray = frame_rgb
            else:
                frame_gray = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2GRAY)

            # Último frame: crear la máscara ECG
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

        # Voltear horizontalmente si se solicita
        if flip:
            frame_gray = cv2.flip(frame_gray, 1)

        frames.append(frame_gray)

    return frames, frame_ecg_mask

def new_resize_img(pixel_array, mask_array, width, height):
    # Calculos de la máscara para identificar tamaños, se contabiliza la distancia del cono en pixeles y cm
    mask_array = np.array(mask_array, dtype='uint8')
    if mask_array.ndim == 3:
        row, cols, _ = np.where(mask_array > 0)
    else:
        row, cols = np.where(mask_array > 0)
    original_dist_x = np.max(cols) - np.min(cols)
    original_dist_y = np.max(row) - np.min(row)
    center_x = round(original_dist_x/2) + np.min(cols)
    center_y = round(original_dist_y/2) + np.min(row)

    # Redimensión de la mascara a las nuevas dimensiones escaladas
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
    new_pixel_size = None #parchecito

    return resized_image, resized_mask, new_pixel_size

def new_orig_img(pixel_array, mask_array):
    # Calculos de la máscara para identificar tamaños, se contabiliza la distancia del cono en pixeles y cm
    mask_array = np.array(mask_array, dtype='uint8')
    if mask_array.ndim == 3:
        row, cols, _ = np.where(mask_array > 0)
    else:
        row, cols = np.where(mask_array > 0)
    original_dist_x = np.max(cols) - np.min(cols)
    original_dist_y = np.max(row) - np.min(row)
    center_x = round(original_dist_x/2) + np.min(cols)
    center_y = round(original_dist_y/2) + np.min(row)

    # Redimensión de la mascara a las nuevas dimensiones escaladas
    if original_dist_x > original_dist_y:
        new_image = img_crop_v2(pixel_array, original_dist_x+6, original_dist_x+6, point=(center_x, center_y))
        new_mask = img_crop_v2(mask_array, original_dist_x+6, original_dist_x+6, point=(center_x, center_y))
    else:
        new_image = img_crop_v2(pixel_array, original_dist_y+6, original_dist_y+6, point=(center_x, center_y))
        new_mask = img_crop_v2(mask_array, original_dist_y+6, original_dist_y+6, point=(center_x, center_y))

    new_image = new_image.astype(np.float32)
    new_mask = new_mask.astype(np.float32)
    
    return new_image, new_mask

def new_orig_img_aha(pixel_array, mask_array, aha_array):
    # Calculos de la máscara para identificar tamaños, se contabiliza la distancia del cono en pixeles y cm
    mask_array = np.array(mask_array, dtype='uint8')
    if mask_array.ndim == 3:
        row, cols, _ = np.where(mask_array > 0)
    else:
        row, cols = np.where(mask_array > 0)
    original_dist_x = np.max(cols) - np.min(cols)
    original_dist_y = np.max(row) - np.min(row)
    center_x = round(original_dist_x/2) + np.min(cols)
    center_y = round(original_dist_y/2) + np.min(row)

    # Redimensión de la mascara a las nuevas dimensiones escaladas
    if original_dist_x > original_dist_y:
        new_image = img_crop_v4(pixel_array, original_dist_x+6, original_dist_x+6, point=(center_x, center_y))
        new_mask = img_crop_v4(mask_array, original_dist_x+6, original_dist_x+6, point=(center_x, center_y))
        new_aha = img_crop_v4(aha_array, original_dist_x+6, original_dist_x+6, point=(center_x, center_y))
    else:
        new_image = img_crop_v4(pixel_array, original_dist_y+6, original_dist_y+6, point=(center_x, center_y))
        new_mask = img_crop_v4(mask_array, original_dist_y+6, original_dist_y+6, point=(center_x, center_y))
        new_aha = img_crop_v4(aha_array, original_dist_y+6, original_dist_y+6, point=(center_x, center_y))

    new_image = new_image.astype(np.float32)
    new_mask = new_mask.astype(np.float32)
    new_aha = new_aha.astype(np.float32)
    
    return new_image, new_mask, new_aha

def img_crop_v4(img: np.ndarray, target_height: int, target_width: int, point: tuple = None) -> np.ndarray:
    """
    Recorta o añade padding a una imagen (2D o 3D) para ajustarse a un tamaño objetivo.
    Si se proporciona un punto (x, y), se usará como centro para el recorte/padding.
    """

    current_height, current_width = img.shape[:2]
    is_color = img.ndim == 3

    # Calcular cuánto padding se necesita
    pad_width = max(0, target_width - current_width)
    pad_height = max(0, target_height - current_height)

    pad_width_left = pad_width // 2
    pad_width_right = pad_width - pad_width_left
    pad_height_top = pad_height // 2
    pad_height_bottom = pad_height - pad_height_top

    # Padding dependiendo si es 2D o 3D
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

    # Ajustar punto al nuevo sistema de coordenadas (tras el padding)
    if point is not None:
        # point = (x, y) en la imagen original -> sumar el padding correspondiente
        point = (point[0] + pad_width_left, point[1] + pad_height_top)
    else:
        point = (padded_img.shape[1] // 2, padded_img.shape[0] // 2)

    start_x = max(0, int(point[0] - target_width // 2))
    start_y = max(0, int(point[1] - target_height // 2))

    # Evitar salirse de los límites
    start_x = min(start_x, padded_img.shape[1] - target_width)
    start_y = min(start_y, padded_img.shape[0] - target_height)
    end_x = start_x + target_width
    end_y = start_y + target_height

    # Recortar según si es color o no
    if is_color:
        cropped_img = padded_img[start_y:end_y, start_x:end_x, :]
    else:
        cropped_img = padded_img[start_y:end_y, start_x:end_x]

    return cropped_img


def img_crop_v2(img: np.ndarray, target_height: int, target_width: int, point: tuple = None) -> np.ndarray:
    """
    Recorta o añade padding a una imagen (2D o 3D) para ajustarse a un tamaño objetivo.
    Si se proporciona un punto (x, y), se usará como centro para el recorte/padding.
    """    
    current_height, current_width = img.shape[:2]
    is_color = img.ndim == 3

    # Calcular cuánto padding se necesita
    pad_width = max(0, target_width - current_width)
    pad_height = max(0, target_height - current_height)

    pad_width_left = pad_width // 2
    pad_width_right = pad_width - pad_width_left
    pad_height_top = pad_height // 2
    pad_height_bottom = pad_height - pad_height_top

    # Padding dependiendo si es 2D o 3D
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

    # Si no se da un punto, usar el centro
    if point is None:
        point = (padded_img.shape[1] // 2, padded_img.shape[0] // 2)

    start_x = max(0, point[0] - target_width // 2)
    start_y = max(0, point[1] - target_height // 2)

    # Evitar salirse de los límites
    start_x = min(start_x, padded_img.shape[1] - target_width)
    start_y = min(start_y, padded_img.shape[0] - target_height)
    end_x = start_x + target_width
    end_y = start_y + target_height

    # Recortar según si es color o no
    if is_color:
        cropped_img = padded_img[start_y:end_y, start_x:end_x, :]
    else:
        cropped_img = padded_img[start_y:end_y, start_x:end_x]

    return cropped_img

def extract_frames_from_dicom_parche(dicom_path, gray=False, flip=False):
    """
    Función que recibe un dicom de entrada y retorna todos sus frames como arrays de numpy.
    
    @dicom_path : string que contiene el path al DICOM
    @flip : boolean, True para voltear los frames horizontalmente (default: False)

    @return : tuple con (frames, frame_ecg_mask) donde:
              - frames es una lista de arrays con los frames procesados
              - frame_ecg_mask es un array (o None si no aplica)
    """
    # Leer el archivo DICOM
    ds = pydicom.dcmread(dicom_path)

    is_color = ds.PhotometricInterpretation not in ['MONOCHROME1', 'MONOCHROME2']
    
    # Verificar si contiene datos de pixeles
    if not hasattr(ds, "PixelData"):
        print(f"The DICOM file {dicom_path} does not have pixel data.")
        return [], None

    # Obtener los datos de video
    video_data = ds.pixel_array
    video_data = video_data[:, 50:, :]
    
    # Verificar si tiene múltiples frames
    if len(video_data.shape) < 4 and is_color:
        print("DICOM no tiene múltiples frames.")
        return [], None

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

            # Último frame: crear la máscara ECG
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

        # Voltear horizontalmente si se solicita
        if flip:
            frame_gray = cv2.flip(frame_gray, 1)

        frames.append(frame_gray)

    return frames, frame_ecg_mask