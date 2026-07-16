import cv2
import numpy as np
import os
from skimage.transform import probabilistic_hough_line
from itertools import combinations
import sympy as sp
from collections import defaultdict
from skimage.draw import polygon2mask
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from math import sqrt, cos, sin, atan2


def difference_frames(all_frames, mask_ecg):
    def process_frame(frame):
        if frame.ndim == 3 and frame.shape[2] == 3:
            frame = cv2.cvtColor(frame.astype(np.uint8), cv2.COLOR_RGB2GRAY)
        return frame

    # Procesar todos los frames
    all_frames_gray = [process_frame(frame) for frame in all_frames]

    # Ahora all_frames_gray contiene todos los frames convertidos a escala de grises si era necesario
    all_frames_ori = np.asarray(all_frames_gray)
    max_frames = all_frames_ori.max(axis=(0)) #max values
    min_frames = all_frames_ori.min(axis=(0)) #min values
    dif_rel_frames = max_frames - min_frames #distance between max and min

    if mask_ecg is None:
        mask_ecg = np.ones(all_frames[0].shape)

    dif_frames_filt = np.where(dif_rel_frames>6, 1, 0) * mask_ecg # * mask_bin_new #if difference is>3 is 1, else 0 
                                                        # con el nuevo cambio de color deberá ser -10
    frame_array = all_frames_ori[-1]
    
    return dif_frames_filt, frame_array

def draw_contours(dif_frames_filt, length=30):
    # Utiliza la función findContours de OpenCV para detectar los contornos
    contours, hierarchy = cv2.findContours(dif_frames_filt.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Crea una imagen de fondo negro con las mismas dimensiones que result_array
    mask = np.zeros_like(dif_frames_filt)

    # Dibuja los contornos en la máscara con fill (relleno) establecido a 1 (blanco)
    cone = max(contours, key = cv2.contourArea)

    cv2.drawContours(mask, [cone], -1, color=1, thickness=cv2.FILLED)

    image = mask * 255
    edges = cv2.Canny(image.astype('uint8'), 10, 200)

    lines = probabilistic_hough_line(edges, threshold=10, line_length=length,
                                    line_gap=1)
    
    angles = []

    for line in lines:
        p0, p1 = line
        # Calculate the current length of the line
        dx = p1[0] - p0[0]
        dy = p1[1] - p0[1]

        # Calculate the angle of the line
        theta = round(atan2(dy, dx)*180/np.pi,2)
        angles.append(theta)

    return mask, lines, angles

def extend_line(p0, p1, length=1000):
    """
    Extend the line defined by points p0 and p1 to a new length in both directions from p0.
    """
    from math import sqrt, cos, sin, atan2

    # Calculate the current length of the line
    dx = p1[0] - p0[0]
    dy = p1[1] - p0[1]

    # Calculate the angle of the line
    theta = atan2(dy, dx)

    # Calculate new end point using the angle and the desired length
    new_dx = cos(theta) * length
    new_dy = sin(theta) * length

    # Calculate new points with p0 as the center point
    new_p1 = (p0[0] + new_dx, p0[1] + new_dy)
    new_p0 = (p0[0] - new_dx, p0[1] - new_dy)

    return new_p0, new_p1 #, theta

def combine_segments(angles, segments):
    # Create a dictionary to hold segments by their rounded angle to account for floating point precision
    segments_by_angle = defaultdict(list)

    bucket_size = 6  # Esto representa un rango de 0.3 hacia cada lado

    for angle, segment in zip(angles, segments):
        # Encuentra el bucket para el ángulo actual
        if 30>np.abs(angle)>=0 or 180>=np.abs(angle)>150:
            if np.abs(angle) == 180 or np.abs(angle)== 0:
                bucket = np.random.random()
            else:
                bucket=angle
        elif 100>=np.abs(angle)>=82:
            continue
        else:
            bucket = round(np.abs(angle) / bucket_size)

        # Agrupa el segmento en el bucket correspondiente
        segments_by_angle[bucket].append(segment)
    # Function to find the extreme points from a list of segments considering their direction
    def extreme_points(segments):
        # Initialize min and max points as the first point of the first segment
        min_point, max_point = segments[0][0], segments[0][1]
        
        for segment in segments:
            # Unpack segment points
            p1, p2 = segment
            # Update min and max points
            if p1[1] < min_point[1]:
                min_point = p1
            if p2[1] > max_point[1]:
                max_point = p2
            # Swap if the segment is in the reverse direction
            if p1[1] > p2[1]:
                if p2[1] < min_point[1]:
                    min_point = p2
                if p1[1] > max_point[1]:
                    max_point = p1

        return (min_point, max_point)

    # Combine the segments with the same angle
    combined_segments = {}
    for angle, angle_segments in segments_by_angle.items():
        combined_segments[angle] = extreme_points(angle_segments)
    # print(combined_segments)
    # Convert the combined segments back to a list format
    combined_angles_segments = list(combined_segments.items())
    # print(combined_segments)
    return combined_angles_segments

def check_intersection(segment1, segment2):
    (x1, y1), (x2, y2) = segment1
    (x3, y3), (x4, y4) = segment2

    # Calculate denominators
    den1 = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    den2 = (x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)

    # If the denominators are 0, the lines are parallel
    if den1 == 0 or (x1 - x2) == 0 or (x3 - x4) == 0:
        return None

    # Calculate the intersection point
    t = den2 / den1
    x = x1 + t * (x2 - x1)
    y = y1 + t * (y2 - y1)

    # Check if the intersection point is within both line segments
    if min(x1, x2) <= x <= max(x1, x2) and min(y1, y2) <= y <= max(y1, y2) and \
       min(x3, x4) <= x <= max(x3, x4) and min(y3, y4) <= y <= max(y3, y4):
        return (x, y)
    else:
        return None

def find_cone_peak(segments):
    intersections = []
    intersect_segments = []
    # Check for intersection between all pairs of segments
    for segment1, segment2 in combinations(segments, 2):
        intersection = check_intersection(segment1, segment2)
        if intersection:
            intersections.append(intersection)
            intersect_segments.append((segment1, segment2))


    # Return the intersection with the lowest y-value if any intersections exist
    if intersections:
        min_value = min(intersections, key=lambda x: x[1])
        min_index = intersections.index(min_value)
        min_segments = intersect_segments[min_index]
        return min_value, min_segments
    else:
        return None

def inter_circ_seg(centro, radio, segment1, segment2):
    def interseccion_circulo_segmento(c, r, s1, s2):
        # Calcular la ecuación de la línea del segmento
        dx, dy = s2[0] - s1[0], s2[1] - s1[1]
        dr2 = float(dx**2 + dy**2)
        D = (s1[0] - c[0]) * (s2[1] - c[1]) - (s2[0] - c[0]) * (s1[1] - c[1])

        # Calcular el discriminante
        discriminante = r**2 * dr2 - D**2

        if discriminante < 0:
            # No hay intersección
            return []

        # Calcular los puntos de intersección
        sign_dy = 1 if dy >= 0 else -1
        sqrt_disc = np.sqrt(discriminante)
        x1 = (D * dy + sign_dy * dx * sqrt_disc) / dr2 + c[0]
        y1 = (-D * dx + abs(dy) * sqrt_disc) / dr2 + c[1]

        intersecciones = [(x1, y1)]

        if discriminante > 0:
            x2 = (D * dy - sign_dy * dx * sqrt_disc) / dr2 + c[0]
            y2 = (-D * dx - abs(dy) * sqrt_disc) / dr2 + c[1]
            intersecciones.append((x2, y2))

        return intersecciones

    puntos_interseccion1 = interseccion_circulo_segmento(centro, radio, segment1[0], segment1[1])
    puntos_interseccion2 = interseccion_circulo_segmento(centro, radio, segment2[0], segment2[1])

    return puntos_interseccion1, puntos_interseccion2

def polygon_cone(cone_peak, cone_segments, dif_frames_filt, image):
    # all_points = [point for segment in lines for point in segment]
    # max_dist_circ = max(all_points, key=lambda x: x[1])
    filas, columnas = np.where(image == 1)
    max_dist_circ = np.max(filas) if filas.size > 0 else None

    # max_dist_circ = np.max(np.where(image[:, round(cone_peak[0])] == 1)[0])

    # print(np.where(dif_frames_filt[:, round(cone_peak[0])] == 1), round(cone_peak[0]))
    # dist_circ = ((cone_peak[0]-max_dist_circ[0])**2 + (cone_peak[1]-max_dist_circ[1])**2 )**0.5
    dist_circ = np.abs(cone_peak[1]-max_dist_circ)

    circ_borders1, circ_borders2 = inter_circ_seg(cone_peak, dist_circ, cone_segments[0], cone_segments[1])
    
    circ_borders1 = max(circ_borders1, key=lambda x: x[1])
    circ_borders2 = max(circ_borders2, key=lambda x: x[1])

    circ_val_x = np.linspace(circ_borders1[0], circ_borders2[0],20)
    
    def comp_circ_points(circ_val_x, dist_circ, cone_peak):
        circ_points = []

        for x_val in circ_val_x:
            y_val = (dist_circ**2 - (x_val - cone_peak[0])**2)**0.5 + cone_peak[1] 
            if y_val>0:
                circ_points.append((x_val, y_val))

        circ_points.sort(key=lambda point: point[0])
        return circ_points
    
    y_coord_small = min(np.where(dif_frames_filt[:, round(cone_peak[0])] == 1)[0])
    
    dist_circ_small = np.abs(cone_peak[1]-y_coord_small)
    
    circ_points = comp_circ_points(circ_val_x, dist_circ, cone_peak)
    
    if dist_circ_small<25:
        polygon_points = [cone_peak]
        for point in circ_points:
            polygon_points.append(point)
    else:
        polygon_points = []
        dist_circ_small = (cone_peak[1]-y_coord_small)
        circ_borders_small1, circ_borders_small2 = inter_circ_seg(cone_peak, dist_circ_small, cone_segments[0], cone_segments[1])
        circ_borders_small1 = max(circ_borders_small1, key=lambda x: x[1])
        circ_borders_small2 = max(circ_borders_small2, key=lambda x: x[1])
        circ_val_small_x = np.linspace(circ_borders_small1[0], circ_borders_small2[0], 100)
        circ_points_small = comp_circ_points(circ_val_small_x, dist_circ_small, cone_peak)
        for point in circ_points:
            polygon_points.append(point)
        for point in circ_points_small[::-1]:
            polygon_points.append(point)

    def polygon_area(points):
        """
        Calculate the area of a polygon given its vertices.

        :param points: A list of tuples/lists representing the vertices of the polygon (e.g., [(x1, y1), (x2, y2), ...]).
        :return: The area of the polygon.
        """
        n = len(points)  # Number of vertices
        area = 0.0

        for i in range(n):
            j = (i + 1) % n
            area += points[i][0] * points[j][1]
            area -= points[j][0] * points[i][1]

        area = abs(area) / 2.0
        return area

    p_area = polygon_area(polygon_points)

    polygon_points_inv = [(point[1], point[0]) for point in polygon_points]

    binary_mask = polygon2mask((image.shape), polygon_points_inv)

    binary_array = binary_mask.astype(int)

    return binary_array, p_area

def cone_iteration(i, dif_frames_filt):
    try:
        # Retornamos la imagen 
        image, lines, angles = draw_contours(dif_frames_filt)

        # Calculate the combined segments
        combined_segments = [seg[1] for seg in combine_segments(angles, lines)]
        extended_lines = [extend_line(p0, p1, length=1000)[:2] for p0, p1 in combined_segments]
        
        # Final segments
        cone_peak, cone_segments = find_cone_peak(extended_lines)

        binary_array, polygon_area = polygon_cone(cone_peak, cone_segments, dif_frames_filt, image)

        return i, binary_array, polygon_area

    except Exception as e:
        # En caso de error, retornar ceros
        print(f"Error: {e}")
        return -1, np.zeros(dif_frames_filt.shape), 0

def cone_extract(all_frames, mask_ecg, n_iters):
    """
        Función que extrae el cono de la eco. Itera varias veces para sacar la mejor opción.
    """
    start = time.time()
    dif_frames_filt, frame = difference_frames(all_frames, mask_ecg)


    # Compute the final polygon cone binary array
    final_binary_array = np.zeros(dif_frames_filt.shape)
    final_polygon_area = 0

    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(cone_iteration, i, dif_frames_filt) for i in range(n_iters)]
        for future in as_completed(futures):
            i, binary_array, polygon_area = future.result()
            # print(i, polygon_area)
            if polygon_area > final_polygon_area:
                final_polygon_area = polygon_area
                final_binary_array = binary_array    

    # Justo antes del return
    if all_frames[0].ndim == 3 and all_frames[0].shape[2] == 3:
        final_binary_array = np.repeat(final_binary_array[:, :, np.newaxis], 3, axis=2)

    return final_binary_array