from .data_structures import *

import numpy as np
import cv2

from enum import Enum

SCREEN_SIZE = (1000, 1000)
SCREEN_OFFSET = (50, 50)

FONT = cv2.FONT_HERSHEY_SIMPLEX

WINDOW_NAME = 'Mappa'

img = np.zeros((SCREEN_SIZE[1] , SCREEN_SIZE[0], 3), np.uint8)

class Colors(Enum) :
    WHITE = (255, 255, 255)
    RED = (0, 0, 255)
    GREEN = (0, 255, 0)
    BLUE = (255, 0, 0)
    CYAN = (255, 255, 0)


def find_min_coords (data : list[Node]) -> cv2.typing.Point2d :
    min_coord_y = 360
    min_coord_x = 360

    for node in data :
        min_coord_y = min(min_coord_y, node.coords[0])
        min_coord_x = min(min_coord_x, node.coords[1])

    return (min_coord_y, min_coord_x)


def find_max_coords (data : list[Node]) -> cv2.typing.Point2d :
    max_coord_y = -360
    max_coord_x = -360

    for node in data :
        max_coord_y = max(max_coord_y, node.coords[0])
        max_coord_x = max(max_coord_x, node.coords[1])

    return (max_coord_y, max_coord_x)


def normalize_coords (coords : cv2.typing.Point2d, min_coords : cv2.typing.Point2d, max_coords : cv2.typing.Point2d, 
                      screen_size : cv2.typing.Size, offset : cv2.typing.Size) -> cv2.typing.Point2i :
    # Invert y to get real world drawing
    coord_y = (coords[0] - min_coords[0]) / (max_coords[0] - min_coords[0]) * (screen_size[1] - 2 * offset[1])
    coord_y = (screen_size[1] - 2 * offset[1]) - coord_y + offset[1]

    coord_x = (coords[1] - min_coords[1]) / (max_coords[1] - min_coords[1]) * (screen_size[0] - 2 * offset[0]) + offset[0]

    return (int(coord_x), int(coord_y)) 


def put_centered_text (img : cv2.typing.MatLike, text : str, org : cv2.typing.Point, 
                       fontFace : int, fontScale : float, color : cv2.typing.Scalar, thickness : int = 1) :
    
    textsize = cv2.getTextSize(text, fontFace, fontScale, thickness)[0]
    cv2.putText(img, text, (org[0] - (textsize[0] // 2), org[1]), fontFace, fontScale, color, thickness)


def find_center_of_line (point1 : cv2.typing.Point2i, point2 : cv2.typing.Point2i) -> cv2.typing.Point2i :
    sum = (point1[0] + point2[0], point1[1] + point2[1])
    return np.floor_divide(sum, 2)


def is_window_open (window_name : str) -> bool :
    return cv2.getWindowProperty(window_name, 0) >= 0
    

def draw (data : list[Node]) :
    min_coords = find_min_coords(data)
    max_coords = find_max_coords(data)
    
    for i in range(len(data)) :
        normalized_coords = normalize_coords(data[i].coords, min_coords, max_coords, SCREEN_SIZE, SCREEN_OFFSET)

        data[i].set_normalized_coords(normalized_coords)

    for i in range(len(data)) :
        # Draw nodes
        cv2.circle(img, data[i].normalized_coords, 5, Colors.WHITE.value, -1)
        put_centered_text(img, data[i].name, (data[i].normalized_coords[0], data[i].normalized_coords[1] + 20), FONT, 0.5, Colors.WHITE.value)

        # Draw links
        for link in data[i].links :
            cv2.line(img, data[i].normalized_coords, link[0].normalized_coords, Colors.BLUE.value)

            line_text_coords = find_center_of_line(data[i].normalized_coords, link[0].normalized_coords)
            put_centered_text(img, str(link[1]), line_text_coords, FONT, 0.65, Colors.CYAN.value, 2)

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_AUTOSIZE)
    cv2.moveWindow(WINDOW_NAME, 350, -60)
    cv2.imshow(WINDOW_NAME, img)


def draw_solution_path (path : list[Node]) :
    cv2.circle(img, path[0].normalized_coords, 5, Colors.GREEN.value, -1)
    cv2.circle(img, path[-1].normalized_coords, 5, Colors.GREEN.value, -1)

    for i in range(len(path) - 1) :
        current_node = path[i]
        next_node = path[i + 1]

        cv2.line(img, current_node.normalized_coords, next_node.normalized_coords, Colors.GREEN.value)

    cv2.imshow(WINDOW_NAME, img)