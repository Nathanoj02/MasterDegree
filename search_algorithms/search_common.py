from data_structures import Node
from cv2 import typing
import math

def reconstruct_path (start : Node, goal : Node, prev : dict[Node, Node]) -> list[Node] :
    if prev is None :
        return None
    
    path = [goal]   # solution path
    current_node = goal
    
    # Reconstruct path from last node
    while current_node != start :
        previous_node = prev[current_node]
        path.insert(0, previous_node)
        current_node = previous_node
    
    return path


def coords_to_distance (coords1 : typing.Point2d, coords2 : typing.Point2d) -> float :
    R = 6_371   # in Km
    lat1, lon1 = coords1
    lat2, lon2 = coords2

    phi1 = lat1 * math.pi / 180
    phi2 = lat2 * math.pi / 180
    delta_phi = (lat2 - lat1) * math.pi / 180
    delta_lambda = (lon2 - lon1) * math.pi / 180

    a = math.sin(delta_phi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    return R * c    # in Km