from lib.data_structures import Node
from lib.draw import draw, draw_solution_path, is_window_open, WINDOW_NAME
from lib.uninformed_search import BFS, DFS, UCS, DLS, IDS
from lib.informed_search import calculate_euristic_cost, GBFS, A_star

import cv2
import numpy as np

def test_priority_queue () :
    from lib.data_structures import PriorityQueue, QueueOrder
    import random

    p = PriorityQueue(QueueOrder.ASC)
    q = PriorityQueue(QueueOrder.DESC)

    for i in range(10) :
        n = random.randint(1, 20)
        p.enqueue(n, n)
        q.enqueue(n, n)
    
    print('Ascending queue:')
    print(p)
    print('Pop : ' + str(p.pop()))
    
    print('\nDescending queue:')
    print(q)
    print('Pop : ' + str(q.pop()))


def test_coords_to_distance () :
    from lib.search_common import coords_to_distance

    data_dict = {}

    data_dict['povegliano'] = Node('Povegliano', (45.342297, 10.873024))
    data_dict['borgo_roma'] = Node('Borgo Roma', (45.403335, 10.997704))
    data_dict['nogarole'] = Node('Nogarole', (45.292449, 10.884405))
    data_dict['bagnolo'] = Node('Bagnolo', (45.259508, 10.913514))
    data_dict['verona'] = Node('Verona', (45.439146, 10.991763))

    for node in data_dict.values() :
        for goal_node in data_dict.values() :
            print(f'{node} - {goal_node} : {coords_to_distance(node.coords, goal_node.coords)} Km')
        print()


def set_data_verona () :
    data = []
    data_dict = {}

    # Create nodes
    data_dict['povegliano'] = Node('Povegliano', (45.342297, 10.873024))
    data_dict['borgo_roma'] = Node('Borgo Roma', (45.403335, 10.997704))
    data_dict['verona'] = Node('Verona', (45.439146, 10.991763))
    data_dict['nogarole'] = Node('Nogarole', (45.292449, 10.884405))
    data_dict['bagnolo'] = Node('Bagnolo', (45.259508, 10.913514))   # Modified to fit the drawing well
    data_dict['vigasio'] = Node('Vigasio', (45.317706, 10.940659))
    data_dict['castel_azzano'] = Node('Castel d\'Azzano', (45.355398, 10.956398))
    data_dict['villafranca'] = Node('Villafranca', (45.351617, 10.844285))
    data_dict['mozzecane'] = Node('Mozzecane', (45.308702, 10.817966))
    data_dict['dossobuono'] = Node('Dossobuono', (45.393925, 10.913574))
    data_dict['pradelle'] = Node('Pradelle', (45.281103, 10.872574))
    data_dict['trevenzuolo'] = Node('Trevenzuolo', (45.269176, 10.933379))
    data_dict['cadidavid'] = Node('Cadidavid', (45.382643, 10.997859))
    data_dict['santa_lucia'] = Node('Santa Lucia', (45.407275, 10.930214))
    data_dict['alpo'] = Node('Alpo', (45.375906, 10.920560))
    data_dict['campagnola'] = Node('Campagnola', (45.348744, 11.056556))


    # Create links with respective weights
    data_dict['povegliano'].add_link(data_dict['villafranca'], 5)
    data_dict['povegliano'].add_link(data_dict['alpo'], 6)
    data_dict['povegliano'].add_link(data_dict['castel_azzano'], 8)
    data_dict['povegliano'].add_link(data_dict['vigasio'], 7)
    data_dict['povegliano'].add_link(data_dict['bagnolo'], 12)
    data_dict['povegliano'].add_link(data_dict['nogarole'], 8)
    data_dict['povegliano'].add_link(data_dict['pradelle'], 8)
    data_dict['povegliano'].add_link(data_dict['mozzecane'], 8)
    data_dict['villafranca'].add_link(data_dict['dossobuono'], 10)
    data_dict['dossobuono'].add_link(data_dict['santa_lucia'], 3)
    data_dict['santa_lucia'].add_link(data_dict['verona'], 11)
    data_dict['alpo'].add_link(data_dict['verona'], 21)
    data_dict['nogarole'].add_link(data_dict['pradelle'], 5)
    data_dict['nogarole'].add_link(data_dict['bagnolo'], 6)
    data_dict['pradelle'].add_link(data_dict['bagnolo'], 5)
    data_dict['bagnolo'].add_link(data_dict['trevenzuolo'], 3)
    data_dict['trevenzuolo'].add_link(data_dict['vigasio'], 14)
    data_dict['vigasio'].add_link(data_dict['castel_azzano'], 10)
    data_dict['castel_azzano'].add_link(data_dict['cadidavid'], 12)
    data_dict['cadidavid'].add_link(data_dict['borgo_roma'], 16)
    data_dict['borgo_roma'].add_link(data_dict['verona'], 23)
    data_dict['campagnola'].add_link(data_dict['cadidavid'], 8)
    data_dict['campagnola'].add_link(data_dict['borgo_roma'], 19)


    # Add to data
    for el in data_dict.values() :
        data.append(el)

    return data, data_dict


def search_algorithm_selection () :
    algorithms = [BFS, UCS, DFS, DLS, IDS, GBFS, A_star]

    print('Select one of the following algorithms:')
    for i, al in enumerate(algorithms):
        print(f'\t{i + 1}. {al.__name__}')
    print()

    a = 0
    while not 1 <= a <= len(algorithms) :
        a = int(input('Choose an option: '))
    
    return algorithms[a - 1]


def states_selection (data_dict : dict[str, Node]) -> tuple[Node, Node] :
    print('Available states:')
    for name in data_dict:
        print(name.capitalize())
    print()

    start_state = ''
    while start_state not in data_dict :
        start_state = input('Choose a starting state: ').lower()
    
    goal_state = ''
    while goal_state not in data_dict :
        goal_state = input('Choose a goal state: ').lower()
    
    return data_dict[start_state], data_dict[goal_state]


def search_algorithms_main () :
    # Get graph with connections
    data, data_dict = set_data_verona()

    # Draw a first map
    draw(data)

    print('Click on the map and press any key to continue...', end='\n\n')
    cv2.waitKey()

    # Ask start and goal states
    start, goal = states_selection(data_dict)
    print()
    
    # Ask which algorithms to use
    algorithm = search_algorithm_selection()
    print()

    res_path = None

    # Calculate euristic costs if it's an informed search algorithm
    if algorithm == GBFS or algorithm == A_star :
        for node in data_dict.values() :
            calculate_euristic_cost(node, goal)

    # DLS requires an extra parameter
    if algorithm == DLS :
        max_depth = int(input('Insert max depth: '))
        res_path = DLS(start, goal, depth_limit = max_depth)
    else :
        res_path = algorithm(start, goal)

    if res_path is not None :
        print(f'\nSolution path = {res_path}')
        
        draw_solution_path(res_path)
    else :
        print(f'\nNo path found from {start} to {goal}')

    print('\nClick on the map and press Q to terminate...')
    
    # Wait while window is opened or Q is pressed
    while is_window_open(WINDOW_NAME) :
        key = cv2.waitKey()

        if key == ord('q') or key == ord('Q'):
            break


if __name__ == '__main__' :
    search_algorithms_main()
    # test_coords_to_distance()
    # test_priority_queue

