from lib.data_structures import Node
from lib.draw import draw, draw_solution_path, is_window_open, WINDOW_NAME
from lib.uninformed_search import BFS, DFS, UCS, DLS, IDS

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
    return IDS


def search_algorithms_main () :
    data, data_dict = set_data_verona()

    draw(data)

    algorithm = search_algorithm_selection()

    res_path = algorithm(data_dict['bagnolo'], data_dict['verona'])

    print(res_path)

    draw_solution_path(res_path)
    
    # Wait while window is opened or Q is pressed
    while is_window_open(WINDOW_NAME) :
        key = cv2.waitKey()

        if key == ord('q'):
            break


if __name__ == '__main__' :
    search_algorithms_main()

