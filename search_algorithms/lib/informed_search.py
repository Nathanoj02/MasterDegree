from .data_structures import Node, PriorityQueue, QueueOrder
from .search_common import reconstruct_path, coords_to_distance

def calculate_euristic_cost (node : Node, goal : Node) :
    node.set_euristics(coords_to_distance(node.coords, goal.coords))


# ------------------------------------------ GBFS -------------------------------------------

def GBFS (start : Node, goal : Node) -> list[Node] :
    assert hasattr(start, 'euristic_cost_to_goal'), 'No euristic cost was calculated before executing an informed search'

    prev = _GBFS(start, goal)

    return reconstruct_path(start, goal, prev)


def _GBFS (start : Node, goal : Node) -> dict[Node, Node] :
    prev = {}   # Dict of previous node for each explored node

    if start == goal :
        return prev
    
    frontier = PriorityQueue(QueueOrder.ASC)  # priority queue
    frontier.enqueue(start, start.euristic_cost_to_goal)
    explored = set()    # explored nodes
    explored_list = []  # Only for printing states

    while frontier :
        current_node = frontier.pop()
        explored_list.append(current_node)

        if current_node == goal :
            print(f'Explored States: {explored_list}')
            return prev
        
        explored.add(current_node)

        for link in current_node.links :
            child = link[0]

            if child not in explored and child not in frontier:
                frontier.enqueue(child, child.euristic_cost_to_goal)
                prev[child] = current_node
            
            elif child in frontier :
                _, prev_cost = frontier.get(child)

                if prev_cost > child.euristic_cost_to_goal :
                    frontier.replace(child, child.euristic_cost_to_goal)
                    prev[child] = current_node

    return None


# ------------------------------------------- A* --------------------------------------------

def A_star (start : Node, goal : Node) -> list[Node] :
    assert hasattr(start, 'euristic_cost_to_goal'), 'No euristic cost was calculated before executing an informed search'

    prev = _A_star(start, goal)

    return reconstruct_path(start, goal, prev)


def _A_star (start : Node, goal : Node) -> dict[Node, Node] :
    prev = {}   # Dict of previous node for each explored node

    if start == goal :
        return prev
    
    frontier = PriorityQueue(QueueOrder.ASC)  # priority queue
    frontier.enqueue(start, start.euristic_cost_to_goal)
    explored = set()    # explored nodes
    explored_list = []  # Only for printing states

    while frontier :
        current_node = frontier.pop()
        explored_list.append(current_node)

        if current_node == goal :
            print(f'Explored States: {explored_list}')
            return prev
        
        explored.add(current_node)

        for link in current_node.links :
            child = link[0]
            child.path_cost = current_node.path_cost + link[1]
            total_cost = child.path_cost + child.euristic_cost_to_goal

            if child not in explored and child not in frontier:
                frontier.enqueue(child, total_cost)
                prev[child] = current_node
            
            elif child in frontier :
                _, prev_cost = frontier.get(child)

                if prev_cost > total_cost :
                    frontier.replace(child, total_cost)
                    prev[child] = current_node

    return None
