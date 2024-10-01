from data_structures import Node, PriorityQueue, QueueOrder
from search_common import reconstruct_path

# ------------------------------------------- BFS -------------------------------------------

def BFS (start : Node, goal : Node) -> list[Node] :
    prev = _BFS(start, goal)

    return reconstruct_path(start, goal, prev)


def _BFS (start : Node, goal : Node) -> dict[Node, Node] :
    prev = {}   # Dict of previous node for each explored node

    if start == goal :
        return prev
    
    frontier = [start]  # FIFO queue (normal queue)
    explored = set()    # explored nodes

    while frontier :
        current_node = frontier.pop(0)
        explored.add(current_node)

        for link in current_node.links :
            child = link[0]

            if child not in explored and child not in frontier:
                prev[child] = current_node
                
                if child == goal :
                    return prev
                
                frontier.append(child)

    return None


# ------------------------------------------- UCS -------------------------------------------

def UCS (start : Node, goal : Node) -> list[Node] :
    prev = _UCS(start, goal)

    return reconstruct_path(start, goal, prev)


def _UCS (start : Node, goal : Node) -> dict[Node, Node] :
    prev = {}   # Dict of previous node for each explored node

    if start == goal :
        return prev
    
    frontier = PriorityQueue(QueueOrder.ASC)  # priority queue
    frontier.enqueue(start, 0)
    explored = set()    # explored nodes

    while frontier :
        current_node = frontier.pop()

        if current_node == goal :
            return prev
        
        explored.add(current_node)

        for link in current_node.links :
            child = link[0]
            child.path_cost = current_node.path_cost + link[1]

            if child not in explored and child not in frontier:
                frontier.enqueue(child, child.path_cost)
                prev[child] = current_node
            
            elif child in frontier :
                _, prev_cost = frontier.get(child)

                if prev_cost > child.path_cost :
                    frontier.replace(child, child.path_cost)
                    prev[child] = current_node

    return None


# ------------------------------------------- DFS -------------------------------------------

def DFS (start : Node, goal : Node) -> list[Node] :
    prev = _DFS(start, goal)

    return reconstruct_path(start, goal, prev)


def _DFS (start : Node, goal : Node) -> dict[Node, Node] :
    prev = {}   # Dict of previous node for each explored node

    if start == goal :
        return prev
    
    frontier = [start]  # LIFO queue (stack)
    explored = set()    # explored nodes

    while frontier :
        current_node = frontier.pop()
        explored.add(current_node)

        for link in current_node.links :
            child = link[0]

            if child not in explored and child not in frontier :
                prev[child] = current_node

                if child == goal :
                    return prev
            
                frontier.append(child)

    return None


# ------------------------------------------- DLS -------------------------------------------

def DLS (start : Node, goal : Node, depth_limit : int = 1_000_000) -> list[Node] :
    prev = _DLS(start, goal, depth_limit)   

    return reconstruct_path(start, goal, prev)


def _DLS (start : Node, goal : Node, depth_limit : int) -> dict[Node, Node] :
    prev = {}   # Dict of previous node for each explored node

    if start == goal :
        return prev
    
    start.path_cost = 0
    frontier = [start]  # LIFO queue (stack)
    explored = set()    # explored nodes

    while frontier :
        current_node = frontier.pop()

        if current_node.path_cost >= depth_limit :
            continue

        explored.add(current_node)

        for link in current_node.links :
            child = link[0]
            
            if child not in explored and child not in frontier :
                prev[child] = current_node

                if child == goal :
                    return prev
            
                child.path_cost = current_node.path_cost + 1
                frontier.append(child)

    return None

# ------------------------------------------- IDS -------------------------------------------

def IDS (start : Node, goal : Node) -> list[Node] :
    prev = _IDS(start, goal)   

    return reconstruct_path(start, goal, prev)


def _IDS (start : Node, goal : Node) -> dict[Node, Node] :
    current_depth = 1

    while True :
        prev = _DLS(start, goal, current_depth)

        if prev is not None :
            return prev

        current_depth += 1

# ---------------------------------- Bidirectional Search -----------------------------------