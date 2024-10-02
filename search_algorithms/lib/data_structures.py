from cv2 import typing
from enum import Enum

class Node :
    def __init__(self, name, coords) :
        self.name = name
        self.coords = coords

        # Initializing parameters (to not get None)
        self.normalized_coords = (0, 0)
        self.links = []
        self.path_cost = 0 # Changing in-real-time

    def set_normalized_coords (self, normalized_coords : typing.Point2i) :
        self.normalized_coords = normalized_coords

    def add_link (self, node : 'Node', cost : int) :
        self.links.append((node, cost))
        node.links.append((self, cost))

    def set_euristics (self, euristic_cost_to_goal) :
        self.euristic_cost_to_goal = euristic_cost_to_goal

    def __repr__(self) -> str:
        return self.name
    

class QueueOrder(Enum) :
    ASC = 0
    DESC = 1


class PriorityQueue :
    def __init__(self, order : QueueOrder) :
        self.queue = []
        self.order = order
    
    def enqueue (self, element, order_value : int) :
        # If empty
        if not self.queue :
            self.queue.insert(0, (element, order_value))
            return
        
        # Perform a binary search to insert the new element -> O(log n)
        left = 0
        right = len(self.queue) - 1 

        while left < right :
            pos = (left + right) // 2

            if order_value > self.queue[pos][1]:
                left = min(pos + 1, right)
            else :
                right = max(pos - 1, left)

        # Add 1 to value if element is bigger / smaller than the last compared element
        # Insert into priority queue
        self.queue.insert(left + (order_value > self.queue[left][1]), (element, order_value))


    def pop (self) :
        return self.queue.pop(0)[0] if self.order == QueueOrder.ASC else self.queue.pop()[0]
    
    def get (self, element) :
        for i in range(len(self.queue)) :
            if (element == self.queue[i][0]) :
                return self.queue[i]
        
        return None
    
    def replace (self, element, order_value : int) :
        for i in range(len(self.queue)) :
            if (element == self.queue[i][0]) :
                self.queue.pop(i)
                break
        
        self.enqueue(element, order_value)
    
    def __contains__(self, element) -> bool :
        element_list = [el[0] for el in self.queue]
        return element in element_list
    
    def __repr__(self) -> str:
        return ', '.join(str((el[0], el[1])) for el in self.queue)
    
    def __iter__(self) :
        return (el[0] for el in self.queue)