import cv2
import numpy as np
import heapq

from path_planning import *
from path_planning.a_star_planner import AStarPlanner


class AStarImplementation(AStarPlanner):
    def preloop(self):
        # initialize open_list and g_score
        self.open_list = []
        self.g_score = {self.start_node: 0.0}
        
        # calculate initial f_cost for the start node and add it to the open list
        h_start = calculate_node_distance(self.start_node, self.goal_node)
        self.start_node.cost = 0.0 #
        heapq.heappush(self.open_list, (h_start, self.start_node))
        
        self.visited_nodes: set[PathNode] = set()
        # Track nodes in the open list to avoid duplicates
        self.enqueued_nodes = {self.start_node: 0.0}

    def step(self):
        if not self.open_list:
            self.is_done.set()
            return

        # Get the node with the lowest f_cost
        current_f, current_node = heapq.heappop(self.open_list)
        
        if current_node in self.visited_nodes:
            return
        
        # visit the current node
        self.visited_nodes.add(current_node)

        # Check if we have reached the goal
        dist_to_goal = calculate_node_distance(current_node, self.goal_node)
        if dist_to_goal <= self.goal_threshold:
            # Connect goal node to current node
            self.goal_node.parent = current_node
            self.goal_node.cost = self.g_score[current_node] + dist_to_goal
            self.visited_nodes.add(self.goal_node)
            self.is_done.set()
            return

        # Visit neighbors
        for neighbor in self.get_neighbor_nodes(current_node):
            if neighbor in self.visited_nodes:
                continue
            
            # Calculate tentative g_score
            tentative_g_score = self.g_score[current_node] + calculate_node_distance(current_node, neighbor)
            
            if neighbor not in self.enqueued_nodes or tentative_g_score < self.g_score.get(neighbor, float('inf')):
                # Update the best path to neighbor
                neighbor.parent = current_node
                neighbor.cost = tentative_g_score
                self.g_score[neighbor] = tentative_g_score
                f_score = tentative_g_score + calculate_node_distance(neighbor, self.goal_node)
                self.enqueued_nodes[neighbor] = tentative_g_score
                heapq.heappush(self.open_list, (f_score, neighbor))

    def postloop(self):
        path = collect_path(self.goal_node) if self.goal_node.parent else []
        return (path, self.visited_nodes)