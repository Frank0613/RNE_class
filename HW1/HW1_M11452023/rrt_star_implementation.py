import cv2
import numpy as np

from path_planning import *
from path_planning.rrt_star_planner import RRTStarPlanner


class RRTStarImplementation(RRTStarPlanner):
    def preloop(self):
        self.visited_nodes: set[PathNode] = set()
        self.visited_nodes.add(self.start_node)
        self.start_node.cost = 0.0

    def step(self):
        # Sample random node
        rnd_node = self.sample_random_node()
        
        # Find nearest node in the tree
        nearest_node = min(self.visited_nodes, key=lambda n: calculate_node_distance(n, rnd_node))
        
        # Steering
        dist = calculate_node_distance(nearest_node, rnd_node)
        if dist == 0: return
        
        theta = np.arctan2(rnd_node.coordinates.y - nearest_node.coordinates.y, 
                           rnd_node.coordinates.x - nearest_node.coordinates.x)
        
        new_x = nearest_node.coordinates.x + self.step_size * np.cos(theta)
        new_y = nearest_node.coordinates.y + self.step_size * np.sin(theta)
        new_node = PathNode(coordinates=PixelCoordinates(new_x, new_y))
        
        # Collision Check
        if not check_inside_map(self.occupancy_map, new_node): return
        if not check_collision_free(self.occupancy_map, nearest_node, new_node): return

        # Find nearby nodes within search radius
        near_nodes = [n for n in self.visited_nodes 
                      if calculate_node_distance(n, new_node) <= self.search_radius]
        
        # Re-Parent
        best_parent = nearest_node
        min_cost = nearest_node.cost + calculate_node_distance(nearest_node, new_node)
        
        for near_node in near_nodes:
            if check_collision_free(self.occupancy_map, near_node, new_node):
                cost = near_node.cost + calculate_node_distance(near_node, new_node)
                if cost < min_cost:
                    best_parent = near_node
                    min_cost = cost
        
        new_node.parent = best_parent
        new_node.cost = min_cost
        self.visited_nodes.add(new_node)

        # Rewire
        for near_node in near_nodes:
            if near_node == best_parent: continue
            new_potential_cost = new_node.cost + calculate_node_distance(new_node, near_node)
            if new_potential_cost < near_node.cost:
                if check_collision_free(self.occupancy_map, new_node, near_node):
                    near_node.parent = new_node
                    near_node.cost = new_potential_cost

        # Check if goal can be connected
        if calculate_node_distance(new_node, self.goal_node) <= self.goal_threshold:
            if check_collision_free(self.occupancy_map, new_node, self.goal_node):
                self.goal_node.parent = new_node
                self.goal_node.cost = new_node.cost + calculate_node_distance(new_node, self.goal_node)
                self.visited_nodes.add(self.goal_node)
                self.is_done.set()

    def postloop(self):
        path = collect_path(self.goal_node) if self.goal_node.parent else []
        return (path, self.visited_nodes)