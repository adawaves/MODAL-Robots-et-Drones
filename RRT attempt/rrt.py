#!/usr/bin/env python3
import rospy
import numpy as np
from scipy.spatial import cKDTree
from geometry_msgs.msg import PoseStamped, Point
from nav_msgs.msg import Path, OccupancyGrid
from ackermann_msgs.msg import AckermannDriveStamped


class Node:
    """ Elements of the Tree. """
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.parent = None
        self.cost = 0.0
        self.children = []

class RRT_Star_Fast:
    def __init__(self, start, goal, obstacle_list, rand_area, max_iter=1000, delta_s=2.0, delta_r=10.0):
        self.start = Node(start.x, start.y)
        self.goal = Node(goal.x, goal.y)
        self.min_rand, self.max_rand = rand_area
        self.max_iter = max_iter
        self.delta_s = delta_s
        self.delta_r = delta_r

        # Convert obstacles to a NumPy Matrix (N, 4)
        self.obs_matrix = np.array(obstacle_list)
        self.node_list = []

    def planning(self, occupancy_grid, sampling_method=None):

        if sampling_method is None:
            sampling_method = self.get_random_node

        self.node_list = [self.start]
        self.node_coords = np.array([[self.start.x, self.start.y]])

        for k in range(self.max_iter):

            # Sampling
            rnd_node = sampling_method()

            # Nearest Neighbor
            # Rebuilding tree cKDTree is faster
            spatial_tree = cKDTree(self.node_coords)
            _, nearest_ind = spatial_tree.query([rnd_node.x, rnd_node.y])
            nearest_node = self.node_list[nearest_ind]

            # Steer
            new_node = self.steer(nearest_node, rnd_node, self.delta_s)

            if self.check_collision(nearest_node, new_node, occupancy_grid): # Collision Check

                # Find Neighbors
                near_indices = spatial_tree.query_ball_point([new_node.x, new_node.y], self.delta_r)

                # OPTIMIZATION 1 Best Parent

                new_node.parent = nearest_node
                new_node.cost = nearest_node.cost + self.calc_dist(nearest_node, new_node)
                # Loop through neighbors to find better parent
                for i in near_indices:
                    near_node = self.node_list[i]
                    # Fast collision check
                    if self.check_collision(nearest_node, new_node, occupancy_grid):
                        d = self.calc_dist(near_node, new_node)
                        if near_node.cost + d < new_node.cost:
                            new_node.cost = near_node.cost + d
                            new_node.parent = near_node

                # Add to Tree
                self.node_list.append(new_node)
                self.node_coords = np.vstack((self.node_coords, [new_node.x, new_node.y]))
                new_node.parent.children.append(new_node)


                # OPTIMIZATION 2 Rewire

                for i in near_indices:
                    near_node = self.node_list[i]
                    dist_to_neighbor = self.calc_dist(new_node, near_node)
                    if new_node.cost + dist_to_neighbor < near_node.cost:
                        if self.check_collision(nearest_node, new_node, occupancy_grid):
                            # Update Parent
                            if near_node in near_node.parent.children:
                                near_node.parent.children.remove(near_node)
                            near_node.parent = new_node
                            near_node.cost = new_node.cost + dist_to_neighbor
                            new_node.children.append(near_node)
                            # Update Children Costs
                            self.propagate_cost_to_leaves(near_node)

            if k % 1000 == 0:
                print(f"Iter: {k}, Tree Size: {len(self.node_list)}")

        return self.get_path_to_goal()


    def check_collision(self, n1, n2, occupancy_grid):
        # On échantillonne des points le long du segment entre n1 et n2
        points = np.linspace([n1.x, n1.y], [n2.x, n2.y], num=10)
        for p in points:
            # Convertir coordonnée réelle (mètres) en index de grille (pixels)
            grid_x = int((p[0] - occupancy_grid.info.origin.position.x) / occupancy_grid.info.resolution)
            grid_y = int((p[1] - occupancy_grid.info.origin.position.y) / occupancy_grid.info.resolution)
            
            index = grid_x + grid_y * occupancy_grid.info.width
            if occupancy_grid.data[index] > 50: # Si probabilité > 50%, c'est un mur
                return False # Collision !
        return True


    def steer(self, from_node, to_node, extend_length):
        dist = self.calc_dist(from_node, to_node)
        if dist > extend_length:
            theta = np.arctan2(to_node.y - from_node.y, to_node.x - from_node.x)
            new_node = Node(from_node.x + extend_length * np.cos(theta),
                            from_node.y + extend_length * np.sin(theta))
        else:
            new_node = Node(to_node.x, to_node.y)
        return new_node


    def get_random_node(self, perc = 0.1):
        if np.random.rand() < perc: # 10% of the time steer towards the goal
              return Node(self.goal.x, self.goal.y)

        return Node(np.random.uniform(self.min_rand, self.max_rand),
                    np.random.uniform(self.min_rand, self.max_rand))


    def calc_dist(self, n1, n2):
        return np.hypot(n1.x - n2.x, n1.y - n2.y)


    def propagate_cost_to_leaves(self, parent_node):
        for child in parent_node.children:
            child.cost = parent_node.cost + self.calc_dist(parent_node, child)
            self.propagate_cost_to_leaves(child)


    def get_path_to_goal(self):
        spatial_tree = cKDTree(self.node_coords)
        _, last_index = spatial_tree.query([self.goal.x, self.goal.y])
        goal_node = self.node_list[last_index]

        path = []
        curr = goal_node
        while curr is not None:
            path.append([curr.x, curr.y])
            curr = curr.parent

        return np.array(path), goal_node.cost



    # For Question 20

    def optimize_path(self, path, occupancy_grid):
        """
        Safe greedy smoothing for STATIC obstacles (Robot 1).
        """
        if len(path) < 3:
            return np.array(path), 0.0

        optimized_path = [path[0]]
        current_idx = 0
        total_cost = 0.0

        while current_idx < len(path) - 1:
            check_idx = len(path) - 1
            found_shortcut = False

            n1 = Node(path[current_idx][0], path[current_idx][1])

            while check_idx > current_idx:
                n2 = Node(path[check_idx][0], path[check_idx][1])

                # Static check only
                if self.check_collision(n1, n2, occupancy_grid):
                    total_cost += self.calc_dist(n1, n2)
                    optimized_path.append(path[check_idx])
                    current_idx = check_idx
                    found_shortcut = True
                    break

                check_idx -= 1

            # SAFETY VALVE: If no shortcut found (not even the original next step), force step forward
            if not found_shortcut:
                current_idx += 1
                # We append the original next node to avoid getting stuck
                # (In a perfect world, this node is already in optimized_path if we just stepped,
                # but usually this block handles the 'un-optimizable' corners)
                optimized_path.append(path[current_idx])
                total_cost += self.calc_dist(n1, Node(path[current_idx][0], path[current_idx][1]))

        return np.array(optimized_path), total_cost


    # For Question 22
    def get_random_node_int(self, perc=0.1, perc_intelligent=0.3, sigma=1.0, fixed_margin=1.0):
        r = np.random.rand()

        if r < perc:
            return Node(self.goal.x, self.goal.y)

        elif r < (perc + perc_intelligent):
            obs_idx = np.random.randint(0, len(self.obs_matrix))
            ox, oy, w, h = self.obs_matrix[obs_idx]

            corners = [
                (ox, oy),          # Bottom-Left
                (ox + w, oy),      # Bottom-Right
                (ox + w, oy + h),  # Top-Right
                (ox, oy + h)       # Top-Left
            ]

            # Remove corners that touch the map borders
            valid_corners = []
            for (cx, cy) in corners:
                is_on_border = (cx <= self.min_rand + 1e-3 or cx >= self.max_rand - 1e-3 or
                                cy <= self.min_rand + 1e-3 or cy >= self.max_rand - 1e-3)
                if not is_on_border:
                    valid_corners.append((cx, cy))

            # FALLBACK
            if not valid_corners:
                return Node(np.random.uniform(self.min_rand, self.max_rand),
                            np.random.uniform(self.min_rand, self.max_rand))

            # Select a random valid corner
            cx, cy = valid_corners[np.random.randint(0, len(valid_corners))]

            # Calculate Center to determine push-out direction
            center_x, center_y = ox + w / 2.0, oy + h / 2.0
            sign_x, sign_y = np.sign(cx - center_x), np.sign(cy - center_y)

            base_x = cx + (sign_x * fixed_margin)
            base_y = cy + (sign_y * fixed_margin)

            # Gaussian Noise
            final_x = base_x + np.random.normal(0, sigma)
            final_y = base_y + np.random.normal(0, sigma)

            return Node(np.clip(final_x, self.min_rand, self.max_rand),
                        np.clip(final_y, self.min_rand, self.max_rand))

        # 3. Uniform Sampling
        else:
            return Node(np.random.uniform(self.min_rand, self.max_rand),
                        np.random.uniform(self.min_rand, self.max_rand))
        


class RRTStarNode:
    def __init__(self):
        rospy.init_node('rrt_star_planner')

        # Paramètres
        self.map_size = rospy.get_param('~map_size', 10.0)
        self.max_iter = rospy.get_param('~max_iter', 500)
        
        # État interne
        self.current_pose = None
        self.goal_pose = None
        self.obstacles = [] # Format: [[x, y, w, h], ...]

        # Publishers / Subscribers
        self.path_pub = rospy.Publisher('/global_path', Path, queue_size=1)
        self.drive_pub = rospy.Publisher('/nav', AckermannDriveStamped, queue_size=10)
        
        rospy.Subscriber('/map', OccupancyGrid, self.map_callback)
        rospy.Subscriber('/gt_pose', PoseStamped, self.pose_callback) # Ajuste selon ton robot
        rospy.Subscriber('/move_base_simple/goal', PoseStamped, self.goal_callback)

        rospy.loginfo("RRT* Node Ready. En attente d'un Goal dans RVIZ...")

    def pose_callback(self, msg):
        self.current_pose = msg.pose

    def goal_callback(self, msg):
        self.goal_pose = msg.pose
        self.plan_and_publish()

    def map_callback(self, msg):
        self.current_map = msg
        rospy.loginfo("Carte reçue et stockée.")

    def plan_and_publish(self):
        if self.current_pose is None or self.goal_pose is None:
            rospy.logwarn("Position ou Goal manquant !")
            return

        # 1. Préparation du RRT*
        start = Node(self.current_pose.position.x, self.current_pose.position.y)
        goal = Node(self.goal_pose.position.x, self.goal_pose.position.y)
        
        rrt = RRT_Star_Fast(
            start=start, 
            goal=goal, 
            obstacle_list=self.obstacles, 
            rand_area=[-self.map_size, self.map_size],
            max_iter=self.max_iter
        )

        # 2. Exécution du calcul
        rospy.loginfo("Calcul du chemin RRT*...")
        raw_path, cost = rrt.planning(self.current_map, sampling_method=rrt.get_random_node)

        if len(raw_path) > 0:
            # 3. Optimisation (Smoothing)
            smooth_path, _ = rrt.optimize_path(raw_path, self.current_map)
            
            # 4. Publication du chemin pour RVIZ
            self.publish_path(smooth_path)
            rospy.loginfo(f"Chemin trouvé ! Coût: {cost:.2f}")
        else:
            rospy.logerr("Impossible de trouver un chemin.")

    def publish_path(self, points):
        path_msg = Path()
        path_msg.header.frame_id = "map"
        path_msg.header.stamp = rospy.Time.now()
        
        for p in points:
            pose = PoseStamped()
            pose.pose.position.x = p[0]
            pose.pose.position.y = p[1]
            path_msg.poses.append(pose)
        
        self.path_pub.publish(path_msg)

if __name__ == '__main__':
    try:
        node = RRTStarNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass