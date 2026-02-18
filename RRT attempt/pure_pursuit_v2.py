#!/usr/bin/env python3
import rospy
import tf
import numpy as np
import math
from geometry_msgs.msg import PoseStamped
from ackermann_msgs.msg import AckermannDriveStamped
from visualization_msgs.msg import Marker
from nav_msgs.msg import Odometry, Path # On ajoute Path ici

class PurePursuit(object):
    def __init__(self):
        self.lookahead_distance = 1.0
        self.wheelbase = 0.33 
        self.max_steering_angle = 0.4189
        self.max_speed = 2.0 # On baisse un peu pour tester
        
        self.wp_x = np.array([])
        self.wp_y = np.array([])
        
        # Il écoute le chemin envoyé par RRT
        self.path_sub = rospy.Subscriber('/global_path', Path, self.path_callback, queue_size=1)

        self.drive_pub = rospy.Publisher('/nav', AckermannDriveStamped, queue_size=1)
        self.viz_pub = rospy.Publisher('/lookahead_marker', Marker, queue_size=1)
        self.pose_sub = rospy.Subscriber('/odom', Odometry, self.pose_callback, queue_size=1)

    def path_callback(self, path_msg):
        """ Reçoit le chemin du RRT* et le transforme en waypoints utilisables """
        new_wp_x = []
        new_wp_y = []
        for pose in path_msg.poses:
            new_wp_x.append(pose.pose.position.x)
            new_wp_y.append(pose.pose.position.y)
        
        self.wp_x = np.array(new_wp_x)
        self.wp_y = np.array(new_wp_y)
        rospy.loginfo("Nouveau chemin RRT* reçu !")

    def pose_callback(self, odom_msg):
        # Sécurité : Si le RRT* n'a pas encore envoyé de chemin, on ne bouge pas
        if len(self.wp_x) == 0:
            return

        curr_x = odom_msg.pose.pose.position.x
        curr_y = odom_msg.pose.pose.position.y
        
        # Orientation (Yaw)
        quat = odom_msg.pose.pose.orientation
        euler = tf.transformations.euler_from_quaternion([quat.x, quat.y, quat.z, quat.w])
        yaw = euler[2]
        
        # Transformation vers le repère voiture (Ta logique actuelle est parfaite)
        dx = self.wp_x - curr_x
        dy = self.wp_y - curr_y
        local_x = dx * math.cos(yaw) + dy * math.sin(yaw)
        local_y = -dx * math.sin(yaw) + dy * math.cos(yaw)
        
        distances = np.sqrt(local_x**2 + local_y**2)
        valid = (local_x > 0.0) & (distances >= self.lookahead_distance * 0.8)
        
        if np.any(valid):
            diff = np.abs(distances - self.lookahead_distance)
            target_idx = np.argmin(diff)
            
            lx = local_x[target_idx]
            ly = local_y[target_idx]
            
            # Formule Pure Pursuit
            L_sq = lx**2 + ly**2

            if L_sq > 0.001: # On ne calcule que si on n'est pas sur le point
                steering_angle = math.atan((2.0 * self.wheelbase * ly) / L_sq)
            else:
                steering_angle = 0.0 # On reste droit si on est arrivé ou trop proche
            
            # Envoi commande
            drive_msg = AckermannDriveStamped()
            drive_msg.header.stamp = rospy.Time.now()
            drive_msg.drive.speed = self.max_speed
            drive_msg.drive.steering_angle = np.clip(steering_angle, -self.max_steering_angle, self.max_steering_angle)
            self.drive_pub.publish(drive_msg)
            
            self.publish_marker(self.wp_x[target_idx], self.wp_y[target_idx])

    def publish_marker(self, x, y):
        m = Marker()
        m.header.frame_id = "map" # Vérifie bien que c'est "map" ou "world"
        m.header.stamp = rospy.Time.now()
        m.id = 0
        m.type = Marker.SPHERE
        m.action = Marker.ADD
        m.pose.position.x = x
        m.pose.position.y = y
        m.pose.position.z = 0.0
        m.pose.orientation.w = 1.0
        m.scale.x = 0.3
        m.scale.y = 0.3
        m.scale.z = 0.3
        m.color.a = 1.0
        m.color.r = 1.0 # Rouge pour bien le voir
        m.color.g = 0.0
        m.color.b = 0.0
        m.lifetime = rospy.Duration(0.1) # Durée de vie courte pour qu'il suive bien
        self.viz_pub.publish(m)

if __name__ == '__main__':
    try:
        rospy.init_node('pure_pursuit_node')
        pp = PurePursuit()
        rospy.spin() # <--- C'est ça qui empêche le script de s'arrêter
    except rospy.ROSInterruptException:
        pass