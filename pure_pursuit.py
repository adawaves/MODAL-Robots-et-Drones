#!/usr/bin/env python
import rospy
import tf
import numpy as np
import math
import os
from geometry_msgs.msg import PoseStamped
from ackermann_msgs.msg import AckermannDriveStamped
from visualization_msgs.msg import Marker
from nav_msgs.msg import Odometry

class PurePursuit(object):
    def __init__(self):
        self.lookahead_distance = 1
        self.wheelbase = 0.33 
        self.max_steering_angle = 0.4189
        self.max_speed = 1.5
        
        # Load waypoints
        file_path = os.path.expanduser('~/rcws/logs/ftg3.csv')
        self.waypoints = np.loadtxt(file_path, delimiter=',')
        
        # Precompute waypoint x and y arrays for speed
        self.wp_x = self.waypoints[:, 0]
        self.wp_y = self.waypoints[:, 1]
        
        self.drive_pub = rospy.Publisher('/nav', AckermannDriveStamped, queue_size=1)
        self.viz_pub = rospy.Publisher('/lookahead_marker', Marker, queue_size=1)
        self.pose_sub = rospy.Subscriber('/odom', Odometry, self.pose_callback, queue_size=1)

    def pose_callback(self, odom_msg):
        # Get car position
        curr_x = odom_msg.pose.pose.position.x
        curr_y = odom_msg.pose.pose.position.y
        
        # Get car orientation
        quat = odom_msg.pose.pose.orientation
        euler = tf.transformations.euler_from_quaternion([quat.x, quat.y, quat.z, quat.w])
        yaw = euler[2]
        
        # Precompute trig
        cos_yaw = math.cos(yaw)
        sin_yaw = math.sin(yaw)
        
        # Transform ALL waypoints to car frame (vectorized)
        dx = self.wp_x - curr_x
        dy = self.wp_y - curr_y
        
        local_x = dx * cos_yaw + dy * sin_yaw
        local_y = -dx * sin_yaw + dy * cos_yaw
        
        # Distances (vectorized)
        distances = np.sqrt(local_x**2 + local_y**2)
        
        # Find valid points: in front AND near lookahead
        valid = (local_x > 0.5) & (distances >= self.lookahead_distance * 0.9)
        
        if np.any(valid):
            # Closest valid point to lookahead distance
            diff = np.abs(distances - self.lookahead_distance)
            diff[~valid] = np.inf
            target_idx = np.argmin(diff)
        else:
            # Fallback: furthest point ahead
            target_idx = np.argmax(local_x)
        
        # Get target coordinates
        lx = local_x[target_idx]
        ly = local_y[target_idx]
        
        # Pure pursuit steering
        L_sq = lx**2 + ly**2
        steering_angle = math.atan((2.0 * self.wheelbase * ly) / L_sq)
        
        # Publish marker
        self.publish_marker(self.wp_x[target_idx], self.wp_y[target_idx])
        
        # Send drive command
        drive_msg = AckermannDriveStamped()
        drive_msg.header.stamp = rospy.Time.now()
        drive_msg.drive.speed = self.max_speed
        drive_msg.drive.steering_angle = np.clip(steering_angle, -self.max_steering_angle, self.max_steering_angle)
        self.drive_pub.publish(drive_msg)

    def publish_marker(self, x, y):
        m = Marker()
        m.header.frame_id = "map"
        m.header.stamp = rospy.Time.now()
        m.id = 0
        m.type = Marker.SPHERE
        m.action = Marker.ADD
        m.pose.position.x = x
        m.pose.position.y = y
        m.pose.position.z = 0.0
        m.pose.orientation.w = 1.0
        m.scale.x = 0.5
        m.scale.y = 0.5
        m.scale.z = 0.5
        m.color.a = 1.0
        m.color.r = 0.0
        m.color.g = 1.0
        m.color.b = 0.0
        m.lifetime = rospy.Duration(0)
        self.viz_pub.publish(m)

def main():
    rospy.init_node('pure_pursuit_node')
    pp = PurePursuit()
    rospy.spin()

if __name__ == '__main__':
    main()