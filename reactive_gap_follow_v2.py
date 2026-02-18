#!/usr/bin/env python3
import sys
import math
import numpy as np
import rospy
from sensor_msgs.msg import LaserScan
from ackermann_msgs.msg import AckermannDriveStamped

class ReactiveFollowGap:
    def __init__(self):
        self.lidar_sub = rospy.Subscriber('/scan', LaserScan, self.lidar_callback)
        self.drive_pub = rospy.Publisher('/nav', AckermannDriveStamped, queue_size=10)
        
        # Paramètres de contrôle
        self.max_steering_angle = 0.4189
        self.min_alpha, self.max_alpha = 0.3, 0.8
        self.prev_steering = 0.0
        
        # Paramètres de navigation - Boostés pour la course
        self.gap_threshold = 0.3
        self.max_lookahead = 1.5
        self.max_speed = 1.5 # Augmenté
        self.car_width = 0.55 # Légèrement augmenté pour plus de marge de sécurité

    def preprocess_lidar(self, ranges):
        proc_ranges = np.array(ranges)
        proc_ranges[np.isinf(proc_ranges)] = 10.0
        proc_ranges[np.isnan(proc_ranges)] = 0.0
        return np.clip(proc_ranges, 0, self.max_lookahead)

    def find_max_gap(self, free_space_ranges):
        mask = free_space_ranges > self.gap_threshold
        if not np.any(mask): return None, None
        padded = np.concatenate(([False], mask, [False]))
        diff = np.diff(padded.astype(int))
        starts = np.where(diff == 1)[0]
        ends = np.where(diff == -1)[0]
        longest_idx = np.argmax(ends - starts)
        return starts[longest_idx], ends[longest_idx]

    def find_best_point(self, start_i, end_i, ranges):
        gap_ranges = ranges[start_i:end_i]
        max_val = np.max(gap_ranges)
        max_indices = np.where(gap_ranges == max_val)[0] + start_i
        deepest_point = np.mean(max_indices)
        center_idx = (start_i + end_i) / 2.0
        return int(0.5 * deepest_point + 0.5 * center_idx)
    
    def lidar_callback(self, data):
        angle_range = math.radians(70) 
        n = int(angle_range / data.angle_increment)
        mid = len(data.ranges) // 2
        proc_ranges = self.preprocess_lidar(data.ranges[mid-n : mid+n])

        # Bulle de sécurité
        closest_idx = np.argmin(proc_ranges)
        min_dist = proc_ranges[closest_idx]
        if min_dist > 0:
            radius_idx = int((self.car_width / 2.0) / min_dist / data.angle_increment)
            radius_idx = min(radius_idx, 150) 
            start_b = max(closest_idx - radius_idx, 0)
            end_b = min(closest_idx + radius_idx, len(proc_ranges))
            proc_ranges[start_b : end_b] = 0.0

        # Recherche du Gap
        start_gap, end_gap = self.find_max_gap(proc_ranges)
        
        if start_gap is None:
            drive_msg = AckermannDriveStamped()
            drive_msg.header = data.header
            drive_msg.drive.speed = -1.0 # Frein d'urgence
            self.drive_pub.publish(drive_msg)
            return
        
        best_idx = self.find_best_point(start_gap, end_gap, proc_ranges)
        target_angle = (-angle_range) + (best_idx * data.angle_increment)

        # Lissage Steering
        alpha = self.min_alpha + (self.max_alpha - self.min_alpha) * (abs(target_angle) / self.max_steering_angle)
        steering = (alpha * target_angle) + ((1.0 - alpha) * self.prev_steering)
        self.prev_steering = steering

        # --- GESTION DE VITESSE HYBRIDE ---
        
        # 1. Ta base fluide en Cosinus
        speed = self.max_speed * math.cos(steering)**2

        # 2. La chute soudaine (Quick Drop) si le virage est trop brusque
        # À plus de 0.20 rad (~11°), on considère que le virage est dangereux à 6m/s
        abs_steering = abs(steering)
        if abs_steering > 0.30:     # Virage serré
            speed = min(speed, 2.0)
        elif abs_steering > 0.18:   # Virage modéré
            speed = min(speed, 3.5)

        # Plancher de vitesse
        speed = max(speed, 0.5)

        # Publication
        drive_msg = AckermannDriveStamped()
        drive_msg.header = data.header
        drive_msg.drive.steering_angle = steering
        drive_msg.drive.speed = speed
        self.drive_pub.publish(drive_msg)

def main(args):
    rospy.init_node("FollowGap_node", anonymous=True)
    rfgs = ReactiveFollowGap()
    rospy.spin()

if __name__ == '__main__':
    main(sys.argv)