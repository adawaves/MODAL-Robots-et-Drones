#!/usr/bin/env python3
import sys
import math
import rospy
from sensor_msgs.msg import LaserScan
from ackermann_msgs.msg import AckermannDriveStamped

class WallFollow:
    def __init__(self):
        rospy.init_node("WallFollow_node", anonymous=True)
        
        self.drive_pub = rospy.Publisher("/nav", AckermannDriveStamped, queue_size=10)
        self.lidar_sub = rospy.Subscriber("/scan", LaserScan, self.lidar_callback)

        self.kp = 0.8
        self.kd = 0.3
        self.ki = 0.0
        self.DESIRED_DISTANCE = 0.60 # Distance souhaitée 
        self.VELOCITY = 4
        self.LOOKAHEAD_DIST = 1.0
        
        self.prev_error = 0.0 
        self.integral = 0.0
        
        # "LEFT", "RIGHT"
        self.side = None 

        # Angles (0=Face, 90=Gauche, -90=Droite)
        self.angle_b_left = 90.0
        self.angle_a_left = 40.0
        self.angle_b_right = -90.0
        self.angle_a_right = -40.0

    def getRange(self, data, angle_deg):
        target_rad = math.radians(angle_deg)
        index = int((target_rad - data.angle_min) / data.angle_increment)

        if index < 0 or index >= len(data.ranges):
            return 10.0
        dist = data.ranges[index]

        if math.isinf(dist) or math.isnan(dist) or dist == 0:
            return 10.0
        
        return dist

    def calculate_error(self, data, angle_a, angle_b, side):
        """ Calcule l'erreur de distance pour un côté donné """

        a = self.getRange(data, angle_a)
        b = self.getRange(data, angle_b)
        theta = math.radians(abs(angle_b - angle_a))

        # Formule géométrique du Wall Following
        alpha = math.atan((a * math.cos(theta) - b) / (a * math.sin(theta)))
        dist_current = b * math.cos(alpha)
        dist_predicted = dist_current + self.LOOKAHEAD_DIST * math.sin(alpha)
        
        error = self.DESIRED_DISTANCE - dist_predicted
        # Pour la droite, l'erreur doit être inversée 
        return error if side == "LEFT" else -error

    def lidar_callback(self, data):
        dist_left = self.getRange(data, self.angle_b_left)
        dist_right = self.getRange(data, self.angle_b_right)

        # Si on ne suit rien, ou si le mur actuel est trop loin (> 2m)
        if self.side is None or (self.side == "LEFT" and dist_left > 2.0) or (self.side == "RIGHT" and dist_right > 2.0):
            if dist_left < dist_right:
                self.side = "LEFT"
                rospy.loginfo("Nouveau mur détecté à GAUCHE")
            else:
                self.side = "RIGHT"
                rospy.loginfo("Nouveau mur détecté à DROITE")

        # Calcul de l'erreur selon le côté choisi
        if self.side == "LEFT":
            error = self.calculate_error(data, self.angle_a_left, self.angle_b_left, "LEFT")
        else:
            error = self.calculate_error(data, self.angle_a_right, self.angle_b_right, "RIGHT")

        self.pid_control(error)

    def pid_control(self, error):
        P = self.kp * error
        self.integral += error
        D = self.kd * (error - self.prev_error)
        self.prev_error = error

        steering_angle = - (P + D + (self.ki * self.integral))

        # Vitesse dynamique
        velocity = self.VELOCITY
        if abs(steering_angle) > 0.3: velocity *= 0.5
        elif abs(steering_angle) > 0.15: velocity *= 0.75

        drive_msg = AckermannDriveStamped()
        drive_msg.header.stamp = rospy.Time.now()
        drive_msg.drive.steering_angle = steering_angle
        drive_msg.drive.speed = velocity
        self.drive_pub.publish(drive_msg)

def main(args):
    wf = WallFollow()
    rospy.spin()

if __name__=='__main__':
    main(sys.argv)