#!/usr/bin/env python3
import rospy
import numpy as np
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from ackermann_msgs.msg import AckermannDriveStamped
from std_msgs.msg import Bool

class Safety(object):
    """
    The class that handles emergency braking.
    """
    def __init__(self):
        """
        One publisher should publish to the /brake topic with a AckermannDriveStamped brake message.

        One publisher should publish to the /brake_bool topic with a Bool message.

        You should also subscribe to the /scan topic to get the LaserScan messages and
        the /odom topic to get the current speed of the vehicle.

        The subscribers should use the provided odom_callback and scan_callback as callback methods

        NOTE that the x component of the linear velocity in odom is the speed
        """
        self.speed = 0
        # TODO: create ROS subscribers and publishers.
        self.pub1 = rospy.Publisher("/brake", AckermannDriveStamped, queue_size=10)
        self.pub2 = rospy.Publisher("/brake_bool", Bool, queue_size=10)

        rospy.Subscriber("/scan", LaserScan, self.scan_callback)
        rospy.Subscriber("/odom", Odometry, self.odom_callback)


    def odom_callback(self, odom_msg):
        # TODO: update current speed
        self.speed = odom_msg.twist.twist.linear.x

    def scan_callback(self, scan_msg):

        minang = - np.pi/12
        maxang = np.pi/12

        start = int((minang - scan_msg.angle_min) / scan_msg.angle_increment)
        end= int((maxang - scan_msg.angle_min) / scan_msg.angle_increment)

        # Petite sécurité pour ne pas sortir du tableau
        start = max(0, start)
        end = min(len(scan_msg.ranges), end)


        distances= np.array(scan_msg.ranges[start:end])

        angles = scan_msg.angle_min + np.arange(start, end) * scan_msg.angle_increment


        vitesses_proj = self.speed * np.cos(angles)
        vitesses_proj = np.maximum(vitesses_proj, 0.001)

        ttc = distances/vitesses_proj

        # TODO: publish brake message and publish controller bool
        if np.min(ttc)< 0.5:
            bool_msg = Bool()
            bool_msg.data = True
            self.pub2.publish(bool_msg)

            brake_msg = AckermannDriveStamped()
            brake_msg.drive.speed = 0.0
            self.pub1.publish(brake_msg)
        
        else:
            bool_msg = Bool()
            bool_msg.data = False
            self.pub2.publish(bool_msg)


def main():
    rospy.init_node('safety_node')
    sn = Safety()
    rospy.spin()

if __name__ == '__main__':
    main()
