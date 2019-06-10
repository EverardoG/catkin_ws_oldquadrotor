#!/usr/bin/env python

import rospy
from matplotlib import pyplot as plt #plot function
from drawnow import drawnow #live plotting
from geometry_msgs.msg import Twist
from time import time #keep track of time

class ROS_Plot_Node():
    def __init__(self,name):
        """This initializes the object"""
        self.name = name

        #For keeping track of the drone's current velocity
        self.vel_x = None
        self.vel_y = None

        #For plotting the drone's current and past velocities
        self.times = []
        self.x_s = []
        self.y_s = []

    def plot_current_vel(self):
        """This function live plots the quadrotor's measured velocity"""
        #save the current velocities
        self.times.append(time())
        self.x_s.append(self.vel_x)
        self.y_s.append(self.vel_y)

        #plot the velocities
        plt.title("Live Velocities")
        plt.ylabel("Velocity")
        plt.xlabel("Time")
        plt.ylim(-20,20)
        try:
            plt.plot(self.times[-60:],self.x_s[-60:],'r-',label="X Velocity")
            plt.plot(self.times[-60:],self.y_s[-60:],'b-',label="Y Velocity")
        except:
            plt.plot(self.times,self.x_s,'r-',label="X Velocity")
            plt.plot(self.times,self.y_s,'b-',label="Y Velocity")
        plt.legend(loc="upper right")

def callback(msg,ros_plot_node):
    #this loads in the twist msg from ROS
    ros_plot_node.vel_x = msg.linear.x
    ros_plot_node.vel_y = msg.linear.y

    drawnow(ros_plot_node.plot_current_vel)

def main():
    #Initialize ros node
    rospy.init_node('calc_vel_plot', anonymous=True)

    #Create ros plot object
    vel_plot = ROS_Plot_Node(name='Velcoity Plotter')

    #Subscrive to appropriate ros topic
    rospy.Subscriber("/quadrotor/ardrone/calc_vel", Twist, callback, vel_plot)

    #keep running until ROS shuts down
    rospy.spin()

if __name__ == '__main__':
    main()
