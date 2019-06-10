#!/usr/bin/env python

import rospy
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui
import numpy as np
from geometry_msgs.msg import Twist
from time import time #keep track of time
import sys

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

        #Setting up Window for plots
        self.win = pg.GraphicsWindow()
        self.win.setWindowTitle('Live X and Y Velcoities')

        #setting up Plots
        self.p_x = self.win.addPlot()
        self.p_y = self.win.addPlot()

        #Plotting velocities (which are currently empty)
        self.curve_x = self.p_x.plot(self.x_s)
        self.curve_y = self.p_y.plot(self.y_s)

        self.ptr = 0

    def update(self):
        self.curve_x.setData(self.x_s)
        self.curve_y.setData(self.y_s)
        self.curve_y.setPos(1,0)

def callback(msg,ros_plot_node):
    #this loads in the twist msg from ROS
    ros_plot_node.vel_x = msg.linear.x
    ros_plot_node.vel_y = msg.linear.y

    ros_plot_node.x_s.append(ros_plot_node.vel_x)
    ros_plot_node.y_s.append(ros_plot_node.vel_y)

    try:
        ros_plot_node.x_s = ros_plot_node.x_s[-20:]
        ros_plot_node.y_s = ros_plot_node.y_s[-20:]
    except:
        pass

#Create ros plot object
vel_plot = ROS_Plot_Node(name='Velcoity Plotter')

#Create a timer that will update our plots
timer = pg.QtCore.QTimer()
timer.timeout.connect(vel_plot.update)
timer.start(50)

#Initialize ros node
rospy.init_node('calc_vel_plot', anonymous=True)
#Subscribe to appropriate ros topic
rospy.Subscriber("/quadrotor/ardrone/calc_vel", Twist, callback, vel_plot)
if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
        QtGui.QApplication.instance().exec_()
#keep running until ROS shuts down
rospy.spin()
