#!/usr/bin/env python

import numpy as np #for dealing with image arrays
import rospy #so we can communicate with ROS
from sensor_msgs.msg import Image #this is the ROS message type we want to work with
import matplotlib.pyplot as plt #this is so we can plot our image
# import matplotlib.image as mpimg
from drawnow import *

image_array = 0 #initialize our image array as a global variable


def makeFig():
    """ This is where we define the figure visualized for drawnow"""
    plt.imshow(image_array) #plot the current image

def callback(msg):
    global image_array #use the global variable image_array
    image_vector = np.fromstring(msg.data,np.uint8) #turn the ROS uint8[] message into a vector with all the info we need
    image_array = np.reshape(image_vector,(msg.height,msg.width,3)) #reshape the vector into a height x width x color channels array
    drawnow(makeFig) #update the figure

def main():
    rospy.init_node('camera_visualizer') #initialize our ROS node
    rospy.Subscriber("/quadrotor/ardrone/front/ardrone/front/image_raw", Image, callback) #Subscribe to the ROS topic /ardrone/image_raw
    rospy.spin() #keeps python from exiting until the ROS node has stopped

if __name__ == '__main__': #don't run unless this is file is being executed directly through python
    main() #run main()
