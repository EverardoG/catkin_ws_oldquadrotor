#!/usr/bin/env python

import cv2
import rospy
from sensor_msgs.msg import Image #this is the ROS message type we want to work with
import numpy as np
from scipy import ndimage

class ROS_video_topic():
    def __init__(self,topic,name,norm=45):
        self.topic = topic
        self.name = name
        self.norm = norm

def callback(msg,name):
    print(name)
    image_vector = np.fromstring(msg.data,np.uint8) #turn the ROS uint8[] message into a vector with all the info we need
    image_array = np.reshape(image_vector,(msg.height,msg.width,3)) #reshape the vector into a height x width x color channels array
    rgb_image = cv2.cvtColor(image_array,cv2.COLOR_RGB2BGR)
    gray_image = (cv2.cvtColor(rgb_image,cv2.COLOR_BGR2GRAY)+200)/45
    c_matrix = np.array([[-1, -1, -1],[-1, 8, -1],[-1, -1, -1]])
    # c_matrix = np.array([[0, 1, 0],[1, -4, 1],[0, 1, 0]])
    convolved_img = ndimage.convolve(gray_image, c_matrix, mode="constant",cval=0, origin=0)
    # gradients = np.array(np.gradient(convolved_img))
    # gradients_sum = np.sqrt(gradients[0,:,:]**2 + gradients[1,:,:]**2)


    cv2.namedWindow(name)
    cv2.imshow(name,convolved_img)
    if cv2.waitKey(1) == 27:
        import sys
        sys.exit(0)

def main():
    # extrastr="working"
    # normalizer=int(raw_input("Type in a normalizer for the gray image here. (int) \n"))
    rospy.init_node('camera_visualizer_bottom') #initialize our ROS node
    # front_camera = ROS_video_topic("/ardrone/image_raw","Front Camera")

    rospy.Subscriber("/quadrotor/ardrone/bottom/ardrone/bottom/image_raw", Image, callback, "Bottom")#, raw_input("Type normalizer here\n")) #Subscribe to the ROS topic /ardrone/image_raw
    rospy.spin()

if __name__ == '__main__':
    main()
