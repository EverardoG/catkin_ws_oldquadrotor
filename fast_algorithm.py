#!/usr/bin/env python

import cv2 #opencv
import rospy #ROS
from sensor_msgs.msg import Image #ROS msg
import numpy as np #nupmy
from scipy import ndimage #convolution function

def callback(msg,name):
    print("Camera: ",name)

    #this loads in the Image msg from ROS and turns it into a numpy array
    image_vector = np.fromstring(msg.data,np.uint8)
    image_array = np.reshape(image_vector,(msg.height,msg.width,3))

    #this turns the numpy array into an open cv image
    rgb_image = cv2.cvtColor(image_array,cv2.COLOR_RGB2BGR)
    gray_image = cv2.cvtColor(rgb_image,cv2.COLOR_BGR2GRAY)

    # Initiate FAST object with default values
    fast = cv2.FastFeatureDetector().create(25)

    # find and draw the keypoints
    kp = fast.detect(gray_image, None)
    fast_image = cv2.drawKeypoints(gray_image, kp, None, color=(255,0,0))

    #This displays the image in a window
    cv2.namedWindow(name)
    cv2.imshow(name,fast_image)
    if cv2.waitKey(1) == 27:
        import sys
        sys.exit(0)

def main():
    rospy.init_node('fast_front') #initialize our ROS node
    rospy.Subscriber("/quadrotor/ardrone/front/ardrone/front/image_raw", Image, callback, "FAST Front")#, raw_input("Type normalizer here\n")) #Subscribe to the ROS topic /ardrone/image_raw
    rospy.spin()

if __name__ == '__main__':
    main()
