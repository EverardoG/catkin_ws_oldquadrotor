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

    #we can blur our image here
    g_matrix = 0.125*np.array([[1,2,1],[2,4,2],[1,2,1]])
    blurred_image = np.asarray(ndimage.convolve(gray_image, g_matrix, mode="constant",cval=0, origin=0))

    #we can adjust our image here
    beta = 0 #changes brightness
    gamma = 80#changes contrast
    adjusted_image = np.where((255-blurred_image)<beta,255,blurred_image+beta)/gamma

    #this part does the actual edge detection
    e_matrix = np.array([[0, -1, 0],[-1, 4, -1],[0, -1, 0]])
    edge_image = ndimage.convolve(adjusted_image, e_matrix, mode="constant",cval=0, origin=0)


    #Uncomment the code below for a different edge detection algorithm!
    # edge_layers = np.array(np.gradient(blurred_image))
    # edge_image = np.sqrt(edge_layers[0,:,:]**2 + edge_layers[1,:,:]**2)

    #This recolors the image to make it look like the matrix
    # recolored_image = cv2.cvtColor(edge_image,cv2.COLOR_GRAY2RGB)
    # recolored_image[:,:,0][recolored_image[:,:,0]>128]=17
    # recolored_image[:,:,1][recolored_image[:,:,1]>128]=143
    # recolored_image[:,:,2][recolored_image[:,:,2]>128]=0

    #This displays the image in a window
    cv2.namedWindow(name)
    cv2.imshow(name,edge_image)
    if cv2.waitKey(1) == 27:
        import sys
        sys.exit(0)

def main():
    #Initialize our ROS node
    rospy.init_node('camera_vis_bottom')

    #Subscribe to the ROS topic /quadrotor/ardrone/bottom/ardrone/bottom/image_raw
    rospy.Subscriber("/quadrotor/ardrone/bottom/ardrone/bottom/image_raw", Image, callback, "Bottom")

    #keep checking for new messages
    rospy.spin()

if __name__ == '__main__':
    main()
