#!/usr/bin/env python

import cv2
import rospy
from sensor_msgs.msg import Image
import numpy as np
from scipy import ndimage

class ROS_video_topic():
    def __init__(self,topic,name,norm=45):
        self.topic = topic
        self.name = name
        self.norm = norm

def callback(msg,name):
    print("Camera: ",name)
    image_vector = np.fromstring(msg.data,np.uint8) #turn the ROS uint8[] message into a vector with all the info we need
    image_array = np.reshape(image_vector,(msg.height,msg.width,3)) #reshape the vector into a height x width x color channels array
    rgb_image = cv2.cvtColor(image_array,cv2.COLOR_RGB2BGR)
    gray_image = cv2.cvtColor(rgb_image,cv2.COLOR_BGR2GRAY)
    flat_gray = gray_image.flatten()
    flat_gray.sort()

    # print("The max value in gray_image: ",np.amax(gray_image))
    # print("The min value in gray_image: ",np.amin(gray_image))
    # print("All elements in gray_image are: ", flat_gray)

    adjusted_image = (gray_image+200)/45
    # print("The max value in adjusted_image: ",np.amax(adjusted_image))
    # print("The min value in adjusted_image: ",np.amin(adjusted_image))
    # flat_adj = adjusted_image.flatten()
    # flat_adj.sort()
    # print("All elements in adjusted_image are: ",flat_adj)

    # all_val = []
    # for val in flat_adj:
    #     if val not in all_val:
    #         all_val.append(val)
    # print(all_val)



    # c_matrix = np.array([[0, 1, 0],[1, -4, 1],[0, 1, 0]])
    c_matrix = np.array([[-1, -1, -1],[-1, 8, -1],[-1, -1, -1]])
    convolved_img = ndimage.convolve(adjusted_image, c_matrix, mode="constant",cval=0, origin=0)
    # gradients = np.array(np.gradient(convolved_img))
    # gradients_sum = np.sqrt(gradients[0,:,:]**2 + gradients[1,:,:]**2)

    # displayed_img = cv2.applyColorMap(convolved_img,cv2.COLORMAP_JET)

    cv2.namedWindow(name)
    cv2.imshow(name,convolved_img)
    if cv2.waitKey(1) == 27:
        import sys
        sys.exit(0)

def main():
    # extrastr="working"
    # normalizer=int(raw_input("Type in a normalizer for the gray image here. (int) \n"))
    rospy.init_node('camera_visualizer') #initialize our ROS node
    # front_camera = ROS_video_topic("/ardrone/image_raw","Front Camera")

    rospy.Subscriber("/quadrotor/ardrone/front/ardrone/front/image_raw", Image, callback, "Front")#, raw_input("Type normalizer here\n")) #Subscribe to the ROS topic /ardrone/image_raw
    rospy.spin()

if __name__ == '__main__':
    main()
