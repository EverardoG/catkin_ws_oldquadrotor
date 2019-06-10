#!/usr/bin/env python

from cv2 import ORB_create, BFMatcher, NORM_L1, drawMatches, imshow, waitKey, namedWindow, cvtColor, COLOR_RGB2BGR, COLOR_BGR2GRAY#computper vision
import rospy #ROS
from sensor_msgs.msg import Image #ROS msg
from geometry_msgs.msg import Twist
from numpy import asarray, ones, concatenate, sqrt, mean, square, fromstring, reshape, uint8 #numpy - matrix operations
from scipy import ndimage #convolution function
from matplotlib import pyplot as plt #plot function
from drawnow import drawnow #live plotting
from time import time #keep track of time

class ROS_Video_Node():
    def __init__(self,name):
        """This initializes the object"""
        self.name = name
        self.last_frame = None
        self.current_frame = None
        self.match_frames = None

        #Create an ORB object for keypoint detection
        self.orb = ORB_create(nfeatures=100,scaleFactor=2,edgeThreshold=100,fastThreshold=10)

        #Create a BF object for keypoint matching
        self.bf = BFMatcher(NORM_L1,crossCheck=True)

        #For keeping track of the drone's current velocity
        self.vel_x = None
        self.vel_y = None

        #For plotting the drone's current and past velocities
        self.times = []
        self.x_s = []
        self.y_s = []

        #Creating a publisher that will publish the calculated velocity to the ROS topic /quadrotor/ardrone/calc_vel
        self.pub = rospy.Publisher("/quadrotor/ardrone/calc_vel",Twist,queue_size = 10)

    def detect_motion(self):
        """This function detects the motion between the current and last frame."""
        if (self.last_frame == self.current_frame).all():
            print("Beep boop! I think my current frame is exactly the same as my last frame. \nEither I'm not moving, or something is very wrong.\n")
            namedWindow(self.name)
            imshow(self.name,self.current_frame)
            waitKey(1)
            return None

        #This finds the keypoints and descriptors with SIFT
        kp1, des1 = self.orb.detectAndCompute(self.last_frame,None)
        kp2, des2 = self.orb.detectAndCompute(self.current_frame, None)

        #Match descriptors
        matches = self.bf.match(des1,des2)

        #Sort them in the order of their distance
        matches = sorted(matches, key = lambda x:x.distance)

        #This is a test to see if I can filter out bad matches from the getgo
        #Right now I've set a threshold that I think will keep only bad matches to see if it makes the data super noisy
        # print("MATCH DISTANCES")
        #undoiing test, ELEPHANT
        filtered_matches = matches
        # filtered_matches = []
        # for match in matches:
        #     if match.distance > 1000:
        #         filtered_matches.append(match)

        if len(filtered_matches) < 5:
            print("Beep boop! Not enough good matches were found. This data is unreliable. \n Try moving me above the building, please.\n")
            namedWindow(self.name)
            imshow(self.name,self.current_frame)
            waitKey(1)
            return None

        #create arrays of the coordinates for keypoints in the two frames
        kp1_coords = asarray([kp1[mat.queryIdx].pt for mat in filtered_matches])
        kp2_coords = asarray([kp2[mat.trainIdx].pt for mat in filtered_matches])

        #calculate the translations needed to get from the first set of keypoints to the next set of keypoints
        translations = kp2_coords - kp1_coords

        least_error = 5
        for translation in translations:
            a = translation[0] * ones((translations.shape[0],1))
            b = translation[1] * ones((translations.shape[0],1))
            test_translation = concatenate((a,b),axis=1)
            test_coords = kp1_coords + test_translation
            error = sqrt(mean(square(kp2_coords - test_coords)))
            if error < least_error:
                least_error = error
                best_translation = translation

        self.vel_x = translation[0]
        self.vel_y = translation[1]



        # Draw first matches.
        draw_params = dict(matchColor = (0,255,0), # draw matches in green color
                        singlePointColor = None,
                        matchesMask = None, # draw only inliers
                        flags = 2)
        self.match_frames = drawMatches(self.last_frame,kp1,self.current_frame,kp2,matches, None, **draw_params)

        namedWindow(self.name)
        imshow(self.name,self.match_frames)
        waitKey(1)

        if least_error < 2:
            print("This is a new match")
            print("X velocity: ", self.vel_x)
            print("Y velocity: ", self.vel_y)
            print("least_error: ",least_error)

            #Publish the velocities found
            calc_vel = Twist()
            calc_vel.linear.x = self.vel_x
            calc_vel.linear.y = self.vel_y

            self.pub.publish(calc_vel)

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


def callback(msg,ros_video_node):
    #this loads in the Image msg from ROS and turns it into a numpy array
    image_vector = fromstring(msg.data,uint8)
    image_array = reshape(image_vector,(msg.height,msg.width,3))

    #this recolors the image from RGB to grayscale
    rgb_image = cvtColor(image_array,COLOR_RGB2BGR)
    gray_image = cvtColor(rgb_image,COLOR_BGR2GRAY)

    #set this image to the current frame
    ros_video_node.current_frame = gray_image

    try:
        ros_video_node.detect_motion()
    except Exception as e:
        print(e)

    #that same frame is now the last frame
    ros_video_node.last_frame = ros_video_node.current_frame


def main():
    #Initialize our ROS node
    rospy.init_node('camera_vis_bottom_orb',anonymous=True)

    #Create a ROS_Video_Node object
    orb_cam = ROS_Video_Node(name="Orb Cam")

    #Subscribe to the ROS topic /quadrotor/ardrone/bottom/ardrone/bottom/image_raw
    rospy.Subscriber("/quadrotor/ardrone/bottom/ardrone/bottom/image_raw", Image, callback, orb_cam)

    #keep running until ROS shuts down
    rospy.spin()

if __name__ == '__main__':
    main()
