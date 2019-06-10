import cv2
import rospy
from sensor_msgs.msg import Image #this is the ROS message type we want to work with
import numpy as np

class ROS_video_topic():
    def __init__(self,topic,name):
        self.topic = topic
        self.name = name


def callback(msg,extrastr):
    print(extrastr)
    image_vector = np.fromstring(msg.data,np.uint8) #turn the ROS uint8[] message into a vector with all the info we need
    image_array = np.reshape(image_vector,(msg.height,msg.width,3)) #reshape the vector into a height x width x color channels array
    cv2.namedWindow("Video Feed")
    cv2.imshow("Video Feed",image_array)
    if cv2.waitKey(1) == 27:
        pass
def main():
    extrastr="hello"
    rospy.init_node('camera_visualizer') #initialize our ROS node
    # front_camera = ROS_video_topic("/ardrone/image_raw","Front Camera")

    rospy.Subscriber("/quadrotor/ardrone/front/ardrone/front/image_raw", Image, callback, extrastr) #Subscribe to the ROS topic /ardrone/image_raw
    rospy.spin()

if __name__ == '__main__':
    main()
