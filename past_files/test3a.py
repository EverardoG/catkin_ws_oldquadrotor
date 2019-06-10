import numpy as np
import rospy
from sensor_msgs.msg import Image
import cv2

class VideoFeed():
    def __init__(self):
        self.name = "Video Feed"
        # self.camera = 0
        cv2.namedWindow(self.name)
        # self.cap = cv2.VideoCapture(self.camera)
        # self.ret, self.frame = self.cap.read()

    def refresh(self):
        # self.ret, self.frame = self.cap.read()
        image = cv2.imread('dangit.png')
        cv2.imshow(self.name,image)
        cv2.waitKey(25)

    def loop(self):
        while 1:
            try:
                live_video.refresh()
            except StopVideo:
                break

class StopVideo(Exception):
    pass

def callback(msg,VideoFeed_obj):
    image_vector = np.fromstring(msg.data,np.uint8) #turn the ROS uint8[] message into a vector with all the info we need
    image_array = np.reshape(image_vector,(msg.height,msg.width,3)) #reshape the vector into a height x width x color channels array
    VideoFeed_obj.refresh()

def main(VideoFeed_obj):
    rospy.init_node('camera_visualizer') #initialize our ROS node
    rospy.Subscriber("/ardrone/image_raw", Image, callback, VideoFeed_obj) #Subscribe to the ROS topic /ardrone/image_raw
    rospy.spin()

if __name__ == '__main__':
    live_video = VideoFeed()
    main(live_video)
