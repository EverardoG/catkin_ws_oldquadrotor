import cv2

class VideoFeed():
    def __init__(self):
        self.name = "Video Feed"
        self.camera = 0
        cv2.namedWindow(self.name)
        self.cap = cv2.VideoCapture(self.camera)
        self.ret, self.frame = self.cap.read()

    def refresh(self):
        self.ret, self.frame = self.cap.read()
        cv2.imshow(self.name,self.frame)
        if cv2.waitKey(1) == 27:
            raise StopVideo

    def loop(self):
        while self.ret:
            try:
                live_video.refresh()
            except StopVideo:
                break

class StopVideo(Exception):
    pass

if __name__ == '__main__':
    live_video = VideoFeed()
    live_video.loop()
