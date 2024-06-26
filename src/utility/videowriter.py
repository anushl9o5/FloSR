import cv2
import os 
class Videowriter(object):
    def __init__(self,img_width,img_height,vid_name,vid_path,frame_rate=5):
        self.w, self.h = img_width,img_height
        self.vid_name = vid_name
        self.vid_path = vid_path
        self.writer = cv2.VideoWriter(os.path.join(self.vid_path, self.vid_name),0x7634706d, frame_rate, (self.w,self.h))

    def add_frame(self,frame):
        self.writer.write(frame)
    
    def checkout_video(self):
        self.writer.release()

