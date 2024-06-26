import os
import random
import cv2

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

root = '/nas/EOS/users/siva/data/carla_data/simulations/rand_weather_camera_pert_2'
seq = 'night_Town06'
rgb_dir = os.path.join(root,seq)
gt_dir = os.path.join(root,seq,'cam_poses')
savedir = "/nas/EOS/users/aman/results"

fnames = sorted(os.listdir(gt_dir))
img_fids = list(map(lambda x : x.split('.')[0],fnames))

print(img_fids)
# cam_left,cam_right = 0,2

# vid_writer = Videowriter(2560,1440,f'carla_{seq}_cam{cam_left}_cam{cam_right}_pert.mp4',savedir,frame_rate=5)

# for img_fid in img_fids[:100]:
#     image_left_pb = cv2.imread(os.path.join(rgb_dir,f'cam_p_{cam_left}',img_fid+'.png'))
#     vid_writer.add_frame(image_left_pb)

# vid_writer.checkout_video()

