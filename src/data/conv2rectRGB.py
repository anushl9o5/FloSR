import os
import cv2
import math
from imageio import imread, imwrite
import pickle
import skimage.io as skio
import glob
from raw2rgb_utils import convert_raw2rgb_fast, read_raw,convert_raw2rgb
from conti_raw2rgb import load_conti_image
from ar0820_raw2rgb import load_ar0820_image
from tqdm.contrib import tzip
from tqdm import tqdm
# from concurrent.futures import ProcessPoolExecutor, as_completed
import argparse

from time import time
from multiprocessing import Process, Queue
from joblib import Parallel, delayed, cpu_count

from threading import Thread

def makedirs(path):
    os.umask(0o000)
    os.makedirs(path, exist_ok=True, mode=0o775)

def load_rectmaps(filename):
    with open(filename, 'rb') as f:
        rect_map = pickle.load(f)
    lmapx = rect_map['left']['mapx']
    lmapy = rect_map['left']['mapy']
    rmapx = rect_map['right']['mapx']
    rmapy = rect_map['right']['mapy']
    return lmapx, lmapy, rmapx, rmapy

def rectify(im_ref, im_follow, rectmaps):
    lmapx, lmapy, rmapx, rmapy = rectmaps
    im_ref_rect = cv2.remap(im_ref, lmapx, lmapy, cv2.INTER_LINEAR)
    im_follow_rect = cv2.remap(im_follow, rmapx, rmapy, cv2.INTER_LINEAR)
    return im_ref_rect, im_follow_rect

def generate_rgb_image(left_raw_path, right_raw_path, camtype, is_rect, use_fast):
    """This function reads left and right raw images and convert them 
       into rgb images and rectify them using pre calculated rect maps.

    Args:
        left_raw_path ([str]): left raw image path
        right_raw_path ([str]): right raw image path
        use_fast (bool, optional): [skip decompanding for raw to rgb conversion]. Defaults to True.

    Returns:
        None
    """

    if camtype == 'fsc231':
        left_rgb = load_conti_image(left_raw_path)
        right_rgb = load_conti_image(right_raw_path)
    elif camtype == 'ar0231':
        # Read RAW image
        left_raw = read_raw(left_raw_path)
        right_raw = read_raw(right_raw_path)

        # Convert to rgb 
        if use_fast:
            left_rgb = convert_raw2rgb_fast(left_raw)
            right_rgb = convert_raw2rgb_fast(right_raw)
        else:
            left_rgb = convert_raw2rgb(left_raw)
            right_rgb = convert_raw2rgb(right_raw)
    elif camtype == 'ar0820':
        left_rgb = load_ar0820_image(left_raw_path)
        right_rgb = load_ar0820_image(right_raw_path)
    else:
        raise Exception(f'camtype "{camtype}" is not supported.')

    #Rectify
    if is_rect:
        left_rgb,right_rgb = rectify(left_rgb,right_rgb,rectmaps)

    return left_rgb, right_rgb

def write_rgb_image(left_raw_path, right_raw_path, camtype, is_rect, use_fast):
    """This function reads left and right raw images and convert them 
       into rgb images and rectify them using pre calculated rect maps.

    Args:
        left_raw_path ([str]): left raw image path
        right_raw_path ([str]): right raw image path
        use_fast (bool, optional): [skip decompanding for raw to rgb conversion]. Defaults to True.

    Returns:
        None
    """
    savenmame = os.path.basename(left_raw_path).split('.')[0] + "." + args.ext
    if os.path.exists(os.path.join(savedir_left,savenmame)) and os.path.exists(os.path.join(savedir_right,savenmame)):
        return
    
    left_rgb, right_rgb = generate_rgb_image(left_raw_path, right_raw_path, camtype, is_rect, use_fast)

    # write images
    cv2.imwrite(os.path.join(savedir_left,savenmame),left_rgb)
    cv2.imwrite(os.path.join(savedir_right,savenmame),right_rgb)

def process_loop(pid, output_queue, left_paths, right_paths, camtype, is_rect, use_fast):
    for i in range(len(left_paths)):
        left_raw_path, right_raw_path = left_paths[i], right_paths[i]
        savenmame = os.path.basename(left_raw_path).split('.')[0] + "." + args.ext
        if os.path.exists(os.path.join(savedir_left,savenmame)) and os.path.exists(os.path.join(savedir_right,savenmame)):
            # print(f'[{i}] {savenmame} pass None.')
            output_queue.put(False)
        else:
            # print(f'[{i}] {savenmame}')
            left_rgb, right_rgb = generate_rgb_image(left_raw_path, right_raw_path, camtype, is_rect, use_fast)
            
            # write images
            output_queue.put({
                'left': {
                    'path':os.path.join(savedir_left,savenmame),
                    'image':left_rgb
                },
                'right': {
                    'path':os.path.join(savedir_right,savenmame),
                    'image':right_rgb
                }
            })
    print(f'[DONE] process {pid}')
        
def write_target_function(output_queue, n_files):
    for i in tqdm(range(n_files)):
        item = output_queue.get()
        # print('[writer]', item)
        if item is not False:
            cv2.imwrite(item['left']['path'],item['left']['image'])
            cv2.imwrite(item['right']['path'],item['right']['image'])

def write(Q):
    item = Q.get()
    while item is not None:
        cv2.imwrite(item['left']['path'],item['left']['image'])
        cv2.imwrite(item['right']['path'],item['right']['image'])
        item = Q.get()

class Writers:
    def __init__(self, n, Q):
        self.writers = []
        for i in range(n):
            w = Thread(target=write, args=(Q,))
            self.writers.append(w)
            w.start()
    def join(self):
        for w in self.writers:
            w.join()

if __name__=="__main__":
    parser = argparse.ArgumentParser(description="Raw to rectified stereo images")
    parser.add_argument('-d','--root_dir',      required=True,              help   ='directory containing left and right rgb images')
    parser.add_argument('-s','--result_dir',    required=True,              help   ='directory to store results')
    parser.add_argument('-l','--leftdir_name',  required=True,              help   = 'left dir name')
    parser.add_argument('-r','--rightdir_name', required=True,              help   = 'right dir name')
    parser.add_argument('-w','--workers',       default=10,   type=int,     help   ='number of workers (default is 10)')
    parser.add_argument('-c', '--camtype', choices=['ar0231', 'ar0820', 'fsc231'], help   ='select the camera type')
    parser.add_argument('-m','--rectmap_path',  default=None,              help   ='rectfication maps to rectify images')
    parser.add_argument('--start', default=0, type=int, help='start index of the list (default is 0')
    parser.add_argument('--end', default=None, type=int, help='last index of the list (default is end of the list)')
    parser.add_argument('--step', default=1, type=int, help='sampling step (default is 1)')
    parser.add_argument('--ext', default='jpg', help='output image files extension. (default is jpg)')
    parser.add_argument('--joblib', action='store_true', help='profile joblib otherwise multiprocess')
    parser.add_argument('--joblib_thread', action='store_true', help='use multithreading (instead of multiprocessing)')

    args = parser.parse_args()

    root_dir   = args.root_dir
    result_dir = args.result_dir


    savedir_left = os.path.join(result_dir, args.leftdir_name)
    savedir_right = os.path.join(result_dir,args.rightdir_name)

    makedirs(savedir_left)
    makedirs(savedir_right)

    rect = args.rectmap_path is not None
    if rect:
        rectmaps = load_rectmaps(args.rectmap_path)

    left_paths = sorted(glob.glob(os.path.join(root_dir,args.leftdir_name,"*.raw")))
    right_paths = sorted(glob.glob(os.path.join(root_dir,args.rightdir_name,"*.raw"))) 
    
    assert len(left_paths) == len(right_paths), "Missing frames!"

    end = len(left_paths) if args.end is None else args.end

    w = min(cpu_count(), args.workers)
    print(f'*** number of workers {w}.')
    st = time()
    if args.joblib or args.joblib_thread:
        print('using joblib ...')
        prefer = 'threads' if args.joblib_thread else None
        Parallel(w, prefer=prefer)(delayed(write_rgb_image)(left_paths[i], right_paths[i], args.camtype, rect, False) 
                                                for i in tqdm(range(args.start, end, args.step)))
    else:
        print('using multiprocessing')
        processes = []
        inds = list(range(args.start, end, args.step))
        qsize = 600
        output_queue = Queue(qsize)
        output_writer_queue = Queue(qsize)
        writers = Writers(w, output_writer_queue)
        for i in range(w):
            left_sub_list = [left_paths[inds[j]] for j in range(i, len(inds), w)]
            right_sub_list = [right_paths[inds[j]] for j in range(i, len(inds), w)]
            p = Process(target=process_loop, args=(i, output_queue, left_sub_list, right_sub_list, args.camtype, rect, False))
            processes.append(p)
            p.start()
            print(f'[STARTED] process {i}.')

        for i in tqdm(range(len(inds))):
            item = output_queue.get()

            if item is not False:
                output_writer_queue.put(item)
                # print(output_queue.qsize(), output_writer_queue.qsize())
            # if item is not False:
            #     cv2.imwrite(item['left']['path'],item['left']['image'])
            #     cv2.imwrite(item['right']['path'],item['right']['image'])
        print('last', output_queue.qsize(), output_writer_queue.qsize())
        for i in range(w):
            output_writer_queue.put(None)
        writers.join()
        for i, p in enumerate(processes):
            p.join()
            # print(f'[JOINED] process {i}')
    print(time()-st)
