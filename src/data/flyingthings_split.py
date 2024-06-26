import os
import csv
import glob
import pandas as pd

from tqdm import tqdm

def read_calib(calib_path):
    lines = []
    with open(calib_path, 'r') as f:
        for line in f.readlines():
            if line.strip():
                lines.append(line.rstrip())
    
    calib_dict = {"{:04d}.png".format(int(key.split(' ')[-1])): {val1.split(' ')[0]: val1.split(' ')[1:], val2.split(' ')[0]: val2.split(' ')[1:]} for key, val1, val2 in zip(lines[::3], lines[1::3], lines[2::3])}

    return calib_dict

def parse_data(data_dirs, sample_freq=10):
    data_dict = {}

    for data_dir in tqdm(data_dirs):
        data_split = data_dir.split('/')
        calib_file = os.path.join(data_dir.replace('cleanpass/frames_cleanpass', 'camera_data'), 'camera_data.txt')
        calib = read_calib(calib_file)
        img0s = sorted(glob.glob(os.path.join(data_dir, "left/*.png")))
        img1s = sorted(glob.glob(os.path.join(data_dir, "right/*.png")))
        recording_id = os.path.join(data_split[-3], data_split[-2])
        for img0, img1 in zip(img0s[::sample_freq], img1s[::sample_freq]):
            if not os.path.basename(img0) in list(calib.keys()):
                continue
            data_dict["img0"] = data_dict.get("img0", []) + [img0]
            data_dict["img1"] = data_dict.get("img1", []) + [img1]
            data_dict["calib"] = data_dict.get("calib", []) + [calib_file]
            data_dict["id"] = data_dict.get("id", []) + [recording_id]

    return data_dict

def save_splits(data_dict, mode='train'):
    df = pd.DataFrame.from_dict(data_dict)
    print(mode, df['id'].value_counts())
    df.to_csv("{}_dataset.csv".format(mode), index=False, header=False)


if __name__ == "__main__":
    train_dirs = glob.glob("/home/eos/workspace/nas.anush/flyingthings/cleanpass/frames_cleanpass/TRAIN/*/*/") # Path to the training data
    val_dirs = glob.glob("/home/eos/workspace/nas.anush/flyingthings/cleanpass/frames_cleanpass/TEST/*/*/") # Path to the validation data

    train_dict = parse_data(train_dirs, sample_freq=1)
    val_dict = parse_data(val_dirs, sample_freq=1)

    save_splits(train_dict, mode='train')
    save_splits(val_dict, mode='val')


    