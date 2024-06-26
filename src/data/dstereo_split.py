import os
import csv
import glob
import pandas as pd

from tqdm import tqdm

def parse_data(data_dirs, sample_freq=10):
    data_dict = {}

    for data_dir in data_dirs:
        mode = data_dir.split("/")[-2]
        calib_file = os.path.join(data_dir, "{}-calib/full-image-calib/{}.txt")
        img0s = sorted(glob.glob(os.path.join(data_dir, "{}-left-image/*.jpg".format(mode))))
        img1s = sorted(glob.glob(os.path.join(data_dir, "{}-right-image/*.jpg".format(mode))))
        for img0, img1 in tqdm(zip(img0s[::sample_freq], img1s[::sample_freq])):
            recording_id = os.path.basename(img0).split("_")[0]
            data_dict["img0"] = data_dict.get("img0", []) + [img0]
            data_dict["img1"] = data_dict.get("img1", []) + [img1]
            data_dict["calib"] = data_dict.get("calib", []) + [calib_file.format(mode, recording_id)]
            data_dict["id"] = data_dict.get("id", []) + [recording_id]

    return data_dict

def save_splits(data_dict, mode='train'):
    df = pd.DataFrame.from_dict(data_dict)
    print(mode, df['id'].value_counts())
    df.to_csv("{}_dataset.csv".format(mode), index=False, header=False)

if __name__ == "__main__":
    train_dirs = glob.glob("/nas/EOS/users/anush/drivingstereo/train/") # Path to the training data
    val_dirs = glob.glob("/nas/EOS/users/anush/drivingstereo/test/") # Path to the validation data

    train_dict = parse_data(train_dirs, sample_freq=10)
    val_dict = parse_data(val_dirs, sample_freq=2)

    save_splits(train_dict, mode='train')
    save_splits(val_dict, mode='val')


    