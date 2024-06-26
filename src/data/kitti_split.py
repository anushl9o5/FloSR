import random
import os
import csv
import glob

import pandas as pd
import numpy as np

def parse_kitti(flip_aug=True):
    data_dirs = glob.glob("/home/eos/workspace/nas.anush/kitti/*/*")
    data_dict = {}
    for data_dir in data_dirs:
        data_split = data_dir.split('/')
        calib_file = os.path.join(os.path.dirname(data_dir), "calib_cam_to_cam.txt")
        img0s = sorted(glob.glob(os.path.join(data_dir, "image_02/data/*")))
        img1s = sorted(glob.glob(os.path.join(data_dir, "image_03/data/*")))
        recording_date = data_split[-2]
        recording_id = data_split[-1]
        for img0, img1 in zip(img0s[::5], img1s[::5]):
            data_dict["img0"] = data_dict.get("img0", []) + [img0]
            data_dict["img1"] = data_dict.get("img1", []) + [img1]
            data_dict["calib"] = data_dict.get("calib", []) + [calib_file]
            data_dict["date"] = data_dict.get("date", []) + [recording_date]
            data_dict["id"] = data_dict.get("id", []) + [recording_id]
            data_dict["is_flipped"] = data_dict.get("is_flipped", []) + [False]

            if np.random.rand() > 0.5 and flip_aug:
                data_dict["img0"] = data_dict.get("img0", []) + [img1]
                data_dict["img1"] = data_dict.get("img1", []) + [img0]
                data_dict["calib"] = data_dict.get("calib", []) + [calib_file]
                data_dict["date"] = data_dict.get("date", []) + [recording_date]
                data_dict["id"] = data_dict.get("id", []) + [recording_id]
                data_dict["is_flipped"] = data_dict.get("is_flipped", []) + [True]

    split(data_dict)

def split(data_dict):
    combined = pd.DataFrame.from_dict(data_dict)

    all_recs = list(set(combined["date"]))
    random.shuffle(all_recs)
    counts = combined["date"].value_counts()
    val_len = int(0.09*len(combined))
    train_len = len(combined) - val_len

    val_recs = []
    train_recs = []
    val_count = 0
    for rec in all_recs:
        if random.random() > 0.5:
            train_recs.append(rec)
        else:
            if val_count < val_len:
                val_recs.append(rec)
                val_count += counts[rec]
            else:
                train_recs.append(rec)

    train_list = []
    val_list = []

    for idx, row in combined.iterrows():
        cur_rec = row["date"]

        if cur_rec in train_recs:
            train_list.append(list(row))
        if cur_rec in val_recs:
            val_list.append(list(row))

    train_df = pd.DataFrame(train_list)
    val_df = pd.DataFrame(val_list)

    print(train_df[4].value_counts())
    print(val_df[4].value_counts())
    
    train_df.to_csv("train_dataset.csv", index=False, header=False)
    val_df.to_csv("val_dataset.csv", index=False, header=False)

if __name__ == "__main__":
    parse_kitti()
