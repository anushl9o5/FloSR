import random
import os
import csv
import glob

import numpy as np
import pandas as pd

def parse_root():
    left_dirs = glob.glob("/home/eos/workspace/nas.anush/SintelStereo/training/clean_left/*")
    right_dir = "/home/eos/workspace/nas.anush/SintelStereo/training/clean_right/"
    calib_dir = "/home/eos/workspace/nas.anush/SintelStereo/training/camdata_left/"
    data_dict = {}
    for data_dir in left_dirs:
        data_split = data_dir.split('/')
        recording_id = data_split[-1]
        calib_folder = os.path.join(calib_dir, recording_id)
        img0s = sorted(glob.glob(os.path.join(data_dir, "*.png")))
        img1s = sorted(glob.glob(os.path.join(right_dir, recording_id, "*.png")))
        
        for img0, img1 in zip(img0s, img1s):
            frame_id = os.path.basename(img0).split('.')[0]
            data_dict["img0"] = data_dict.get("img0", []) + [img0]
            data_dict["img1"] = data_dict.get("img1", []) + [img1]
            data_dict["calib"] = data_dict.get("calib", []) + [os.path.join(calib_folder, frame_id+".cam")]
            data_dict["id"] = data_dict.get("id", []) + [recording_id]
            data_dict["is_flipped"] = data_dict.get("is_flipped", []) + [False]

            data_dict["img0"] = data_dict.get("img0", []) + [img0.replace("clean_left", "final_left")]
            data_dict["img1"] = data_dict.get("img1", []) + [img1.replace("clean_right", "final_right")]
            data_dict["calib"] = data_dict.get("calib", []) + [os.path.join(calib_folder, frame_id+".cam")]
            data_dict["id"] = data_dict.get("id", []) + [recording_id]
            data_dict["is_flipped"] = data_dict.get("is_flipped", []) + [False]

            if np.random.rand() > 0.5:
                data_dict["img0"] = data_dict.get("img0", []) + [img1]
                data_dict["img1"] = data_dict.get("img1", []) + [img0]
                data_dict["calib"] = data_dict.get("calib", []) + [os.path.join(calib_folder, frame_id+".cam")]
                data_dict["id"] = data_dict.get("id", []) + [recording_id]
                data_dict["is_flipped"] = data_dict.get("is_flipped", []) + [True]

            if np.random.rand() > 0.5:  
                data_dict["img0"] = data_dict.get("img0", []) + [img1.replace("clean_right", "final_right")]
                data_dict["img1"] = data_dict.get("img1", []) + [img0.replace("clean_left", "final_left")]
                data_dict["calib"] = data_dict.get("calib", []) + [os.path.join(calib_folder, frame_id+".cam")]
                data_dict["id"] = data_dict.get("id", []) + [recording_id]
                data_dict["is_flipped"] = data_dict.get("is_flipped", []) + [True]

    split(data_dict)

def split(data_dict):
    combined = pd.DataFrame.from_dict(data_dict)

    all_recs = list(set(combined["id"]))
    random.shuffle(all_recs)
    counts = combined["id"].value_counts()
    val_len = int(0.2*len(combined))
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
        cur_rec = row["id"]

        if cur_rec in train_recs:
            train_list.append(list(row))
        if cur_rec in val_recs:
            val_list.append(list(row))

    train_df = pd.DataFrame(train_list)
    val_df = pd.DataFrame(val_list)

    print(train_df[3].value_counts())
    print(val_df[3].value_counts())
    
    train_df.to_csv("train_dataset.csv", index=False, header=False)
    val_df.to_csv("val_dataset.csv", index=False, header=False)

if __name__ == "__main__":
    parse_root()
