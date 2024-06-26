import os
import csv
import glob
import random
import pandas as pd

from tqdm import tqdm

def parse_data(data_dir, cams, img_ids, sample_freq=1):
    data_dict = {}

    for img_id in tqdm(img_ids):
        cam = random.sample(cams, 2)
        calib_file = os.path.join(data_dir, "cam_poses/{}.json".format(img_id))
        img0 = os.path.join(data_dir, "cam_p_{}/{}.png".format(cam[0], img_id))
        img1 = os.path.join(data_dir, "cam_p_{}/{}.png".format(cam[1], img_id))

        data_dict['ref'] = data_dict.get("ref", []) + [cam[0]]
        data_dict['tar'] = data_dict.get("tar", []) + [cam[1]]
        data_dict["img0"] = data_dict.get("img0", []) + [img0]
        data_dict["img1"] = data_dict.get("img1", []) + [img1]
        data_dict["calib"] = data_dict.get("calib", []) + [calib_file]

    return data_dict

def save_splits(data_dicts, mode='train'):
    df = [pd.DataFrame.from_dict(data_dict) for data_dict in data_dicts]
    df_merged = pd.concat(df).sample(frac=1)
    print(mode, df_merged['calib'].value_counts())
    df_merged.to_csv("{}_dataset.csv".format(mode), index=False, header=False)

"Generate training and testing splits on the carla pert_2 and pert_3 datasets combined"

if __name__ == "__main__":
    data_dirs = ["/nas/EOS-DB/users/siva/data/carla_data/simulations/rand_weather_camera_pert_2/night_Town06/", "/nas/EOS-DB/users/siva/data/carla_data/simulations/rand_weather_camera_pert_3/night_Town06/"]
    train_dicts, val_dicts = [], []
    for data_dir in data_dirs:
        cams = [0, 1, 2]
        img_ids = glob.glob(os.path.join(data_dir, "cam_poses", "*.json"))
        ids = [os.path.basename(img_id).split('.')[0] for img_id in img_ids]

        random.shuffle(ids)

        train_ids = ids[:int(0.9*len(ids))]
        val_ids = ids[len(train_ids):]

        train_dicts.append(parse_data(data_dir, cams, train_ids, sample_freq=1))
        val_dicts.append(parse_data(data_dir, cams, val_ids, sample_freq=1))

    save_splits(train_dicts, mode='train')
    save_splits(val_dicts, mode='val')



    