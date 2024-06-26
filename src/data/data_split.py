import random
import os
import csv

import pandas as pd

def split():
    train_csv = "split_files/algolux/train_dataset.csv"
    val_csv = "split_files/algolux/val_dataset.csv"

    df1 = pd.read_csv(train_csv)
    df2 = pd.read_csv(val_csv)
    column_names = [i for i in range(len(df1.columns))]
    df1.columns = column_names
    df2.columns = column_names
    combined = pd.concat([df1, df2])

    all_recs = list(set(combined[2]))
    random.shuffle(all_recs)
    counts = combined[2].value_counts()
    val_len = int(0.2*len(combined))
    train_len = len(combined) - val_len

    val_recs = []
    train_recs = []
    val_count = 0
    for rec in all_recs:
        if type(rec) != type(str()):
            continue

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
        cur_rec = row[2]
        if cur_rec == "CAR-377":
            continue

        if cur_rec in train_recs:
            train_list.append(list(row))
        if cur_rec in val_recs:
            val_list.append(list(row))

    train_df = pd.DataFrame(train_list)
    val_df = pd.DataFrame(val_list)

    print(train_df[2].value_counts())
    print(val_df[2].value_counts())
    
    train_df.to_csv("train_dataset.csv", index=False, header=False)
    val_df.to_csv("val_dataset.csv", index=False, header=False)

if __name__ == "__main__":
    split()