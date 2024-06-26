import os
import shutil
import argparse
from tqdm import tqdm 

def copy_folders_from_list(list_file, source_dir, destination_dir):
    with open(list_file, 'r') as file:
        folders = [line.strip() for line in file.readlines()]

    for folder in tqdm(folders):
        source_path = os.path.join(source_dir, folder)
        destination_path = os.path.join(destination_dir, folder)
        shutil.copytree(source_path, destination_path, symlinks=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Copy folders listed in a text file to another directory while preserving the folder structure.')
    parser.add_argument('list_file', help='Path to the text file containing folder names')
    parser.add_argument('source_dir', help='Source directory where the folders are located')
    parser.add_argument('destination_dir', help='Destination directory to copy the folders')
    args = parser.parse_args()

    copy_folders_from_list(args.list_file, args.source_dir, args.destination_dir)