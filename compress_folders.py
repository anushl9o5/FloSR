import os
import glob
import tarfile
import argparse

from tqdm import tqdm


def compress_folders_to_tar(source_dir, destination_dir):
    folders = glob.glob(os.path.join(source_dir, 'torc', '*'))
                        
    for folder in tqdm(folders):
        tar_filename = os.path.basename(folder) + '.tar.gz'
        tar_path = os.path.join(destination_dir, tar_filename)

        with tarfile.open(tar_path, 'w:gz') as tar:
            tar.add(folder, arcname=os.path.basename(folder))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Compress folders at the "caps" level to tar.gz files.')
    parser.add_argument('source_dir', help='Root directory containing folders of images')
    parser.add_argument('destination_dir', help='Directory where tar.gz files will be created')
    args = parser.parse_args()

    compress_folders_to_tar(args.source_dir, args.destination_dir)