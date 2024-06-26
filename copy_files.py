import os
import shutil
import argparse

def copy_files_from_list(list_file, source_dir, destination_dir):
    with open(list_file, 'r') as file:
        files = [line.strip() for line in file.readlines()]

    total_files = len(files)
    progress_count = 0

    for file in files:
        source_path = os.path.join(source_dir, file)
        destination_path = os.path.join(destination_dir, file)

        # Ensure source file exists before copying
        if os.path.exists(source_path):
            os.makedirs(os.path.dirname(destination_path), exist_ok=True)
            shutil.copy2(source_path, destination_path)
            progress_count += 1
            print(f'Progress: {progress_count}/{total_files} files copied')
        else:
            print(f"File '{file}' not found in source directory.")

    print(f'Copied {progress_count} out of {total_files} files.')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Copy files listed in a text file to another directory.')
    parser.add_argument('list_file', help='Path to the text file containing file names')
    parser.add_argument('source_dir', help='Source directory where the files are located')
    parser.add_argument('destination_dir', help='Destination directory to copy the files')
    args = parser.parse_args()

    copy_files_from_list(args.list_file, args.source_dir, args.destination_dir)