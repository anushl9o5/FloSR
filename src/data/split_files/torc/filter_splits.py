import csv
import os

def filter_csv(input_file, output_file, matching_strings):
    with open(input_file, 'r', newline='') as csvfile:
        reader = csv.reader(csvfile)
        rows = [row for row in reader if any(matching_string in ','.join(row) for matching_string in matching_strings)]
        #rows = [row for row in reader if matching_string in ','.join(row)]

    with open(output_file, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(rows)


if __name__ == "__main__":
    
    matches = [['noon'], ['steady'], ['morning'], ['dusk', 'evening']]
    
    for matching_string in matches:
        for input_file in ['val_dataset.csv', 'train_dataset.csv']:

            if not os.path.isdir(matching_string[0]):
                os.makedirs(matching_string[0])
            output_file = "{}/{}".format(matching_string[0], input_file)

            filter_csv(input_file, output_file, matching_string)