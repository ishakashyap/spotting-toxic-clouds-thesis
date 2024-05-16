import os
import shutil
import json

def load_filenames(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)
        filenames = []
        for i in data:
            filenames.append(i["file_name"] + '.mp4')
    return filenames

def move_files(file_list, source_folder, destination_folder):
    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)
    
    for filename in file_list:
        source_path = os.path.join(source_folder, filename)
        destination_path = os.path.join(destination_folder, filename)
        
        if os.path.exists(source_path):
            shutil.move(source_path, destination_path)
        else:
            print(f"File {source_path} not found!")

def main():
    train_json = "./split/metadata_train_split_by_date.json"
    val_json = "./split/metadata_validation_split_by_date.json"
    test_json = "./split/metadata_test_split_by_date.json"

    source_folder = "./videos"
    train_destination = "./train_baseline"
    val_destination = "./validation_baseline"
    test_destination = "./test_baseline"

    # Load filenames from json files
    train_files = load_filenames(train_json)
    val_files = load_filenames(val_json)
    test_files = load_filenames(test_json)


    # Move files to their respective folders
    move_files(train_files, source_folder, train_destination)
    move_files(val_files, source_folder, val_destination)
    move_files(test_files, source_folder, test_destination)

    print("Files have been moved successfully.")

if __name__ == "__main__":
    main()
