# this script does this
# 1. read in the "encoder_output" folder
# 2. for each folder (for example: X) in "encoder_output", copy the params.json file to the "emotion_train_data" folder with the name X.json

import os
from tqdm import tqdm

# 1. read in the "encoder_output" folder
encoder_output_folder = "encoder_output"
encoder_output_folders = os.listdir(encoder_output_folder)

# 2. for each folder (for example: X) in "encoder_output", copy the params.json file to the "emotion_train_data" folder with the name X.json
for folder in tqdm(encoder_output_folders):
    params_file = os.path.join(encoder_output_folder, folder, "params.json")
    emotion_train_data_folder = "emotion_train_data"
    emotion_train_data_file = os.path.join(emotion_train_data_folder, f"{folder}.json")
    print(params_file)
    os.system(f"copy {params_file} {emotion_train_data_file}")
