import os
import glob
import random

# set the directory path containing the files
dir_path = "[Your dataset path]/Million-AID/test"
output_file_path = "[Your dataset path]/Million-AID/random_labels_all.txt"

images = glob.glob(os.path.join(dir_path, "*"))

# open a new text file to write the output to
with open(output_file_path, "w") as f:
    for img in images:
        img_name = os.path.basename(img)
        label = random.randint(0, 51)
        f.write(f"{img_name} {label}\n")


