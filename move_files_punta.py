"""
author: Johannes Roettenbacher
date: 25.10.2019

Script to move nc files into their corresponding daily folders.
Folder Structure:
    Year (Y2019):
        Month (M08):
            Day (D05): Files
"""

import os
import re
import shutil

test = True

while test:
    try:
        print(f"Current directory: {os.getcwd()}")
        path = input("Please provide the path to the directory with all files to move: ")
        print(f"Input: {path}")
        os.chdir(path)
        test = False
    except FileNotFoundError:
        print("No such directory! Please check input")

files = [f for f in os.listdir(".") if f.endswith(".nc")]
outfile = open("files_moved.txt", mode='a')
for f in files:
    date_str = re.search(r"[0-9]{6}", f).group()
    year = date_str[:2]
    month = date_str[2:4]
    day = date_str[4:6]
    output_dir = f"../Y20{year}/M{month}/D{day}/"
    print(f"Moving files to {output_dir}")
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)
    shutil.move(src=f"./{f}", dst=f"{output_dir}{f}")
    outfile.write(f"{f}\n")
print("Done with moving files.")
outfile.close()
