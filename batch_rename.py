#!/usr/bin/env python
# Script to batch rename files via regular expression

import os
import re

# define path where files are
path = "/projekt2/remsens/data/campaigns/eurec4a/RV-METEOR_DSHIP/upload_to_aeris/"
os.chdir(path)
print(os.getcwd())
files = os.listdir(path)
check1 = input(f"Do you want to rename all those files:\n {files}")
if check1.startswith("y"):
    print("Renaming all files...")
    for file in files:
        match = re.search(r"(?P<date>\d{8})", file)
        newname = f"RV-METEOR_DSHIP_1Hz_{match.group('date')}.dat"
        os.rename(file, newname)
        print(f"renamed {file} to {newname}")