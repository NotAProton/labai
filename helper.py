import os
import re

folder = "."  # change if needed

files = sorted(os.listdir(folder))

pattern = re.compile(r"forensics1-(\d+)\.png")

for fname in files:
    match = pattern.match(fname)
    if match:
        num = int(match.group(1)) + 1  # shift by +1
        new_name = f"{num:02d}.png"    # zero-padded (01, 02, ...)

        src = os.path.join(folder, fname)
        dst = os.path.join(folder, new_name)

        print(f"{fname} -> {new_name}")
        os.rename(src, dst)
