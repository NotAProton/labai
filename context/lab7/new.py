import os
import re

files = sorted(
    [f for f in os.listdir('.') if re.match(r'image(-\d+)?\.png$', f)],
    key=lambda x: int(re.search(r'\d+', x).group()) if '-' in x else 0,
    reverse=True
)

# Rename files
for f in files:
    if f == "image.png":
        new = "image-1.png"
    else:
        n = int(re.search(r'\d+', f).group())
        new = f"image-{n+1}.png"
    os.rename(f, new)

