import os
from pathlib import Path

IMGS_DIR = Path(r"/media/luca/TOSHIBA EXT/FixPositioning/2022-04-12-Seq5/KeyframesFixPos/imgs")
OUT_FILE = Path(r"/media/luca/TOSHIBA EXT/FixPositioning/2022-04-12-Seq5/KeyframesFixPos/pairs.txt")
OVERLAP = 10

# MAIN
with open(OUT_FILE, 'w') as out_file:
    imgs = os.listdir(IMGS_DIR)
    imgs.sort()
    for i in range(len(imgs)-OVERLAP):
        for k in range(OVERLAP):
            j = i + k + 1
            im1 = imgs[i]
            im2 = imgs[j]
            out_file.write(f"{im1} {im2} 0 0\n")
