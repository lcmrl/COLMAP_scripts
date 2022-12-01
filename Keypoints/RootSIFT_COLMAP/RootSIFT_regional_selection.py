import os
import numpy as np
from pathlib import Path

x1 = 1400; y1 = 2000 # x1 = 1650; y1 = 1250
x2 = 4570; y2 = 2360 # x2 = 2050; y2 = 3200

img_dir = r"G:\My Drive\O3DM_2022\FinalPaper\NikonTwoStripsGT\TeaCup\Station_01\imgs"
desc_dir = r"G:\My Drive\O3DM_2022\FinalPaper\NikonTwoStripsGT\TeaCup\Station_01\RootSIFT_P0.0026\desc"
out_dir = r"G:\My Drive\O3DM_2022\FinalPaper\NikonTwoStripsGT\TeaCup\Station_01\RootSIFT_P0.0026\by_strips\2\desc"


def RSIFT(img_name, desc_folder):
    np_kpt_path = Path("{}.kpt.npy".format(img_name))
    abs_np_kpt_path = desc_folder / np_kpt_path
    np_dsc_path = Path("{}.dsc.npy".format(img_name))
    abs_np_dsc_path = desc_folder / np_dsc_path
    
    kp = np.load(abs_np_kpt_path)
    desc = np.load(abs_np_dsc_path)
    kp_numb = kp.shape[0]
    
    return kp, desc, kp_numb


### MAIN STARTS HERE
images = os.listdir(img_dir)
kpts_matrix = np.empty((0, 6))
desc_matrix = np.empty((0, 128))
for en,img in enumerate(images):
    print(en)
    kp, desc, kp_numb = RSIFT(img, desc_dir)
    for row in range(kp.shape[0]):
        x = kp[row, 0]
        y = kp[row, 1]
        if x>x1 and y>y1 and x<x2 and y<y2:
            kpts_matrix = np.vstack((kpts_matrix, kp[row, :]))
            desc_matrix = np.vstack((desc_matrix, desc[row, :]))

    out_loc = r"{}/{}.kpt.npy".format(out_dir, img)
    kpts_matrix = kpts_matrix.astype(np.float32)
    np.save(out_loc, kpts_matrix)
    out_loc = r"{}/{}.dsc.npy".format(out_dir, img)
    desc_matrix = desc_matrix.astype(np.uint8)
    np.save(out_loc, desc_matrix)

    kpts_matrix = np.empty((0, 6))
    desc_matrix = np.empty((0, 128))