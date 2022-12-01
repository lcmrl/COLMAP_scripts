import os
import cv2
import numpy as np

img_folder = r"G:\My Drive\O3DM_2022\FinalPaper\pairs\nikon\PlasticBottle\imgs_gradient"
desc_folder = r"G:\My Drive\O3DM_2022\FinalPaper\pairs\nikon\PlasticBottle\kpts_density_gradient"


imgs = os.listdir(img_folder)
for img in imgs:
    print(img)
    with open("{}/{}.txt".format(desc_folder, img), 'r') as kpt_file:
        kpt_matrix = np.loadtxt(kpt_file, skiprows=1, usecols=(0, 1))

        # Convert keypoints in openCV format
        opencv_keypoints = []
        for i in range (0, kpt_matrix.shape[0]):
            opencv_keypoints.append(cv2.KeyPoint(kpt_matrix[i][0], kpt_matrix[i][1], 2))
    
    imgCV = cv2.imread(r"{}/{}".format(img_folder, img))
    imgCV = cv2.cvtColor(imgCV, cv2.COLOR_BGR2GRAY)
    sift = cv2.SIFT_create(8000, 3, 0.04, 10, 1.6)
    keypoints, descriptors = sift.compute(imgCV, opencv_keypoints)
    #keypoints, descriptors = sift.detectAndCompute(imgCV, None)

    # Saving files
    out_loc = r"{}/{}.kpt.npy".format(desc_folder, img)
    keypoints_mat = cv2.KeyPoint_convert(keypoints)
    empty_matrix = np.zeros((keypoints_mat.shape[0], 4))
    keypoints_mat = np.hstack((keypoints_mat, empty_matrix))
    print(keypoints_mat)
    print(out_loc)
    np.save(out_loc, keypoints_mat)

    out_loc = r"{}/{}.dsc.npy".format(desc_folder, img)
    np.save(out_loc, descriptors)
    