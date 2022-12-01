### KEYPOINTS CONVERTER LF-NET -> OPENCV FORMAT
#
# >python 3domLFNet2openCV.py -i C:/Users/Luscias/Desktop/3DOM/07_LFNet/provaGCP/desc_LFNet/LEO_8588_acr.jpg.npz -g C:/Users/Luscias/Desktop/3DOM/07_LFNet/provaGCP/imgs/LEO_8588_acr.jpg

# Libraries 
import cv2
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
import os

# Convert LF-Net descriptors in openCV format
def LFNet2openCV(desc_path):

    np_desc_path = "{}.npz".format(desc_path)
    # Import LFNet keypoints and descriptors
    with np.load(np_desc_path) as data:
        d = dict(zip(("keypoints","descriptors","resolution","scale","orientation"), (data[k] for k in data)))
    kp = d['keypoints']
    kp_numb = d['keypoints'].shape[0]
    opencv_descriptors = d['descriptors']
    
    # Convert keypoints in openCV format
    opencv_keypoints = []
    for i in range (0,kp.shape[0]):
        opencv_keypoints.append(cv2.KeyPoint(kp[i][0],kp[i][1],0,0))
        
    return opencv_keypoints, opencv_descriptors, kp_numb


# Main function
def main():

    # I/O management
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--Input", help = "Input LFNet descriptor path without .npz extension")
    parser.add_argument("-g", "--Image", help = "Input image path")
    args = parser.parse_args()

    if args.Image:
        print("Diplaying image path as: % s" % args.Image)
    if args.Input:
        print("Diplaying image descriptors path as: % s" % args.Input)
    
    desc_path = args.Input
    image_path = args.Image
    
    # Show image with features
    img = cv2.imread(image_path)
    plt.figure(figsize=(10, 10))
    plt.title('Image 1')
    plt.imshow(img)
    plt.show()
    
    opencv_keypoints, opencv_descriptors, kp_numb = LFNet2openCV(desc_path)
    
    image = cv2.drawKeypoints(img,opencv_keypoints,img,color=[255,0,0],flags=0) # For Python API, flags are modified as cv2.DRAW_MATCHES_FLAGS_DEFAULT, cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS, cv2.DRAW_MATCHES_FLAGS_DRAW_OVER_OUTIMG, cv2.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS
    plt.imshow(image)
    plt.show()
    
    print('Descriptors:  {}'.format(opencv_descriptors.shape))
    print(opencv_descriptors)


# driver function 
if __name__=="__main__": 
    main()