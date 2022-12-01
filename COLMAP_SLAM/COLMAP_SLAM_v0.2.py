import subprocess
import time
import shutil
import os
import matplotlib.pyplot as plt
import numpy as np 
#from plyfile import PlyData, PlyElement


SLEEP_TIME = 1
LOOP_CYCLES = 1000
COLMAP_EXE_PATH = r"C:\Users\fbk3d\Desktop\COLMAP\COLMAP-3.6-windows-cuda"
DRIVE_FOLDER = r"C:\Users\fbk3d\Desktop\provvisorio\provvisorio3\imgs"
LOCAL_FOLDER = r"C:\Users\fbk3d\Desktop\provvisorio\provvisorio3\imgs2"
OUT_FOLDER = r"C:\Users\fbk3d\Desktop\provvisorio\provvisorio3\outs"
DATABASE = r"C:\Users\fbk3d\Desktop\provvisorio\provvisorio3"


processed_imgs = []

plt.ion()
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')

for i in range (LOOP_CYCLES):

    drive_imgs = os.listdir(DRIVE_FOLDER)
    print(drive_imgs)
    for img in drive_imgs:
        if img not in processed_imgs:
            shutil.copyfile('{}/{}'.format(DRIVE_FOLDER, img), '{}/{}'.format(LOCAL_FOLDER, img))
            processed_imgs.append(img)

    subprocess.run([r"{}/COLMAP.bat".format(COLMAP_EXE_PATH), "database_creator", "--database_path", rf"{DATABASE}\db.db"])
    subprocess.run([r"{}/COLMAP.bat".format(COLMAP_EXE_PATH), "feature_extractor", "--database_path", rf"{DATABASE}\db.db", "--image_path", rf"{LOCAL_FOLDER}"])
    subprocess.run([r"{}/COLMAP.bat".format(COLMAP_EXE_PATH), "exhaustive_matcher", "--database_path", rf"{DATABASE}\db.db"])
    subprocess.run([r"{}/COLMAP.bat".format(COLMAP_EXE_PATH), "mapper", "--database_path", rf"{DATABASE}\db.db", "--image_path", rf"{LOCAL_FOLDER}", "--output_path", rf"{OUT_FOLDER}"])
    subprocess.run([r"{}/COLMAP.bat".format(COLMAP_EXE_PATH), "model_converter", "--input_path", rf"{OUT_FOLDER}\0", "--output_path", rf"{OUT_FOLDER}", "--output_type", "TXT"])


    points_matrix = np.loadtxt(rf"{OUT_FOLDER}/points3D.txt", dtype='float', comments='#', delimiter=' ', usecols=(1,2,3))
    x = points_matrix[:, 0]
    y = points_matrix[:, 1]
    z = points_matrix[:, 2]


    ax.scatter(x, y, z, c='r', marker='.')
    fig.canvas.draw()	
    fig.canvas.flush_events()
    
    


    time.sleep(SLEEP_TIME)