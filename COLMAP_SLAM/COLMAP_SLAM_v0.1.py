import subprocess
import time
import shutil
import os


SLEEP_TIME = 2
LOOP_CYCLES = 1000
COLMAP_EXE_PATH = r"C:\Users\Luscias\Desktop\3DOM\COLMAP\COLMAP_3_6_windows"
DRIVE_FOLDER = "H:/My Drive/foto_honor6x"
LOCAL_FOLDER = "C:/Users/Luscias/Desktop/SLAM/imgs"


processed_imgs = []

for i in range (LOOP_CYCLES):

    drive_imgs = os.listdir(DRIVE_FOLDER)
    print(drive_imgs)
    for img in drive_imgs:
        if img not in processed_imgs:
            shutil.copyfile('{}/{}'.format(DRIVE_FOLDER, img), '{}/{}'.format(LOCAL_FOLDER, img))
            processed_imgs.append(img)

    subprocess.run([r"{}/COLMAP.bat".format(COLMAP_EXE_PATH), "database_creator", "--database_path", r"C:\Users\Luscias\Desktop\SLAM\db.db"])
    subprocess.run([r"{}/COLMAP.bat".format(COLMAP_EXE_PATH), "feature_extractor", "--database_path", r"C:\Users\Luscias\Desktop\SLAM\db.db", "--image_path", r"C:\Users\Luscias\Desktop\SLAM\imgs"])
    subprocess.run([r"{}/COLMAP.bat".format(COLMAP_EXE_PATH), "exhaustive_matcher", "--database_path", r"C:\Users\Luscias\Desktop\SLAM\db.db"])
    subprocess.run([r"{}/COLMAP.bat".format(COLMAP_EXE_PATH), "mapper", "--database_path", r"C:\Users\Luscias\Desktop\SLAM\db.db", "--image_path", r"C:\Users\Luscias\Desktop\SLAM\imgs", "--output_path", r"C:\Users\Luscias\Desktop\SLAM\outs"])

    time.sleep(SLEEP_TIME)