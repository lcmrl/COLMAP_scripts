import os

input_folder = r"G:\Shared drives\3DOM Research\PhD Luca\workflow\publications\2022\LowCost3D\2_FBK_GOPRO\VIDEO\INPUT_VIDEO\prova"


# Main starts here
file_list = os.listdir(input_folder)
print(file_list)

for file in file_list:
    os.rename(r"{}/{}".format(input_folder, file), r"{}/{}.jpg".format(input_folder, int(file[8:-4])))
    #os.rename("{}/{}".format(input_folder, file), "{}/{}".format(input_folder, file[:-4]))