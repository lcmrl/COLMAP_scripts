# CPU
# docker run -ti --rm --name kornia_container -v /path/to/code_repo:/home kornia_image
# GPU
# docker run --runtime=nvidia -e NVIDIA_VISIBLE_DEVICES=all -ti --name kornia_container -v /path/to/code_repo:/home kornia_image

import os

# Input/Output folders
image_folder = r"C:\Users\Luscias\Desktop\IMAGES_HANDCRAFTED\imgs_SanFrancisco"
desc_folder = r"C:\Users\Luscias\Desktop\IMAGES_HANDCRAFTED\ORB\SanFrancisco\desc"
colmap_desc_folder = r"C:\Users\Luscias\Desktop\IMAGES_HANDCRAFTED\ORB\SanFrancisco\kpts_and_matches_RT_0.90\colmap_desc"
matches_folder = r"C:\Users\Luscias\Desktop\IMAGES_HANDCRAFTED\ORB\SanFrancisco\kpts_and_matches_RT_0.90\matches"
tile_folder = r"C:\Users\User\Desktop\Luca\GitProjects\Github_3DOM\COLMAP_scripts\Keypoints\KeyNetAffNetHardNet\tiles"
custom_pairs = r"C:\Users\Luscias\Desktop\3DOM\ISPRS2022_EUROSDR\SceneViews.json"

# Matching parameters
torch_acceleration = "my_matcher" # "torch" or "my_matcher"
match_mode = 'smnn' # works only for torch_acceleration
multhreading = False # if False, pool_N must be 1
tiling = False
max_kpts_per_tile = 40000
matching_approach = "exhaustive" # "exhaustive" or "custom" or "sequential_2img_overlap"
local_feature = "photomatch" # "ASLFeat", "R2D2", "LFNet", "KeyNet", "photomatch", "RootSIFT", "RoRD" or "D2Net". RootSIFT and ALIKE are equivalent to KeyNet
cross_check = False # Works only with the 'without_Lowe_ratio_test' option
check = 'Lowe_ratio_test'    # 'without_Lowe_ratio_test' or 'Lowe_ratio_test'
matching_distance = 'NORM_HAMMING' # 'L2' or 'NORM_HAMMING'
matching_strategy = 'intersection' # Works only with the 'Lowe_ratio_test' option. 'union', 'unidirectional', 'intersection'
ratio_thresh = 0.90
pool_N = 1 # nuber of logical processors to be used
debug = False
extention_list = ['jpg', 'JPG', 'png']

# Print settings
imgs_list = []
for item in os.listdir(image_folder):
    if item[-3:] in extention_list:
        imgs_list.append(item)
imgs_list = sorted(imgs_list)
if imgs_list == None:
    print("Empty folders or image extantion not in extention_list"), print("Exit")
    quit()

print("torch_acceleration:\t\t", torch_acceleration)
print("number of images:\t\t", len(imgs_list))
print("matcher.py configuration:")
print("\nINPUT/OUTPUT DIR:")
print("image_folder\t\t\t", image_folder)
print("desc_folder\t\t\t", desc_folder)
print("colmap_desc_folder\t\t", colmap_desc_folder)
print("matches_folder\t\t\t", matches_folder)
print("tile_folder\t\t\t", tile_folder)
print("custom_pairs\t\t\t", custom_pairs)

print("\nOptions:")
print("multhreading\t\t\t", multhreading)
print("tilining\t\t\t", tiling)
print("max_kpts_per_tile\t\t", max_kpts_per_tile)
print("matching_approach\t\t", matching_approach)
print("local_feature\t\t\t", local_feature)
print("cross_check\t\t\t", cross_check)
print("check\t\t\t\t", check)
print("matching_distance\t\t", matching_distance)
print("matching_strategy\t\t", matching_strategy)
print("ratio_thresh\t\t\t", ratio_thresh)
print("pool_N\t\t\t\t", pool_N)
print("debug\t\t\t\t", debug)

#userI = input("\nDo you want to continue? y/n\n")
#if userI != "y":
#    quit()

