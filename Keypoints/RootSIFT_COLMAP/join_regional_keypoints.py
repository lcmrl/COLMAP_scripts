import os

working_dir = r"G:\My Drive\O3DM_2022\FinalPaper\NikonTwoStripsGT\TeaCup\Station_01\RootSIFT_P0.0026\by_strips"
img_dir = r"G:\My Drive\O3DM_2022\FinalPaper\NikonTwoStripsGT\TeaCup\Station_01\imgs"
sub_folder_list = [k for k in range(0, 3)]


def read_matches_txt(path_to_matches):
    matches_dict = {}
    with open(path_to_matches, 'r') as matches_file:
        lines = matches_file.readlines()
        matches_list = []
        img1 = None
        img2 = None
        for line in lines:
            if line != "\n":
                line = line.strip()
                element1, element2 = line.split(" ", 1)
                try:
                    match1 = int(element1)
                    match2 = int(element2)
                    matches_list.append([match1, match2])
                    matches_dict[(img1, img2)] = matches_list
                except:
                    img1 = element1
                    img2 = element2
                    matches_list = []
            elif line == "\n":
                print("Found empty line, it is not an error.")
    return matches_dict

def read_kpts_txt(path_to_keypoints):
    kpts_list = []
    with open(path_to_keypoints, 'r') as kp_file:
        kp_lines = kp_file.readlines()
        for c, line in enumerate(kp_lines[1:]):
            if line != r"\n":
                x, y, _ = line.split(" ", 2)
                kpts_list.append([c, x, y])
    return kpts_list

### MAIN ###
kpts_dict = {}
matches_dict = {}
images = os.listdir(img_dir)

for folder in sub_folder_list:
    matches_path = r"{}/{}/matches/matches.txt".format(working_dir, folder)
    current_matches_dict = read_matches_txt(matches_path)
    current_kpts_dict = {}
    for img in images:
        kpt_path = r"{}/{}/colmap_desc/{}.txt".format(working_dir, folder, img)
        current_kpts_dict[img] = read_kpts_txt(kpt_path)
        if img in kpts_dict.keys():
            numb_existing_keypoints = len(kpts_dict[img])
            #print("len(kpts_dict[img])", len(kpts_dict[img])), quit()
            for k in current_kpts_dict[img]:
                k.append(k[0] + numb_existing_keypoints)
            kpts_dict[img] = kpts_dict[img] + current_kpts_dict[img]
        else:
            for k in current_kpts_dict[img]:
                k.append(k[0])
            kpts_dict[img] = current_kpts_dict[img]

    for key in current_matches_dict:
        img1 = key[0]
        img2 = key[1]
        for match in current_matches_dict[key]:
            match[0] = current_kpts_dict[img1][match[0]][3]
            match[1] = current_kpts_dict[img2][match[1]][3]
        if key in matches_dict.keys():
            matches_dict[key] = matches_dict[key] + current_matches_dict[key]
        else:
            matches_dict[key] = current_matches_dict[key]

matches_out_path = r"{}/matches.txt".format(working_dir)
with open(matches_out_path, 'w') as out_match_file:
    for key in matches_dict:
        out_match_file.write("{} {}\n".format(key[0], key[1]))
        for match in matches_dict[key]:
            out_match_file.write("{} {}\n".format(match[0], match[1]))
        out_match_file.write("\n")

for img in kpts_dict:
    keypoint_path = r"{}/{}.txt".format(working_dir, img)
    with open(keypoint_path, 'w') as new_kpt_file:
        new_kpt_file.write("{} 128\n".format(len(kpts_dict[img])))
        for kp in kpts_dict[img]:
            new_kpt_file.write("{} {} 0.000000 0.000000\n".format(kp[1], kp[2]))


#for folder in sub_folder_list:
#    matches_path = r"{}/{}/matches/matches.txt".format(working_dir, folder)
#    current_matches_dict = read_matches_txt(matches_path)
#    for key in current_matches_dict:
#        img1 = key[0]
#        img2 = key[1]
#        kpt1_path = r"{}/{}/colmap_desc/{}.txt".format(working_dir, folder, img1)
#        current_kpts1_list = read_kpts_txt(kpt1_path)
#        kpt2_path = r"{}/{}/colmap_desc/{}.txt".format(working_dir, folder, img2)
#        current_kpts2_list = read_kpts_txt(kpt2_path)
#        if (img1, img2) in matches_dict.keys():
#            matches_list = []
#            for match in current_matches_dict[key]:
#                #print(match)
#                x1 = current_kpts1_list[match[0]][1]
#                y1 = current_kpts1_list[match[0]][2]
#                x2 = current_kpts2_list[match[1]][1]
#                y2 = current_kpts2_list[match[1]][2]
#                #print("type(matches_dict[(img1, img2)])", type(matches_dict[(img1, img2)]))
#                matches_list.append((x1, y1, x2, y2))
#            matches_dict[(img1, img2)] = matches_dict[(img1, img2)] + matches_list
#        else:
#            matches_list = []
#            for match in current_matches_dict[key]:
#                #print(match)
#                x1 = current_kpts1_list[match[0]][1]
#                y1 = current_kpts1_list[match[0]][2]
#                x2 = current_kpts2_list[match[1]][1]
#                y2 = current_kpts2_list[match[1]][2]
#                matches_list.append((x1, y1, x2, y2))
#            matches_dict[(img1, img2)] = matches_list
##print(matches_dict[(img1, img2)])
#
#matches_out_path = r"{}/matches.txt".format(working_dir)
#with open(matches_out_path, 'r') as out_match_file:
#    for key in matches_dict:
#        out_match_file.write("{} {}\n".format(key[0], key[1]))
#        for match in






















        #if img in kpts_dict.keys():
        #    number_Existing_keypoints = len(kpts_dict[img])
        #    for k in kpts_list:
        #        k[0] = k[0] + number_Existing_keypoints
        #    kpts_dict[img] = kpts_dict[img] + kpts_list
        #else:
        #    kpts_dict[img] = kpts_list
        #quit()
            
        