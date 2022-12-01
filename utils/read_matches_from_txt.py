
def read_matches_txt(path_to_matches, matches_dict):
    with open(path_to_matches, 'r') as matches_file:
        lines = matches_file.readlines()
        matches_list = []
        img1 = None
        img2 = None
        for line in lines:
            try:
                line = line.strip()
                element1, element2 = line.split(" ", 1)
                try:
                    match1 = int(element1)
                    match2 = int(element2)
                    matches_list.append((match1, match2))
                    matches_dict[(img1, img2)] = matches_list
                except:
                    img1 = element1
                    img2 = element2
            except:
                print("Found empty line, it is not an error.")

        #print(matches_dict)
        #print(matches_dict.keys())


matches_dict = {}
path_to_matches = r"G:\My Drive\O3DM_2022\FinalPaper\pairs\nikon\PlasticBottle\resized_all_images\PlasticBottle\matches\matches.txt"
read_matches_txt(path_to_matches, matches_dict)
