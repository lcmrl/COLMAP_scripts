
METHODS = ["ASLFeat", "ALIKE", "LFNet", "R2D2", "RoRD"]
PAIRS = ["FlatironBuildingNewYork", "Moscow", "Osnabrueck", "Prague", "SanFrancisco"]
WORKING_DIR = r"C:\Users\Luscias\Desktop\outs_entire_images"
THRESHOLDS = ["0.70", "0.80", "0.90", "1.00"]
OUT_DIR = r"C:\Users\Luscias\Desktop\outs_entire_images\00_matches_RAW"

for method in METHODS:
    for pair in PAIRS:
        for ratio_threshold in THRESHOLDS:
            file_name = pair
            DIR = r"{}\{}\{}\kpts_and_matches_RT_{}".format(WORKING_DIR, method, pair, ratio_threshold)
            print(DIR)

            kpt1_dict ={}
            kpt2_dict ={}

            with open("{}/colmap_desc/{}1.jpg.txt".format(DIR, file_name), "r") as file1, open("{}/colmap_desc/{}2.jpg.txt".format(DIR, file_name), "r") as file2, open("{}/matches/matches.txt".format(DIR), "r") as matches_file, open("{}/{}1.jpg_{}2.jpg_{}_RT{}_RAW.txt".format(OUT_DIR, pair, pair, method, ratio_threshold),"w") as out_file:
                lines1 = file1.readlines()
                lines2 = file2.readlines()
                matches_lines = matches_file.readlines()

                for c1, line1 in enumerate(lines1[1:]):
                    x1, y1, _ = line1.split(" ", 2)
                    kpt1_dict[c1] = [x1, y1]

                for c2, line2 in enumerate(lines2[1:]):
                    x2, y2, _ = line2.split(" ", 2)
                    kpt2_dict[c2] = [x2, y2]
                
                #print(kpt1_dict)
                print("c1 c2 {} {}".format(c1, c2))

                for match_line in matches_lines[1:-2]:
                    m1, m2 = match_line.split(" ", 1)
                    m1 = int(m1)
                    m2 = int(m2)

                    #print("{} {} {} {}\n".format(kpt1_dict[m1][0], kpt1_dict[m1][1], kpt2_dict[m2][0], kpt2_dict[m2][1]))

                    out_file.write("{} {} {} {}\n".format(kpt1_dict[m1][0], kpt1_dict[m1][1], kpt2_dict[m2][0], kpt2_dict[m2][1]))

                    #quit()

