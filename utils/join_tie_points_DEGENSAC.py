
METHODS = ["ASLFeat", "ALIKE", "LFNet", "R2D2", "RoRD"]
PAIRS = ["FlatironBuildingNewYork", "Moscow", "Osnabrueck", "Prague", "SanFrancisco"]
WORKING_DIR = r"C:\Users\Luscias\Desktop\outs_entire_images"
THRESHOLDS = ["0.70", "0.80", "0.90", "1.00"]
OUT_DIR = r"C:\Users\Luscias\Desktop\outs_entire_images\00_matches_DEGENSAC"

for method in METHODS:
    for pair in PAIRS:
        for ratio_threshold in THRESHOLDS:
            try:
                file_name = pair
                DIR = r"{}\{}\{}\kpts_and_matches_RT_{}\degensac".format(WORKING_DIR, method, pair, ratio_threshold)
                print(DIR)

                with open("{}/{}1.jpg.txt".format(DIR, file_name), "r") as file1, open("{}/{}2.jpg.txt".format(DIR, file_name), "r") as file2, open("{}/{}1.jpg_{}2.jpg_{}_RT{}_DEGENSAC.txt".format(OUT_DIR, pair, pair, method, ratio_threshold),"w") as out_file:
                    lines1 = file1.readlines()
                    lines2 = file2.readlines()

                    for line1, line2 in zip(lines1[1:-1], lines2[1:-1]):
                        x1, y1, _ = line1.split(" ", 2)
                        x2, y2, _ = line2.split(" ", 2)
                        out_file.write("{} {} {} {}\n".format(x1, y1, x2, y2))
                    print("Done!")
            except:
                print("File with no matches? Check...")