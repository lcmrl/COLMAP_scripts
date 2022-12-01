# USAGE
# conda activate kornia_env
# python GeometricVerification.py find_fundamental ./images ./kpts_folder ./matches.txt ./out_folder --error_threshold 3 --debug False
# python GeometricVerification.py evaluate_matches ./images ./kpts_folder ./matches.txt ./out_folder --error_threshold 3 --debug False --f_matrix ./f_matrix.txt --flux_corresp ./correspondences.txt
#
# python GeometricVerification.py evaluate_matches ".\imgs" ".\kpts" ".\matches.txt" ".\outs" --error_threshold 4 --debug False --f_matrix ".\F_matrix_Luca.txt" --use_flux True --flux_corresp ".\correspondences_for_flux_GT.txt"


from PIL import Image, ImageDraw
from pathlib import Path
import numpy as np
import pydegensac
import argparse
import os
from scipy.interpolate import LinearNDInterpolator
import matplotlib.pyplot as plt

ALPHA_FLUX_MAX = 1.2
ALPHA_FLUX_MIN = 0.8
ITERATIONS = 500000

### MAIN STARTS HERE
if __name__ == "__main__":
    # I/O foders and options
    parser = argparse.ArgumentParser()
    parser.add_argument("command", help="Digit find_fundamental or evaluate_matches")
    parser.add_argument("image_folder", help="Input path to a folder with only two images")
    parser.add_argument("kpts_folder", help="Input keypoint folder using the COLMAP format")
    parser.add_argument("matches", help="Input matches.txt using the COLMAP format")
    parser.add_argument("out_folder", help="Path to the output folder")
    parser.add_argument("-e","--error_threshold", help="Input matches.txt using the COLMAP format", default = 4, type=int)
    parser.add_argument("-d","--debug", help="True or False", default = "False")
    parser.add_argument("-f","--f_matrix", help="Path to the F_matrix.txt file")
    parser.add_argument("-z","--use_flux", help="True or False", default = "False")
    parser.add_argument("-c","--flux_corresp", help="Path to the correspondences to calculate the flux")
    args = parser.parse_args()

    command = args.command
    image_folder = Path(r"{}".format(args.image_folder))
    kpts_folder = Path(r"{}".format(args.kpts_folder))
    matches_file = Path(r"{}".format(args.matches))
    out_folder = Path(r"{}".format(args.out_folder))
    error_threshold = args.error_threshold
    f_matrix_path = Path(r"{}".format(args.f_matrix))
    flux_path = Path(r"{}".format(args.flux_corresp))
    debug = args.debug
    use_flux = args.use_flux

    # Find the Foundamental matrix from input matches
    if command == "find_fundamental":
        kpts_dict = {}
        matches_dict = {}

        imgs_list = os.listdir(image_folder)
    
        # Importing keypoints from file
        for img in imgs_list:
            with open("{}/{}.txt".format(kpts_folder, img),'r') as kpt_file:
                kpts_dict[img] = np.loadtxt(kpt_file, delimiter=' ', skiprows=1, usecols=(0,1))

        # Importing matches from file
        controller = True
        with open(matches_file, 'r') as match_file:
            lines = match_file.readlines()
            for line in lines:
                if line == '\n':
                    controller = False
                else:
                    line = line.strip()
                    elem1, elem2 = line.split(' ', 1)
                    if elem1 in imgs_list:
                        img1 = elem1
                        img2 = elem2
                        pair = '{} {}'.format(img1, img2)
                        matches_dict[pair] = np.empty((2,2), dtype=float)
                    else:
                        matches_dict[pair] = np.vstack((matches_dict[pair],     np. array([elem1, elem2], ndmin=2)))

        for m in matches_dict:
            matches_dict[m] = matches_dict[m][2:, :]

        # Geometric verification
        verified_kpts = {}
        verified_matches = {}
        for key in matches_dict:
            match1 = matches_dict[key][:, 0]
            match2 = matches_dict[key][:, 1]
            match1 = match1.astype(int)
            match2 = match2.astype(int)
            img1, img2 = key.split(" ", 1)
            feat1 = kpts_dict[img1]
            feat2 = kpts_dict[img2]

            kpt_left = feat1[match1, :]
            kpt_right = feat2[match2, :]

            F, inliers = pydegensac.findFundamentalMatrix(kpt_left,     kpt_right,  error_threshold, 0.99, ITERATIONS)

            if img1 not in verified_kpts.keys():
                verified_kpts[img1] =  kpt_left[inliers]
            if img2 not in verified_kpts.keys():
                verified_kpts[img2] =  kpt_right[inliers]
            verified_matches[key] = np.array([(m, m) for m in range (kpt_left    [inliers].shape[0])])
            print(verified_matches[key])
            print(kpt_left[inliers].shape, kpt_left[inliers].shape)

        for img in imgs_list:
            with open("{}/{}.txt".format(out_folder, img), "w")   as    colmap_desc_file:
                colmap_desc_file.write("{} {}\n".format(verified_kpts[img]. shape    [0], 128))
                for row in range(verified_kpts[img].shape[0]):
                    colmap_desc_file.write("{} {} 0.000000 0.000000\n". format   (verified_kpts[img][row, 0], verified_kpts[img] [row, 1]))
                colmap_desc_file.write("\n")    

        with open("{}/verified_matches.txt".format  (out_folder),    "w") as colmap_matches:
            for math_pair in verified_matches:
                colmap_matches.write("{}\n".format(math_pair))
                for row in range(verified_matches[math_pair].shape[0]):
                    colmap_matches.write("{} {}\n".format(verified_matches      [math_pair][row, 0], verified_matches[math_pair][row,   1]))
                colmap_matches.write("\n")

        with open("{}/F_matrix.txt".format(out_folder),    "w") as F_file:
            F_file.write("{} {} {} {} {} {} {} {} {}".format(F[0,0], F[0,1], F[0,2], F[1,0], F[1,1], F[1,2], F[2,0], F[2,1], F[2,2]))
        
        if debug == "True":
            print('F', F)
            x1, y1 = 292, 191
            m = np.array([[x1], [y1], [1]])
            line_params = F @ m
            a, b, c = line_params[0,0], line_params[1,0], line_params[2,0]
            img2_pil = Image.open(r"{}/{}".format(image_folder, imgs_list[1]))
            line_extrema = [(0, -c/b),(-c/a, 0)]
            print(line_extrema)
            print((img2_pil.width, img2_pil.height))
            draw = ImageDraw.Draw(img2_pil)
            draw.line(line_extrema, fill=255)
            img2_pil.show()


    #######################################################################################
    ### EVALUATE MATCHES
    elif command == "evaluate_matches":
        print("\nImporting F:")
        with open(r"{}".format(f_matrix_path), 'r') as f_matrix_file:
            line = f_matrix_file.readlines()
            print(line)
            f1, f2, f3, f4, f5, f6, f7, f8, f9 = line[0].split(" ", 8)
            F = np.array([
                            [float(f1), float(f2), float(f3)],
                            [float(f4), float(f5), float(f6)],
                            [float(f7), float(f8), float(f9)]
                                                                ])
            if debug == "True": print(F)

        # Importing keypoints from file
        print("\nImporting keypoints from file ...")
        kpts_dict = {}
        matches_dict = {}
        imgs_list = os.listdir(image_folder)
    
        for img in imgs_list:
            with open("{}/{}.txt".format(kpts_folder, img),'r') as kpt_file:
                kpts_dict[img] = np.loadtxt(kpt_file, delimiter=' ', skiprows=1, usecols=(0,1))

            #if debug == "True":
            #    imPil = Image.open(image_folder / f"{img}")
            #    imPil.show()
            #    x_vec = kpts_dict[img][:,0].tolist()
            #    y_vec = kpts_dict[img][:,1].tolist()
            #    draw = ImageDraw.Draw(imPil)
            #    draw.point([(x, y) for (x, y) in zip(x_vec, y_vec)], fill = 'red')
            #    imPil.show()

        # Importing matches from file
        print("\nImporting matches from file ...")
        controller = True
        with open(matches_file, 'r') as match_file:
            lines = match_file.readlines()
            for line in lines:
                if line == '\n':
                    controller = False
                else:
                    line = line.strip()
                    elem1, elem2 = line.split(' ', 1)
                    if elem1 in imgs_list:
                        img1 = elem1
                        img2 = elem2
                        pair = '{} {}'.format(img1, img2)
                        matches_dict[pair] = np.empty((2,2), dtype=float)
                    else:
                        matches_dict[pair] = np.vstack((matches_dict[pair],     np. array([elem1, elem2], ndmin=2)))

        for m in matches_dict:
            matches_dict[m] = matches_dict[m][2:, :]
        
        if debug == "True": print(matches_dict)

        # Evaluation
        print("\nEvaluation ...")
        verified_kpts_img1 = []
        verified_kpts_img2 = []
        verified_matches = []
        cont = 0
        for key in matches_dict:
            match1 = matches_dict[key][:, 0]
            match2 = matches_dict[key][:, 1]
            match1 = match1.astype(int)
            match2 = match2.astype(int)
            img1, img2 = key.split(" ", 1)
            feat1 = kpts_dict[img1]
            feat2 = kpts_dict[img2]

            kpt_left = feat1[match1, :]
            kpt_right = feat2[match2, :]

            ### FLUX calculation
            if use_flux == "True":
                with open(flux_path, 'r') as correspondences_for_flux_file:
                    corresp = np.loadtxt(correspondences_for_flux_file, delimiter=' ')
                    x1_vec = corresp[:,0]
                    y1_vec = corresp[:,1]
                    x2_vec = corresp[:,2]
                    y2_vec = corresp[:,3]
                
                img1_pil = Image.open(r"{}/{}".format(image_folder, imgs_list[0]))
                width = img1_pil.width
                height = img1_pil.height
                draw = ImageDraw.Draw(img1_pil)
                for k in range(len(x1_vec)):
                    draw.line([(x1_vec[k], y1_vec[k]), (x2_vec[k], y2_vec[k])], fill='yellow', width=10)
                img1_pil.show()

                # Interpolation of the flux
                fx_vec = x2_vec - x1_vec
                fy_vec = y2_vec - y1_vec
                interp_x = LinearNDInterpolator(list(zip(x1_vec, y1_vec)), fx_vec)
                interp_y = LinearNDInterpolator(list(zip(x1_vec, y1_vec)), fy_vec)
                X = np.linspace(0, width, 200, dtype=int)
                Y = np.linspace(0, height, 200, dtype=int)
                X, Y = np.meshgrid(X, Y)
                print(X)

                Z1 = interp_x(X, Y)

                plt.pcolormesh(X, Y, Z1, shading='auto')
                #plt.plot(x, y, "ok", label="input point")
                plt.legend()
                plt.colorbar()
                plt.axis("equal")
                plt.show()
                
                Z2 = interp_y(X, Y)

                plt.pcolormesh(X, Y, Z2, shading='auto')
                #plt.plot(x, y, "ok", label="input point")
                plt.legend()
                plt.colorbar()
                plt.axis("equal")
                plt.show()

            for k in range(kpt_left.shape[0]):
                x1 = kpt_left[k, 0]
                y1 = kpt_left[k, 1]

                x2 = kpt_right[k, 0]
                y2 = kpt_right[k, 1]

                m = np.array([[x1], [y1], [1]])
                line_params = F @ m
                a, b, c = line_params[0,0], line_params[1,0], line_params[2,0]

                if debug == "True":
                    img2_pil = Image.open(r"{}/{}".format(image_folder, imgs_list[1]))
                    #line_extrema = [(0, -c/b),(-c/a, 0)]
                    width = img2_pil.width
                    height = img2_pil.height
                    width = img2_pil.width
                    l1 = (0, -c/b)
                    l2 = (width, (-c-a*width)/b)
                    l3 = (-c/a, 0)
                    l4 = ((-c-b*height)/a, height)
                    draw = ImageDraw.Draw(img2_pil)
                    #draw.line(line_extrema, fill=255)
                    draw.line([l1, l2], fill=255)
                    draw.line([l1, l3], fill=255)
                    draw.line([l1, l4], fill=255)
                    draw.line([l2, l3], fill=255)
                    draw.line([l2, l4], fill=255)
                    draw.line([l3, l4], fill=255)
                    draw.ellipse(((x2-15, y2-15), (x2+15, y2+15)), fill='red', outline='blue')
                    img2_pil.show()

                distance_point_line = np.absolute(a*x2 + b*y2 + c) / np.sqrt(a**2 + b**2)
                if debug == "True": print(distance_point_line)

                if distance_point_line < error_threshold and (y2-y1) < ALPHA_FLUX_MAX * interp_y(x1,y1) and (y2-y1) > ALPHA_FLUX_MIN * interp_y(x1,y1):
                    verified_kpts_img1.append((x1, y1))
                    verified_kpts_img2.append((x2, y2))
                    verified_matches.append("{} {}".format(cont, cont))
                    cont += 1
                


            with open("{}/{}.txt".format(out_folder, img1), "w")   as   colmap_desc_file:
                colmap_desc_file.write("{} {}\n".format(len(verified_kpts_img1), 128))
                for point in verified_kpts_img1:
                    colmap_desc_file.write("{} {} 0.000000 0.000000\n".format   (point[0], point[1]))
                colmap_desc_file.write("\n")  

            with open("{}/{}.txt".format(out_folder, img2), "w")   as   colmap_desc_file:
                colmap_desc_file.write("{} {}\n".format(len(verified_kpts_img2), 128))
                for point in verified_kpts_img2:
                    colmap_desc_file.write("{} {} 0.000000 0.000000\n".format   (point[0], point[1]))
                colmap_desc_file.write("\n") 

            with open("{}/verified_matches.txt".format  (out_folder),    "w") as colmap_matches:
                colmap_matches.write("{} {}\n".format(img1, img2))
                for element in verified_matches:
                    colmap_matches.write("{}\n".format(element))
                colmap_matches.write("\n")

    else:
        print("Error! Change command")
        quit()
    



