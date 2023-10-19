import os

img_dir = r"/home/luca/Desktop/temp/join/ventimiglia_nadiral"
images = os.listdir(img_dir)
images.sort()

with open('./imagest.txt', 'w') as file:
    for image in images:
        file.write(f"{image}\n")