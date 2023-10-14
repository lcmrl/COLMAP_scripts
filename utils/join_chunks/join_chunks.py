import numpy as np
from pathlib import Path
from transformations import affine_matrix_from_points

# Should be added RANSAC
# Now position is used for coregistration, but also orientation could be used

path_to_chunks = [
    Path(r"C:\Users\lmorelli\Desktop\Luca\GiHub_lcmrl\COLMAP_scripts\utils\join_chunks\c0\outs\pos.txt"),
    Path(r"C:\Users\lmorelli\Desktop\Luca\GiHub_lcmrl\COLMAP_scripts\utils\join_chunks\c1\outs\pos.txt"),
    Path(r"C:\Users\lmorelli\Desktop\Luca\GiHub_lcmrl\COLMAP_scripts\utils\join_chunks\c2\outs\pos.txt"),
    Path(r"C:\Users\lmorelli\Desktop\Luca\GiHub_lcmrl\COLMAP_scripts\utils\join_chunks\c3\outs\pos.txt"),
]


class Reconstruction:
    def __init__(self, path_to_camera_poses):
        self.camera_poses = {}
        camera_pos_file = open(path_to_camera_poses, 'r')
        cameras_poses = camera_pos_file.readlines()[2:]
        for line in cameras_poses:
            img_name, x, y, z, _ = line.split(' ', 4)
            self.camera_poses[img_name] = (x, y, z)
        camera_pos_file.close()
    
    def get_image_names(self):
        return list(self.camera_poses.keys())
    
    def get_camera_poses_as_matrix(self):
        image_names = list(self.camera_poses.keys())
        matrix_camera_pos = np.zeros((3, len(image_names)))
        for i, img in enumerate(image_names):
            x, y, z = self.camera_poses[img]
            matrix_camera_pos[:, i] = np.array([x, y, z]) 
        return image_names, matrix_camera_pos
    
    def build_homo_matrix(self, matrix):
        shape = matrix.shape
        ones_matrix = np.ones((1, shape[1]))
        homo_matrix = np.vstack((matrix, ones_matrix))
        return homo_matrix
    
    def add_img(self, img_name, x, y, z):
        if img_name not in list(self.camera_poses.keys()):
            self.camera_poses[img_name] = (x, y, z)
    
    def save_trajectory(self, output_file_path):
        with open(output_file_path, 'w') as out_file:
            images = self.get_image_names()
            for img in images:
                x, y, z = self.camera_poses[img]
                out_file.write(f"{img},{x},{y},{z}\n")


### MAIN ###
reconstructions = []

for i, path in enumerate(path_to_chunks):
    reconstructions.append(Reconstruction(path))

    if i == 0:
        continue

    reference = reconstructions[0]
    chunk = reconstructions[i]
    reference_imgs = reference.get_image_names()
    chunk_imgs = chunk.get_image_names()
    
    common_imgs = [x for x in chunk_imgs if x in reference_imgs]
    print(f'\ncommon_imgs chunk 0 vs chunk {i}')
    print(common_imgs)

    reference_pos_matrix = np.zeros((3, len(common_imgs)))
    chunk_pos_matrix = np.zeros((3, len(common_imgs)))

    for j, img in enumerate(common_imgs):
        xr, yr, zr = reference.camera_poses[img]
        x, y, z = chunk.camera_poses[img]

        reference_pos_matrix[:, j] = np.array([xr, yr, zr])
        chunk_pos_matrix[:, j] = np.array([x, y, z])

    transformation_matrix = affine_matrix_from_points(chunk_pos_matrix, reference_pos_matrix, shear=False, scale=True, usesvd=True)
    
    image_names, matrix_camera_pos = chunk.get_camera_poses_as_matrix()
    homo_matrix_camera_pos = chunk.build_homo_matrix(matrix_camera_pos)

    rotranslated_chunk_cameras = np.dot(transformation_matrix, homo_matrix_camera_pos)[:3, :]
    
    for j, img in enumerate(image_names):
        x, y, z = rotranslated_chunk_cameras[:, j]
        reference.add_img(img, x, y, z)
    
    reference.save_trajectory('./joined_trajectory.txt')



