from database import COLMAPDatabase
from database import blob_to_array
from database import array_to_blob
from database import pair_id_to_image_ids
import numpy as np

db_path1 = r"/home/luca/Desktop/FixPos/less_keyframes_extended/RootSIFT/sift.db"
db1 = COLMAPDatabase.connect(db_path1)
db_path2 = r"/home/luca/Desktop/FixPos/less_keyframes_extended/SuperGlue/superglue.db"
db2 = COLMAPDatabase.connect(db_path2)
db_path3 = r"/home/luca/Desktop/FixPos/less_keyframes_extended/RSift_SG/sift_SG.db"
db3 = COLMAPDatabase.connect(db_path3)
db3.create_tables()


### MAIN ###
# Load cameras in merged database
rows = db2.execute("SELECT * FROM cameras")
for row in rows:
    camera_id, model, width, height, params, prior = row
    params = blob_to_array(params, np.float64)
    camera_id1 = db3.add_camera(model, width, height, params, prior)

# Read existing kpts from database 2
keypoints2 = dict(
    (image_id, blob_to_array(data, np.float32, (-1, 6)))
    for image_id, data in db2.execute(
        "SELECT image_id, data FROM keypoints"))

# Read existing matches from database 2
#matches2 = dict(
#    (pair_id_to_image_ids(pair_id),
#     blob_to_array(data, np.uint32, (-1, 2)))
#    for pair_id, data in db2.execute("SELECT pair_id, data FROM matches")
#)
matches2 = {}
for pair_id, r, c, data in db2.execute("SELECT pair_id, rows, cols, data FROM matches"):
    if data != None:
        pair_id = pair_id_to_image_ids(pair_id)
        matches2[(int(pair_id[0]), int(pair_id[1]))] = blob_to_array(data, np.uint32, (-1, 2))

two_views_matches2 = {}
for pair_id, r, c, data in db2.execute("SELECT pair_id, rows, cols, data FROM two_view_geometries"):
    if data != None:
        pair_id = pair_id_to_image_ids(pair_id)
        two_views_matches2[(int(pair_id[0]), int(pair_id[1]))] = blob_to_array(data, np.uint32, (-1, 2))

# Store all imgs
img_list = list(keypoints2.keys())

# Add images in merged database
imgs2 = dict(
    (image_id, (name, camera_id))
    for image_id, name, camera_id in db2.execute(
        "SELECT image_id, name, camera_id FROM images"))

for image_id in list(imgs2.keys()):
    db3.add_image(imgs2[image_id][0], imgs2[image_id][1])

# Read existing kpts from database 1
keypoints1 = dict(
    (image_id, blob_to_array(data, np.float32, (-1, 6)))
    for image_id, data in db1.execute(
        "SELECT image_id, data FROM keypoints"))
print("keypoints2 shape", np.shape(keypoints2[img_list[0]]))
print(keypoints2[img_list[0]][:5,:])
print("keypoints1 shape", np.shape(keypoints1[img_list[0]]))
print(keypoints1[img_list[0]][:5,:])

# Read existing matches from database 1
matches1 = {}
for pair_id, r, c, data in db1.execute("SELECT pair_id, rows, cols, data FROM matches"):
    if data != None:
        pair_id = pair_id_to_image_ids(pair_id)
        matches1[(int(pair_id[0]), int(pair_id[1]))] = blob_to_array(data, np.uint32, (-1, 2))

two_views_matches1 = {}
for pair_id, r, c, data in db1.execute("SELECT pair_id, rows, cols, data FROM two_view_geometries"):
    if data != None:
        pair_id = pair_id_to_image_ids(pair_id)
        two_views_matches1[(int(pair_id[0]), int(pair_id[1]))] = blob_to_array(data, np.uint32, (-1, 2))

n_kpts_dict = {}

# Merge kpts
for im_id in img_list:
    n_kpts = np.shape(keypoints1[im_id])[0]
    keypoints = np.vstack((keypoints1[im_id][:, :2], keypoints2[im_id][:, :2]))
    #keypoints = np.vstack((keypoints1[im_id], keypoints2[im_id]))
    db3.add_keypoints(im_id, keypoints)
    n_kpts_dict[im_id] = n_kpts

# Merge raw matches
all_matches = {}
for pair in matches1:
    all_matches[pair] = matches1[pair]

for pair in list(matches2.keys())[:]:
    im1 = int(pair[0])
    im2 = int(pair[1])
    matches2_im1 = matches2[pair][:, 0] + n_kpts_dict[im1]
    matches2_im2 = matches2[pair][:, 1] + n_kpts_dict[im2]
    matches2_im1 = matches2_im1.reshape((-1, 1))
    matches2_im2 = matches2_im2.reshape((-1, 1))
    joinedmatches2 = np.hstack((matches2_im1, matches2_im2))
    if pair not in matches1.keys():
        all_matches[pair] = joinedmatches2
    elif pair in matches1.keys():
        all_matches[pair] = np.vstack((matches1[pair], joinedmatches2))

for pair in all_matches:
    im1 = int(pair[0])
    im2 = int(pair[1])
    db3.add_matches(im1, im2, all_matches[pair])

# Merge verified matches
all_matches = {}
for pair in two_views_matches1:
    all_matches[pair] = two_views_matches1[pair]

for pair in list(two_views_matches2.keys())[:]:
    im1 = int(pair[0])
    im2 = int(pair[1])
    matches2_im1 = two_views_matches2[pair][:, 0] + n_kpts_dict[im1]
    matches2_im2 = two_views_matches2[pair][:, 1] + n_kpts_dict[im2]
    matches2_im1 = matches2_im1.reshape((-1, 1))
    matches2_im2 = matches2_im2.reshape((-1, 1))
    joinedmatches2 = np.hstack((matches2_im1, matches2_im2))
    if pair not in two_views_matches1.keys():
        all_matches[pair] = joinedmatches2
    elif pair in two_views_matches1.keys():
        all_matches[pair] = np.vstack((two_views_matches1[pair], joinedmatches2))

for pair in all_matches:
    im1 = int(pair[0])
    im2 = int(pair[1])
    db3.add_two_view_geometry(im1, im2, all_matches[pair])

db3.commit()
db1.close()
db2.close()
db3.close()