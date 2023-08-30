import cv2
import torch
from kornia_moons.feature import *
import pycolmap
import os
import h5py
import kornia as K
import kornia.feature as KF
from fastprogress import progress_bar
import matplotlib.pyplot as plt

def load_torch_image(fname, device=torch.device('cpu')):
    img = K.image_to_tensor(cv2.imread(fname), False).float() /255.
    img = K.color.bgr_to_rgb(img.to(device))
    return img

device = torch.device('mps')
device = torch.device('cuda')


dirname = 'images'
img_fnames = [os.path.join(dirname, x) for x in os.listdir(dirname) if '.jpg' in x]




def detect_features(img_fnames,
                    num_feats = 2048,
                    upright = True,
                    device=torch.device('cpu'),
                    feature_dir = '.featureout', resize_to = (800, 608)):
    disk = KF.DISK.from_pretrained('depth').to(device)
    if not os.path.isdir(feature_dir):
        os.makedirs(feature_dir)
    with h5py.File(f'{feature_dir}/lafs.h5', mode='w') as f_laf, \
         h5py.File(f'{feature_dir}/keypoints.h5', mode='w') as f_kp, \
         h5py.File(f'{feature_dir}/descriptors.h5', mode='w') as f_desc:
        for img_path in progress_bar(img_fnames):
            img_fname = img_path.split('/')[-1]
            key = img_fname
            with torch.inference_mode():
                timg = load_torch_image(img_path, device=device)
                H, W = timg.shape[2:]
                if H < W:
                    resize_to = resize_to[1], resize_to[0]
                timg_resized = K.geometry.resize(timg, resize_to, antialias=True)
                h, w = timg_resized.shape[2:]
                features = disk(timg_resized, num_feats, pad_if_not_divisible=True)[0]
                kps1, descs = features.keypoints, features.descriptors
                lafs = KF.laf_from_center_scale_ori(kps1[None],
                                             torch.ones(1, len(kps1), 1, 1,device=device))
                lafs[:,:,0,:] *= float(W) / float(w)
                lafs[:,:,1,:] *= float(H) / float(h)
                desc_dim = descs.shape[-1]
                kpts = KF.get_laf_center(lafs).reshape(-1, 2).detach().cpu().numpy()
                descs = descs.reshape(-1, desc_dim).detach().cpu().numpy()
                f_laf[key] = lafs.detach().cpu().numpy()
                f_kp[key] = kpts
                f_desc[key] = descs
    return
    
    
detect_features(img_fnames, 8192, device=device, resize_to=(1600, 1216))


feature_dir = '.featureout'
with h5py.File(f'{feature_dir}/lafs.h5', mode='r') as f_laf:
    img1 = load_torch_image(img_fnames[0])
    key = img_fnames[0].split('/')[-1]
    lafs = torch.from_numpy(f_laf[key][...])
    visualize_LAF(img1, lafs)



def get_unique_idxs(A):
    # https://stackoverflow.com/questions/72001505/how-to-get-unique-elements-and-their-firstly-appeared-indices-of-a-pytorch-tenso
    unique, idx, counts = torch.unique(A, dim=0, sorted=True, return_inverse=True, return_counts=True)
    _, ind_sorted = torch.sort(idx, stable=True)
    cum_sum = counts.cumsum(0)
    cum_sum = torch.cat((torch.tensor([0],device=cum_sum.device), cum_sum[:-1]))
    first_indicies = ind_sorted[cum_sum]
    return first_indicies

def match_features(img_fnames,
                   index_pairs,
                   feature_dir = '.featureout',
                   device=torch.device('cpu'),
                   min_matches=15, force_mutual = True):
    with h5py.File(f'{feature_dir}/lafs.h5', mode='r') as f_laf, \
         h5py.File(f'{feature_dir}/descriptors.h5', mode='r') as f_desc, \
        h5py.File(f'{feature_dir}/matches.h5', mode='w') as f_match:
        adalam_config = KF.adalam.get_adalam_default_config()
        #adalam_config['orientation_difference_threshold'] = None
        #adalam_config['scale_rate_threshold'] = None
        adalam_config['force_seed_mnn']= False
        adalam_config['search_expansion'] = 16
        adalam_config['ransac_iters'] = 256
        for pair_idx in progress_bar(index_pairs):
                    idx1, idx2 = pair_idx
                    fname1, fname2 = img_fnames[idx1], img_fnames[idx2]
                    key1, key2 = fname1.split('/')[-1], fname2.split('/')[-1]
                    lafs1 = torch.from_numpy(f_laf[key1][...]).to(device)
                    lafs2 = torch.from_numpy(f_laf[key2][...]).to(device)
                    desc1 = torch.from_numpy(f_desc[key1][...]).to(device)
                    desc2 = torch.from_numpy(f_desc[key2][...]).to(device)
                    img1, img2 = cv2.imread(fname1), cv2.imread(fname2)
                    hw1, hw2 = img1.shape[:2], img2.shape[:2]
                    dists, idxs = KF.match_adalam(desc1, desc2,
                                                  lafs1, lafs2, # Adalam takes into account also geometric information
                                                  hw1=hw1, hw2=hw2,
                                                  config=adalam_config) # Adalam also benefits from knowing image size
                    if len(idxs)  == 0:
                        continue
                    # Force mutual nearest neighbors
                    if force_mutual:
                        first_indicies = get_unique_idxs(idxs[:,1])
                        idxs = idxs[first_indicies]
                        dists = dists[first_indicies]
                    n_matches = len(idxs)
                    if False:
                        print (f'{key1}-{key2}: {n_matches} matches')
                    group  = f_match.require_group(key1)
                    if n_matches >= min_matches:
                         group.create_dataset(key2, data=idxs.detach().cpu().numpy().reshape(-1, 2))
    return



# matching all to all
index_pairs = []
for i in range(len(img_fnames)):
    for j in range(i+1, len(img_fnames)):
        index_pairs.append((i,j))



match_features(img_fnames, index_pairs, device=torch.device('cuda'))



from h5_to_db import add_keypoints, add_matches, COLMAPDatabase

def import_into_colmap(img_dir,
                       feature_dir ='.featureout',
                       database_path = 'colmap.db',
                       img_ext='.jpg'):
    db = COLMAPDatabase.connect(database_path)
    db.create_tables()
    single_camera = False
    fname_to_id = add_keypoints(db, feature_dir, img_dir, img_ext, 'simple-radial', single_camera)
    add_matches(
        db,
        feature_dir,
        fname_to_id,
    )

    db.commit()
    return

database_path = 'colmap.db'
import_into_colmap(dirname, database_path=database_path)


pycolmap.match_exhaustive('./colmap.db')

    
    
    


    
    

