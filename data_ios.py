# Author: Yahui Liu <yahui.liu@unitn.it>

import os
import cv2
#import glob
import numpy as np

import torch
import torch.nn.functional as F
import torchvision.transforms as transforms

try:
    from tqdm import tqdm
except ImportError:
    # If not tqdm is not available, provide a mock version of it
    def tqdm(x): return x

def imread(file, resize=128):
    im = cv2.imread(file, cv2.IMREAD_UNCHANGED)
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    return cv2.resize(im, (resize, resize), interpolation=cv2.INTER_CUBIC)

def data_prepare_ndb_jsd(img_list, resize=128):
    assert len(img_list), "Empty list"
    images = np.zeros((len(img_list), resize*resize*3))
    for idx, ll in enumerate(img_list):
        im = imread(ll, resize)
        images[idx] = im.reshape([-1]).astype(float)/255.0
    return images

# build an iterable generator 
def data_prepare_fid_is(files, batch_size=8, resize=299, use_cuda=False):
    assert len(files), "Empty list"
    n_batches = len(files) // batch_size

    if len(files) % batch_size != 0:
        n_batches += 1

    if batch_size > len(files):
        batch_size = len(files)

    for i in tqdm(range(n_batches)):
        print('\rPropagating batch %d/%d' % (i+1, n_batches))
        start = i * batch_size
        end = start + batch_size
        end = end if end <= len(files) else len(files)
        images = np.array([imread(str(f), resize).astype(np.float32)
            for f in files[start:end]])

        images = images.transpose((0, 3, 1, 2))
        images /= 255.0
        batch = torch.from_numpy(images).float()
        
        if use_cuda:
            batch = batch.cuda()
        yield batch
