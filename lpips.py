
import os
import numpy as np

import models
from util import util
import argparse
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--path', type=str, default='./', help='data path')
parser.add_argument('--use_gpu', action='store_true', help='turn on flag to use GPU')
parser.add_argument('--test_list', type=str, default='')
opt = parser.parse_args()

## Initializing the model
model = models.PerceptualLoss(model='net-lin',net='alex',use_gpu=opt.use_gpu)
dists = []

# Load images
with open(opt.test_list, 'r') as f:
    for ll in f:
        path0, path1 = ll.strip().split('\t')
        print(path0, path1)
        img0 = util.im2tensor(util.load_image(os.path.join(opt.path, path0+'.png')))
        img1 = util.im2tensor(util.load_image(os.path.join(opt.path, path1+'.png')))

        if opt.use_gpu:
            img0 = img0.cuda()
            img1 = img1.cuda()

        # Compute distance
        dist01 = model.forward(img0,img1).data.cpu().squeeze().numpy()
        print('Distance: %.4f'%dist01)
        dists.append(dist01)

print('Average distance: %.4f'%(sum(dists)/len(dists)))
print('Standard deviation:', np.array(dists).std())
