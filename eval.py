#! -*- coding: utf-8 -*-
# Author: Yahui Liu <yahui.liu@unitn.it>
"""
metrics IS, FID, NDB and JSD.
"""

import os
import json
import ntpath
import numpy as np

import data_ios

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--metric', type=str, default='is', help='[is | fid | ndb | jsd | lpips]')
parser.add_argument('--pred_list', type=str, help='predict file list path')
parser.add_argument('--gt_list', type=str, help='real file list path')
parser.add_argument('--batch_size', type=int, default=8)
parser.add_argument('--gpu_id', type=str, default='0', help='default is 0th GPU')
parser.add_argument('--resize', type=int, default=128, help='128 for NDB and JSD; 299 for FID and IS')
parser.add_argument('--num_bins', default=100, help='used in NDB and JSD')
args = parser.parse_args()

def print_eval_log(opt):
    message = ''
    message += '----------------- Eval ------------------\n'
    for k, v in sorted(opt.items()):
        message += '{:>20}: {:<10}\n'.format(str(k), str(v))
    message += '----------------- End -------------------'
    print(message)

if __name__ == '__main__':
    use_cuda = args.gpu_id != ''
    if use_cuda:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
    
    batch_size = args.batch_size
    metric_mode = args.metric
    
    pred_list, gt_list = [], []
    with open(args.pred_list, 'r') as fin_pred:
        pred_list = [line.strip() for line in fin_pred]

    if metric_mode in ['fid', 'ndb', 'jsd']:
        with open(args.gt_list, 'r') as fin_gt:
            gt_list = [line.strip() for line in fin_gt]

    final_score = 0.0
    if metric_mode == 'is':
        import tensorflow as tf
        from scores.inception_score_tf import get_inception_score
        images = []
        for ll in pred_list:
            images.append(data_ios.imread(ll.strip(), args.resize))
        with tf.device('/device:GPU:{}'.format(args.gpu_id)):
             final_score, stddev = get_inception_score(images)
        print(final_score, stddev)
    elif metric_mode == 'fid':
        from scores.fid_scores import cal_fid as fid_score
        real_data_generator = data_ios.data_prepare_fid_is(gt_list, batch_size, args.resize, use_cuda)
        fake_data_generator = data_ios.data_prepare_fid_is(pred_list, batch_size, args.resize, use_cuda)
        dims = 2048
        final_score = fid_score(real_data_generator, fake_data_generator, dims, use_cuda)
    elif metric_mode in ['ndb', 'jsd']: 
        from scores.ndb_jsd import NDB
        real_images = data_ios.data_prepare_ndb_jsd(gt_list, args.resize)
        fake_images = data_ios.data_prepare_ndb_jsd(pred_list, args.resize)
        ndb_metric = NDB(training_data=src_images, number_of_bins=args.num_bins, 
            z_threshold=4, whitening=False, cache_folder='./{}'.format(args.dataset))
        results = ndb_metric.evaluate(trg_images, 'test')
        final_score = results['NDB']
    else:
        print('Unknown metric mode.')

    logs = {'num_of_files': len(pred_list),
            'metric_mode': metric_mode,
            'final_score': final_score}
    print_eval_log(logs)