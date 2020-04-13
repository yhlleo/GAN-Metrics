# Fréchet Inception Distance (FID)
# Based on: https://github.com/mseitzer/pytorch-fid

import numpy as np
from scipy import linalg
from inception import InceptionV3
import torch
from torch.nn.functional import adaptive_avg_pool2d

def inception_activations(data_generator, model, dims=2048, use_cuda=True):
    scores = []
    for batch in data_generator:
        pred = model(batch)[0]
        # If model output is not scalar, apply global spatial average pooling.
        # This happens if you choose a dimensionality not equal 2048.
        if pred.shape[2] != 1 or pred.shape[3] != 1:
            pred = adaptive_avg_pool2d(pred, output_size=(1, 1))
        scores.append(pred.view(-1, dims))
    acts = torch.cat(scores, dim=0).cpu().data.numpy()
    # return mu, sigma
    return np.mean(acts, axis=0), np.cov(acts, rowvar=False)

def cal_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """Numpy implementation of the Frechet Distance.
    The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
    and X_2 ~ N(mu_2, C_2) is
            d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).

    Stable version by Dougal J. Sutherland.

    Params:
    -- mu1   : Numpy array containing the activations of a layer of the
               inception net (like returned by the function 'get_predictions')
               for generated samples.
    -- mu2   : The sample mean over activations, precalculated on an
               representative data set.
    -- sigma1: The covariance matrix over activations for generated samples.
    -- sigma2: The covariance matrix over activations, precalculated on an
               representative data set.

    Returns:
    --   : The Frechet Distance.
    """
    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert mu1.shape == mu2.shape, \
        'Training and test mean vectors have different lengths'
    assert sigma1.shape == sigma2.shape, \
        'Training and test covariances have different dimensions'

    diff = mu1 - mu2

    # Product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = ('fid calculation produces singular product; '
               'adding %s to diagonal of cov estimates') % eps
        print(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError('Imaginary component {}'.format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return (diff.dot(diff) + np.trace(sigma1) +
            np.trace(sigma2) - 2 * tr_covmean)


def cal_fid(data1, data2, dims, use_cuda):
    """Calculates the FID of two data generator.
    Params:
    -- data1   : generator of data one.
    -- data2   : generator of data two.
    -- use_cuda: use cuda or not.
    -- dims.   : feature dimensionality 2048.
    """
    block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[dims]
    model = InceptionV3([block_idx])
    if use_cuda:
        model.cuda()
    model.eval()

    m1, s1 = inception_activations(data1, model, dims, use_cuda)
    m2, s2 = inception_activations(data2, model, dims, use_cuda)
    fid_value = cal_frechet_distance(m1, s1, m2, s2)

    return fid_value
