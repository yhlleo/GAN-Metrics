 # Metrics of GANs

  - [x] Fréchet Inception Distance (**FID**)
  - [x] Inception Score (**IS**)
  - [x] Number of statistically-Different Bins (**NDB**) 
  - [x] Jensen-Shannon Divergence (**JSD**)
  - [x] Learned Perceptual Image Patch Similarity (**LPIPS**)

|Metric|Usage|Notes|
|:----:|:----:|:----:|
|FID|Image Quality|the lower, the better|
|IS|Image Quality|the higher, the better|
|NDB|Diversity|the lower, the better|
|JSD|Diversity|the lower, the better|
|LPIPS|Diversity|the higher, the better|


## Configuration

See the [`environment.yaml`](./environment.yaml). We provide an user-friendly configuring method via Conda system, and you can create a new Conda environment using the command:

```
conda env create -f environment.yaml
```

## Usage

 - IS:
```
python eval.py --metric is --pred_list <path/to/pred_list> --gpu_id 0 --resize 299
```

 - FID:

```
python eval.py --metric fid --pred_list <path/to/pred_list> --gt_list <path/to/gt_list> --gpu_id 0 --resize 299
```

 - NBD & JSD:
```
python eval.py --metric ndb --pred_list <path/to/pred_list> --gt_list <path/to/gt_list> --gpu_id 0 --resize 128
```

 - LPIPS:

```
python lpips.py --path <path/to/image_folder> --test_list <path/to/test_list>
```

Example of test list in LPIPS:

```
a1.png	a2.png
a1.png	a3.png
...
```


