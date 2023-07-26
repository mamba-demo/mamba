# MAMBA <img src=https://raw.githubusercontent.com/andrewsong90/temp/main/docs/images/MAMBA_logo.png width="250" align="right"> 

### Weakly Supervised AI for Efficient Analysis of 3D Pathology Samples

[Arxiv] | [[Interactive Demo]](https://mamba-demo.github.io/demo/)

**M**odality-**A**gnostic **M**ultiple instance learning framework for volumetric **B**lock **A**nalysis (**MAMBA**) is a deep-learning-based computational pipeline for volumetric image analysis that can perform weakly-supervised patient prognostication based on 3D morphological features without the need for manual annotations by pathologists.
With the rapid growth and adoption of 3D spatial biology and pathology techniques by researchers and clinicians, MAMBA provides a general and efficient framework for 3D weakly supervised learning for clinical decision support and to reveal novel 3D morphological biomarkers and insights for prognosis and therapeutic response.  

<div>
<img src=https://raw.githubusercontent.com/andrewsong90/temp/main/docs/images/CT.png width="500" align="left">   
<div>
   <img src=https://raw.githubusercontent.com/andrewsong90/temp/main/docs/images/white.png width="100" align="right">
   <img src=https://raw.githubusercontent.com/andrewsong90/temp/main/docs/images/CT_heatmaps.gif width="300" align="right"> 
</div>
</div>

<div>
<img src=https://raw.githubusercontent.com/andrewsong90/temp/main/docs/images/OTLS.png width="500" align="left">
<div>
   <img src=https://raw.githubusercontent.com/andrewsong90/temp/main/docs/images/white.png width="250" align="right">
   <img src=https://raw.githubusercontent.com/andrewsong90/temp/main/docs/images/white.png width="200" align="right">
   <img src=https://raw.githubusercontent.com/andrewsong90/temp/main/docs/images/OTLS_heatmap.gif width="300" align="right"> 
</div>
</div>

<br><br><br><br><br><br><br><br><br><br><br><br><br><br><br><br><br><br><br>
## Updates
(07/17/23) The github is now live

## Installation
### Pre-requisites
* Python (3.9.0)
* pytorchvideo (0.1.5)

## Volumetric image Preprocessing
<img src=https://raw.githubusercontent.com/andrewsong90/temp/main/docs/images/workflow_github_preprocessing.png> 

### Tissue segmentation & patching
To perform segmenting/patching/stitching, run the following command (change the paths accordingly.)

```
cd preprocess
python create_patches_3D.py --patch_mode 3D --patch --seg --stitch --source /media/andrew/microCT/Prostate/tiff --save_dir ~/data/microCT/ --patch_size 96 --step_size 96 --save_mask --slice_mode all --thresh_mode fixed --process_list /home/andrew/workspace/ThreeDimPlayground/csv/process_list_seg.csv
```
Some flags include:
* `--slice_mode` (if `--patch_mode 2D`)
    * **single** (default): patch within single 2D slice at the depth with largest tissue contour area (indicated by best_slice_idx)
    * **all**: patch across all slices
    * **step**: patch within 2D slices that are certain steps apart (specify with `--step_z` flag)
* `--slice_mode` (if `--patch_mode 3D`)
    * **single** (default): patch within single 3D slice at the depth with largest tissue contour area (indicated by best_slice_idx)
    * **all** (recommended): patch across all 3D slices
* `--thresh_mode`
  * **fixed** (default): Uses csv-supplied clip_min, clip_max to threshold the images for all subjects. For CT, use this.
  * **global** (recommend for OTLS): Automatically identifies adaptive upper threshold for each subject (Top 1%). Lower threshold is set to csv-supplied clip_min.

The resulting h5 files will have the following format (e.g., subject name: '00001', block name: 'A')

**filename**: 0001-A_patches.h5
* 'imgs'
  * If patch_mode=3D, numpy array of (numOfpatches, C, Z, W, H)
  * If patch_mode=2D, numpy array of (numOfpatches, C, W, H)
* 'coords'
  * list of (z, x, y)

## Computational processing
<img src=https://raw.githubusercontent.com/andrewsong90/temp/main/docs/images/workflow_github_computational.png> 

### Feature extraction
To perform feature extraction, run the following command
```
CUDA_VISIBLE_DEVICES=0 python extract_features.py --dataroot /home/andrew/data/microCT --extracted_dir patch_96_step_96_3D_all --batch_size 50 --patch_mode 3D --process_list process_list_extract.csv --encoder SwinUNETR --augment_fold 5 --config conf/config_extraction.yaml --data_mode CT
```

Some flags include:
* `--process_list`: csv file with subject, image lower/upper threshold information
* `--patch_mode`: '2D' for stacks of 2D slices processing and '3D' for 3D processing
* `--batch_size`: Batch size (number of patches) for feature extraction
* `--data_mode`: Input data modality
  * 'CT' (applicable for 2D/3D data)
  * 'OTLS' (applicable for 2D/3D data)
* `--encoder`: **2D options** 'SwinUNETR', 'resnet18_2d', 'resnet34_2d', 'resnet50_2d', **3D options** 'SwinUNETR', 'resnet18_3d', 'resnet34_3d', 'resnet50_3d'
* `--augment_fold`: Number of augmentations to perform (if 0, no augmentation performed)

### Training
To run binary classification, run the following command
```
CUDA_VISIBLE_DEVICES=0 python train_cv.py --config conf/config_MIL.yaml --sample_prop 0.5 --split_mode kf
--attn_latent_dim 64 --es --encoder SwinUNETR --numOfaug 5 --seed 10 --task clf --prop_train 0.7 --val_aug
--feats_path '/home/andrew/data/microCT/patch_96_step_96_3D_all/SwinUNETR_h5_patch_features/swinvit_aug'
```

Some flags include:
* `--split_mode`: 'loo' for leave-one-out CV and 'kf' for k-fold CV
* `--prop_train`: Proportion of training dataset to use for training (rest is used for validation)
* `--sample_prop`: Proportion of patches to sample from each bag of patches
* `--numOfaug`: Number of augmentations
* `--task`: Classification (clf) or survival (surv)
* `--decoder_enc`: If specified, Add one-layer MLP encoder on top of the features for further encoding (Useful for learning more discriminative features at the risk of overfitting due to increased number of parameters)

### Testing
The trained models can be used to perform inference on a new sample
```
CUDA_VISIBLE_DEVICES=0 python inference.py --config conf/config_inference.yaml --mode external
```
Some flags include:
* `--mode`: Whether test data is external or internal (used in CV analysis). Required for selection of model checkpoints
  * 'internal': If the dataset was part of CV analysis, identify the CV-fold for which the test data was not part of the training dataset.
  * 'external': If not part of the cv-analysis, all models can be used to perform inference

## Post-hoc interpretation
To create interpretable heatmap imposed on the raw volumetric image
```
CUDA_VISIBLE_DEVICES=0 python create_heatmaps.py --config conf/config_heatmap_CT.yaml --mode full
```

## Contact
For any suggestions or issues, please contact Andrew H. Song <asong@bwh.harvard.edu>

## License
© This code is made available under the Commons Clasuse License and is available for non-commercial academic purposes.

<!--
<img src=https://raw.githubusercontent.com/andrewsong90/temp/main/docs/images/joint_logo.png> 


### Image objects
The image objects **WholeSlideImage**, **SerialTwoDimImage**, and **ThreeDimImage**, each of which is a container for a single image/scan
- **SerialTwoDimImage** This is for 3D image (.tiff, DICOM, NIFTI) and inherits the WholeSlideImage class. This is not truly 3D in a sense that the image is treated as a stack of 2D images. Therefore the segmentation/patching/stitching algorithms are essentially same as 2D, but adapted to treat stack of 2D images simultaneously.
- **ThreeDimImage** This is for 3D image (.tiff, DICOM, NIFTI) and inherits the SerialTwoDimImage class. The only difference from its parent so far is the patching algorithm. This performs basic 3D patching, where (x,y) coordinates are not aligned across the stack.

**Note that the coordinate system is (z, x, y) or (depth, width, height). This emulates how CNN input structures are formulated.**


### Step 4
After running several experiments, we can aggregate the results by running the following command
```
python scripts/aggregate_results.py --folder_path /home/andrew/workspace/results/MIL/clf_loo
```
This will generate *results_agg.csv* that will aggregate the test metrics.

### Step 5
To visualize a three dimensional image in Napari, you can run the following command:
```
python visualize.py /path/to/image_slices_folder/ --rgb --animation_mode
```
This will open the image slices in napari, which then enables easy 3d visualization as well as the ability to generate animations with the 3d image.

### Step 6
If you wish to generate a phantom dataset of cell-like structures with which to analyze in a pipeline, you can run the following script:
```
python gen_artificial_data.py ../data/ 30 --prefix gen-img --n_obj 500 --type cells --h 512 --w 512 --d 512
```
This will generate 30 artificial 3D images populated with cell-like structures, whose properties are determined by statistical distributions that can be manually modified via the gen_3d_img() function in utils/image_gen_utils.py.
-->
