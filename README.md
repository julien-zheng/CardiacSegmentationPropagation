# 3D Consistent & Robust Segmentation of Cardiac Images by Deep Learning with Spatial Propagation

This is an implementation of the models in the following paper which is published in **IEEE Transactions on Medical Imaging**:

	3D Consistent & Robust Segmentation of Cardiac Images by Deep Learning with Spatial Propagation
	Qiao Zheng, Hervé Delingette, Nicolas Duchateau, Nicholas Ayache
	IEEE Transactions on Medical Imaging, 2018


**In case you find the code useful, please consider giving appropriate credit to it by citing the paper above.**

```
@ARTICLE{Qiao:TMI:2018,
	Author = {Zheng, Q and Delingette, H and Duchateau, N and Ayache, N},
	Journal = {IEEE Trans Med Imaging},
	Title = {3D consistent & robust segmentation of cardiac images by deep learning with spatial propagation},
	Volume = {?},
	Pages = {?-?},
	Year = {2018}
}

```

```
DOI: 10.1109/TMI.2018.2820742
```

## Requirements
The code should work with both Python 2 and Python 3, as the compatibility with both has been taken into consideration. It depends on some external libraries which should be installed, including:
- tensorflow
- keras 
- numpy and scipy
- math
- PIL
- cv2
- nibabel
- itertools
- copy
- six
- threading
- warnings
- multiprocessing
- functools
- h5py

On the other hand, **to apply this package on any data, the images should be named, formatted, and arranged accordingly. Some other important information (e.g. the index of the base slice) should also be provided as input**. Please refer to Section VI for more details.


## Usage
The steps of the cardiac segmentation method presented in the paper are described below, along with the corresponding files. **First, modify the `data_root` and `code_root` in *config.py* to the corresponding paths of the data root directory and the code root directory.** This is the very first step in the usage as these paths are necessary for the other scripts. The default values of the other variables in *config.py* are those used by the paper. Then, read the following sections according to your application scenario:
- If the UK Biobank dataset is available and you want to train and test the networks yourself as we have done, read from Section I to Section V.
- If the UK Biobank dataset is available and you are only interested in applying the pretrained networks on the test set of UK Biobank, you may mainly focus on the sections I, II.2, II.3, III.2, and IV.2.
- If you want to train and/or test the networks using another dataset instead of UK Biobank, useful details are provided in Section VI.

Depending on the version of this package, it may or may not contain the pretrained weights of the networks (the .h5 files). For instance, due to the constraints on the size of files, the version on GitHub does not contain the pretrained weights. **If the pretrained weights are needed but not included in the package, they can be downloaded by running the following script**:
```
python download_weights.py
```

### I. Preprocessing of UK Biobank Data
In this section, the preprocessing of UK Biobank data is presented step by step. However, it is also possible to preprocess other datasets accordingly and then use them to train or test our networks (more details are presented in Section VI).

#### I.1 Conversion from the original formats
Download, preprocess and convert the original DICOM images and CVI42 ground-truth segmentation to NIfTI format, which is more convenient for visualization and analysis. The code for this step is provided by Wenjia Bai (Imperial College London, UK) and can be found in https://github.com/baiwenjia/ukbb_cardiac/tree/master/data. More specifically, the following 3 files in https://github.com/baiwenjia/ukbb_cardiac/tree/master/data are used for the conversion of format:
- convert_data_ukbb2964.py: the main script for this step
- parse_cvi42_xml.py: parser for cvi42 contour xml files
- biobank_utils.py: converter from biobank dicom to nifti

#### I.2 Save as 2D images
```
python Preprocessing/convert_nifti_to_2D.py
```
To accelerate the reading of the image of a short-axis slice at an instant, the short-axis images, as well as their ground-truth segmentation (if exists), are converted to and saved as PNG format. The following file is for this step:
- Preprocessing/convert_nifti_to_2D.py

#### I.3 Statistics
```
python Preprocessing/statistics.py
```
Save the useful statistics for each subject to a file. The following file is for this step:
- Preprocessing/statistics.py

and the statistics are saved to:
- Preprocessing/statistics_record.txt

#### I.4 Visually check the quality of the ground-truth
We visually checked the cases and found about one thousand of which the ground-truth segmentation is either unconvincing (visually significant image/mask mismatch) or incomplete (e.g. missing ground-truth segmentation on some slices). These cases are excluded from the dataset for further analysis. The IDs of these cases are saved in:
- Preprocessing/doubtful_segmentation_cases2.txt

#### I.5 Adapt ground-truth
```
python Preprocessing/adapt_ground_truth.py 
```
The ground-truth segmentations are adapted (e.g. removal of the masks of the heart on the slices above the base) using:
- Preprocessing/adapt_ground_truth.py 

and the base slice indices are saved to
- Preprocessing/base_slices.txt



### II. Region of Interest (ROI) Determination

#### II.1 Training
```
python ROI/train_roi_net.py
```
Define and train the ROI-net:
- ROI/train_roi_net.py: the main file to launch the training
- ROI/data_roi_train.py: a function to generate lists of files for training
- ROI/module_roi_net.py: the module that defines ROI-net
- ROI/model_roi_net_epoch050.h5: the weights of the trained model

#### II.2 Prediction
```
python ROI/predict_roi_net.py
```
- ROI/predict_roi_net.py: the main file to launch the prediction
- ROI/data_roi_predict.py: a function to generate lists of files for prediction

#### II.3 ROI cropping
```
python ROI/crop_according_to_roi.py
```
- ROI/crop_according_to_roi.py

and the ROI information for each case is saved to
- ROI/roi_record.txt



### III. Left and Right Ventricles Segmentation Using LVRV-net

#### III.1 Training
```
python LVRV_Segmentation/train_lvrv_net.py
```
Define and train the LVRV-net:
- LVRV_Segmentation/train_lvrv_net.py: the main file to launch the training
- LVRV_Segmentation/data_lvrv_train.py: a function to generate lists of files for training
- LVRV_Segmentation/module_lvrv_net.py: the module that defines LVRV-net
- LVRV_Segmentation/model_lvrv_net_epoch080.h5: the weights of the trained model

#### III.2 Prediction
```
python LVRV_Segmentation/predict_lvrv_net.py
```
- LVRV_Segmentation/predict_lvrv_net.py: the main file to launch the prediction
- LVRV_Segmentation/data_seg_predict.py: a function to generate lists of files for prediction



### IV. Left Ventricle Segmentation Using LV-net

#### IV.1 Training
```
python LV_Segmentation/train_lv_net.py
```
Define and train the LV-net:
- LV_Segmentation/train_lv_net.py: the main file to launch the training
- LV_Segmentation/data_lv_train.py: a function to generate lists of files for training
- LV_Segmentation/module_lv_net.py: the module that defines LV-net
- LV_Segmentation/model_lv_net_epoch080.h5: the weights of the trained model

#### IV.2 Prediction
```
python LV_Segmentation/predict_lv_net.py
```
- LV_Segmentation/predict_lv_net.py: the main file to launch the prediction
- LV_Segmentation/data_seg_predict.py: a function to generate lists of files for prediction



### V. Auxiliary functions
Just for information, the auxiliary functions are defined in the following files:
- helpers.py
- image2.py

In particular, image2.py is used for real-time data augmentation. It is adapted from the code in a file of the Keras project (https://github.com/keras-team/keras/blob/master/keras/preprocessing/image.py).



### VI. Using Other Datasets
Instead of using UK Biobank data, it is also possible to preprocess another dataset accordingly and then use it to train and/or test our networks, as long as the following requirements are met:

#### VI.1 Image Name and Format
In the data directory (specified by `data_root` in *config.py*), the path of a 2D image, which is identified by its case ID string `C` (e.g. '1234567'), the two-digit slice index `S` (e.g. 02) in the stack, and the two-digit instant index `I` (e.g. 14) in the temporal sequence, should be the following:
- *`'{C}/original_2D/original_2D_{S}_{I}.png'`* (e.g. *'1234567/original_2D/original_2D_02_14.png'*)

The corresponding ground-truth, after adaptation if necessary, should have the path:
- *`'{C}/original_2D/original_gt2_2D_{S}_{I}.png'`* (e.g. *'1234567/original_2D/original_gt2_2D_02_14.png'*)

**Please note that the two-digit slice index `S` in the stack should be arranged to increment slice by slice from the base to the apex. This is essential as it makes sure that the propagation is performed in the correct base-to-apex order.**


#### VI.2 Statistics File
As indicated in line 148 of
- Preprocessing/statistics.py

for each case, there are 17 necessary iterms (case ID, indicators of complete short-axis images and ground-truth (0 for imcompleteness, 1 for completeness), dimensions of the image stack squences, instant indices of end of diastole (ED) and end of systole (ES), slice indices of the first and the last slices in the stacks of ED and ES, spacing along different dimensions) to be stored as a space separated list in a line of the statistics file. So the file should look like:

	1001553 1 1 186 208 9 50 0 17 0 8 1 8 1.82692 1.82692 10.0 0.0183
	1003105 1 1 210 208 11 50 0 20 1 10 2 10 1.94231 1.94231 10.0 0.0209
	1003177 1 1 204 208 11 50 0 13 0 9 2 9 1.82692 1.82692 10.0 0.02706
	...

The statistics file should be saved as:
- Preprocessing/statistics_record.txt

#### VI.3 File of Excluded Cases
Set up a file for excluded cases (e.g. those of inappropriate or insufficient image quality, those of missing ground-truth), which contains a column of IDs (each line is the ID of an excluded case):

	1005001
	1011609
	1012733
	...

The file should be saved as:
- Preprocessing/doubtful_segmentation_cases2.txt

#### VI.4 File of Base Slices
For each used (equivalently speaking, not exluded as described in Sub-Section VI.3 above) case, the indices of the base slices at ED and ES should be determined and available. A file like

	1001553 0 2
	1003105 2 4
	1003177 0 2
	...

should be saved as
- Preprocessing/base_slices.txt

**Please note that providing a reasonably well estimated base slice index is important. If an incorrect index is provided such that the propagation starts from some slice located well above the base (e.g. a slice containing the atriums), it might not work well.**

#### VI.5 ROI-net
Modify the definitions of the training dataset (`train_statistics`) and the testing dataset (`test_statistics`) in the following files:
- ROI/data_roi_train.py
- ROI/data_roi_predict.py

Then proceed as indicated in Section II.

#### VI.6 LVRV-net
Modify the definitions of the training dataset (`train_statistics`) and the testing dataset (`test_statistics`) in the following files:
- LVRV_Segmentation/data_lvrv_train.py
- LVRV_Segmentation/data_seg_predict.py

Then proceed as indicated in Section III.

#### VI.7 LV-net
Modify the definitions of the training dataset (`train_statistics`) and the testing dataset (`test_statistics`) in the following files:
- LV_Segmentation/data_lv_train.py
- LV_Segmentation/data_seg_predict.py

Then proceed as indicated in Section IV.









