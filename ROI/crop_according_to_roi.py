""" The main file to launch ROI cropping according to the prediction of ROI-net """

import os
import sys
sys.path.append('..')
sys.path.append(os.path.join('.', 'Preprocessing') )
sys.path.append(os.path.join('..', 'Preprocessing') )

import numpy as np
import scipy
from scipy import ndimage
import math
import nibabel as nib
from PIL import Image

import multiprocessing.pool
from functools import partial
from adapt_ground_truth import adapt_ground_truth

import config


# Auxiliary function
def determine_rectangle_roi(img_path):
    img = Image.open(img_path)
    columns, rows = img.size
    roi_c_min = columns
    roi_c_max = -1
    roi_r_min = rows
    roi_r_max = -1
    box = img.getbbox()
    if box:
        roi_r_min = box[0]
        roi_c_min = box[1]
        roi_r_max = box[2] - 1
        roi_c_max = box[3] - 1
    return [roi_c_min, roi_c_max, roi_r_min, roi_r_max]

# Auxiliary function
def determine_rectangle_roi2(img_path):
    img = Image.open(img_path)
    img_array = np.array(img)
    connected_components, num_connected_components = ndimage.label(img_array)
    if (num_connected_components > 1):
        unique, counts = np.unique(connected_components, return_counts=True)
        max_idx = np.where(counts == max(counts[1:]))[0][0]
        single_component = connected_components * (connected_components == max_idx)
        img = Image.fromarray(single_component)

    columns, rows = img.size
    roi_c_min = columns
    roi_c_max = -1
    roi_r_min = rows
    roi_r_max = -1
    box = img.getbbox()
    if box:
        roi_r_min = box[0]
        roi_c_min = box[1]
        roi_r_max = box[2] - 1
        roi_c_max = box[3] - 1
    return [roi_c_min, roi_c_max, roi_r_min, roi_r_max]




def crop_according_to_roi():

    data_dir = config.data_root
    code_dir = config.code_root

    statistics_file = os.path.join(code_dir, 'Preprocessing', 'statistics_record.txt')
    doubtful_case_file = os.path.join(code_dir, 'Preprocessing', 'doubtful_segmentation_cases2.txt')
    roi_record_file = os.path.join(code_dir, 'ROI', 'roi_record.txt')
    roi_record = open(roi_record_file, 'w')

    # Read the statistics
    with open(statistics_file) as s_file:
        statistics = s_file.readlines()

    statistics = [x.strip() for x in statistics]
    statistics = [[int(z) for z in y.split()[:-4]] + [float(z) for z in y.split()[-4:]] \
        for y in statistics]

    # Read the doubtful cases
    with open(doubtful_case_file) as d_file:
        doubtful_cases = d_file.readlines()

    doubtful_cases = [x.strip() for x in doubtful_cases]
    doubtful_cases = [int(x) for x in doubtful_cases]    

    # We use a case if it's not among the doubtful cases, has both image and ground-truth 
    # files, and has complete ground-truth on both ED and ES.
    used_statistics = [k for k in statistics if (k[0] not in doubtful_cases) and 
        (k[1]==1) and (k[2]==1) and (k[7]>=0) and (k[8]>=0)]

    print('There will be {} used eids'.format(len(used_statistics)) )  


    # The ratio that determines the width of the margin
    pixel_margin_ratio = 0.3
    
    # If for a case there is non-zero pixels on the border of ROI, the case is stored in
    # this list for further examination. This list is eventually empty for UK Biobank cases.
    border_problem_eid = []

    for k in used_statistics:

        eid = k[0]
        rows = k[3]
        columns = k[4]
        slices = k[5]
        times = k[6]
        ed_es_instant0 = k[7]
        ed_es_instant1 = k[8]
        ed_es_instant0_min_slice = k[9]
        ed_es_instant0_max_slice = k[10]
        ed_es_instant1_min_slice = k[11]
        ed_es_instant1_max_slice = k[12]

        print('Processing eid = {} (# {})'.format(eid, used_statistics.index(k)) )

        original_2D_path = os.path.join(data_dir, str(eid), 'original_2D')
        mask_original_2D_path = os.path.join(data_dir, str(eid), 'mask_original_2D')
        crop_2D_path = os.path.join(data_dir, str(eid), 'crop_2D')

        if not os.path.exists(crop_2D_path):
            os.makedirs(crop_2D_path)

        # Only use the predicted masks at instant ED
        used_instants_roi = []
        if (ed_es_instant0 >= 0):
            used_instants_roi += [ed_es_instant0]

        img_path_list = []
        for t in used_instants_roi:
            for s in range(int(round(slices * 0.2 + 0.001)), int(round(slices * 0.6 + 0.001))):
                s_t_mask_image_file = os.path.join(mask_original_2D_path, 'mask_original_2D_{}_{}.png'.format(str(s).zfill(2), str(t).zfill(2)) )
                img_path_list.append(s_t_mask_image_file)
    
        # Multithread
        pool = multiprocessing.pool.ThreadPool()
        function_partial = partial(determine_rectangle_roi2)
        roi_results = pool.map(function_partial, (img_path for img_path in img_path_list))
        roi_c_min = min([res[0] for res in roi_results])
        roi_c_max = max([res[1] for res in roi_results])
        roi_r_min = min([res[2] for res in roi_results])
        roi_r_max = max([res[3] for res in roi_results])
        pool.close()
        pool.join()

        # ROI size (without adding margin)
        roi_c_length = roi_c_max - roi_c_min + 1
        roi_r_length = roi_r_max - roi_r_min + 1
        roi_length = max(roi_c_length, roi_r_length)
        print('roi_length = {}'.format(roi_length) )

        written = '{0} {1} {2} {3} {4} {5}\n'.format(eid, roi_c_min, roi_c_max, roi_r_min, roi_r_max, roi_length)
        roi_record.write(written)

        # The size of margin, determined by the ratio we defined above
        pixel_margin = int(round(pixel_margin_ratio * roi_length + 0.001))

        crop_c_min = ((roi_c_min + roi_c_max) // 2) - (roi_length // 2) - pixel_margin
        crop_c_max = crop_c_min + pixel_margin + roi_length - 1 + pixel_margin
        crop_r_min = ((roi_r_min + roi_r_max) // 2) - (roi_length // 2) - pixel_margin
        crop_r_max = crop_r_min + pixel_margin + roi_length - 1 + pixel_margin


        # Crop the original images and labels
        image_file = os.path.join(data_dir, str(eid), 'sa.nii.gz')
        image_load = nib.load(image_file)
        image_data = image_load.get_data()
        original_r_min = max(0, crop_r_min)
        original_r_max = min(image_data.shape[0]-1, crop_r_max)
        original_c_min = max(0, crop_c_min)
        original_c_max = min(image_data.shape[1]-1, crop_c_max)
        crop_image_data = np.zeros((roi_length + 2 * pixel_margin, roi_length + 2 * pixel_margin,
                                    image_data.shape[2], image_data.shape[3]))
        crop_image_data[(original_r_min - crop_r_min):(original_r_max - crop_r_min + 1), 
                        (original_c_min - crop_c_min):(original_c_max - crop_c_min + 1), 
                        :, 
                        :] = \
            image_data[original_r_min:(original_r_max + 1), 
                       original_c_min:(original_c_max + 1), 
                       :, 
                       :]
        crop_image_data = crop_image_data[::-1, ::-1, :, :]
        crop_image_file = os.path.join(data_dir, str(eid), 'crop_sa.nii.gz')
        nib.save(nib.Nifti1Image(crop_image_data, np.eye(4)), crop_image_file)


        label_file = os.path.join(data_dir, str(eid), 'label_sa.nii.gz')
        label_load = nib.load(label_file)
        label_data = label_load.get_data()
        crop_label_data = np.zeros((roi_length + 2 * pixel_margin, roi_length + 2 * pixel_margin,
                                    image_data.shape[2], image_data.shape[3]))
        crop_label_data[(original_r_min - crop_r_min):(original_r_max - crop_r_min + 1), 
                        (original_c_min - crop_c_min):(original_c_max - crop_c_min + 1), 
                        :, 
                        :] = \
            label_data[original_r_min:(original_r_max + 1), 
                       original_c_min:(original_c_max + 1), 
                       :, 
                       :]
        crop_label_data = crop_label_data[::-1, ::-1, :, :]
        crop_label_file = os.path.join(data_dir, str(eid), 'crop_label_sa.nii.gz')
        nib.save(nib.Nifti1Image(crop_label_data, np.eye(4)), crop_label_file)


        # Check the cropped labels to verify the correctness of ROI
        crop_label_data_np = np.array(crop_label_data)
        max_border_value = max(max(crop_label_data_np[0, :, :, :].flatten()), \
            max(crop_label_data_np[-1, :, :, :].flatten()), \
            max(crop_label_data_np[:, 0, :, :].flatten()), \
            max(crop_label_data_np[:, -1, :, :].flatten()))
        if (max_border_value > 0):
            print('max_border_value = {} : problem with ROI'.format(max_border_value) )
            border_problem_eid += [eid]
    

    
        # Save cropped 2D images
        crop_image_data = nib.load(crop_image_file).get_data()

        max_pixel_value = crop_image_data.max()

        if max_pixel_value > 0:
            multiplier = 255.0 / max_pixel_value
        else:
            multiplier = 1.0

        print('max_pixel_value = {}, multiplier = {}'.format(max_pixel_value, multiplier) )

        for s in range(slices):
            for t in range(times):
                s_t_image_file = os.path.join(crop_2D_path, 'crop_2D_{}_{}.png'.format(str(s).zfill(2), str(t).zfill(2)) )
                Image.fromarray((np.rot90(crop_image_data[:, ::-1, s, t], 3) * multiplier).astype('uint8')).save(s_t_image_file)

    
        # Save cropped 2D labels
        crop_label_data = nib.load(crop_label_file).get_data()

        used_instants_label = []
        if (ed_es_instant0 >= 0):
            used_instants_label += [ed_es_instant0]
        if (ed_es_instant1 >= 0):
            used_instants_label += [ed_es_instant1]

        for s in range(slices):
            for t in used_instants_label:
                s_t_label_file = os.path.join(crop_2D_path, 'crop_2D_gt_{}_{}.png'.format(str(s).zfill(2), str(t).zfill(2)) )
                Image.fromarray((np.rot90(crop_label_data[:, ::-1, s, t], 3) * 50).astype('uint8')).save(s_t_label_file)
    

    print(border_problem_eid)

    roi_record.close()

    # Adapt the cropped ground-truth
    adapt_ground_truth(adapt_original=False)  

    print('Done!')



if __name__ == '__main__':
    crop_according_to_roi()




