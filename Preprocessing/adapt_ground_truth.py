""" Adapt the UK Biobank ground-truth """

import sys
sys.path.append('..')

import os
import numpy as np
from PIL import Image

import config

# Adapt either the ground-truth either on the original-size files or on the cropped files
def adapt_ground_truth(adapt_original=True):

    data_dir = config.data_root
    code_dir = config.code_root

    statistics_file = os.path.join(code_dir, 'Preprocessing', 'statistics_record.txt')
    doubtful_case_file = os.path.join(code_dir, 'Preprocessing', 'doubtful_segmentation_cases2.txt')

    # Write down the base slice indices only once (when adapting the original-size files) 
    if adapt_original:
        base_slices_file = os.path.join(code_dir, 'Preprocessing', 'base_slices.txt')
        base_slices = open(base_slices_file, 'w')


    # Read the statistics from the statistics file
    with open(statistics_file) as s_file:
        statistics = s_file.readlines()

    statistics = [x.strip() for x in statistics]
    statistics = [[int(z) for z in y.split()[:-4]] + [float(z) for z in y.split()[-4:]] \
        for y in statistics]

    # Read the doubtful cases which will be excluded
    with open(doubtful_case_file) as d_file:
        doubtful_cases = d_file.readlines()

    doubtful_cases = [x.strip() for x in doubtful_cases]
    doubtful_cases = [int(x) for x in doubtful_cases] 


    # We use a case if it's not among the doubtful cases, has both image and ground-truth 
    # files, and has complete ground-truth on both ED and ES.
    used_statistics = [k for k in statistics if (k[0] not in doubtful_cases) and 
        (k[1]==1) and (k[2]==1) and (k[7]>=0) and (k[8]>=0)]

    print('There will be {} used eids'.format(len(used_statistics)) )


    # For each used case
    for k in used_statistics:
        # the statistics
        eid = k[0]
        slices = k[5]
        ed_es_instant0 = k[7]
        ed_es_instant1 = k[8]
        ed_es_instant0_min_slice = k[9]
        ed_es_instant0_max_slice = k[10]
        ed_es_instant1_min_slice = k[11]
        ed_es_instant1_max_slice = k[12]

        print(eid)

        crop_2D_path = os.path.join(data_dir, str(eid), 'crop_2D')
        original_2D_path = os.path.join(data_dir, str(eid), 'original_2D')

        used_instants = []
        if (ed_es_instant0 >= 0):
            used_instants += [ed_es_instant0]
        if (ed_es_instant1 >= 0):
            used_instants += [ed_es_instant1]


        written = str(eid)
        for t in used_instants:
            # Find the base slice and determine whether we should keep LV on this slice 
            base_slice = -1
            keep_base_lv = False
            for s in range(int(round(slices * 0.5 + 0.001)), -1, -1):
                gt_file0 = os.path.join(crop_2D_path, \
                    "crop_2D_gt_{}_{}.png".format(str(s).zfill(2), str(t).zfill(2)) )
                gt_file1 = os.path.join(crop_2D_path, \
                    "crop_2D_gt_{}_{}.png".format(str(s+1).zfill(2), str(t).zfill(2)) )
                
                gt_data0 = np.array(Image.open(gt_file0))
                gt_data1 = np.array(Image.open(gt_file1))

                # Compare the RVs on the two adjacent slices
                rv0 = np.where(gt_data0 == 150, 
                               np.ones_like(gt_data0), np.zeros_like(gt_data0))
                rv1 = np.where(gt_data1 == 150, 
                               np.ones_like(gt_data1), np.zeros_like(gt_data1))

                rv0_rv1_intersection = rv0 * rv1

                good_rv_ratio = (float(np.sum(rv0_rv1_intersection)) / np.sum(rv1) > 0.75) or \
                    (float(np.sum(rv0)) / np.sum(rv1) > 0.8)

                # Presence of LVC and LVM
                has_lvc0 = (50 in gt_data0)
                has_lvm0 = (100 in gt_data0)

                # Check if the LVM mask surrounds the LVC mask
                lvm_surround_lvc = True
                shape_r = gt_data0.shape[0]
                shape_c = gt_data0.shape[1]
                for p in range(shape_r * shape_c):
                     r = p // shape_c
                     c = p % shape_c
                     if (gt_data0[r, c] == 50):
                        up_surrounded = (r != 0) and (gt_data0[r-1, c] in [50, 100])
                        down_surrounded = (r != shape_r-1) and (gt_data0[r+1, c] in [50, 100])
                        left_surrounded = (c != 0) and (gt_data0[r, c-1] in [50, 100])
                        right_surrounded = (c != shape_c-1) and (gt_data0[r, c+1] in [50, 100])

                        if not (up_surrounded and down_surrounded and 
                                left_surrounded and right_surrounded):
                            lvm_surround_lvc = False
                            break

                # Determine the basal slice
                if not (good_rv_ratio and has_lvc0 and has_lvm0 and lvm_surround_lvc):
                    base_slice = s
                    keep_base_lv = True
                    break

            print('base_slice = {}   keep_base_lv = {}'.format(base_slice, keep_base_lv) )


            written += (' {}'.format(base_slice))


            # Adaptation on original-size 2D ground-truth
            if adapt_original:
                for s in range(slices):
                    gt_file0 = os.path.join(original_2D_path, 'original_gt_2D_{}_{}.png'.format(str(s).zfill(2), str(t).zfill(2)) )
                    new_gt_file0 = os.path.join(original_2D_path, 'original_gt2_2D_{}_{}.png'.format(str(s).zfill(2), str(t).zfill(2)) )
                    
                    gt0 = Image.open(gt_file0)
                    c, r = gt0.size
                    if (s < base_slice) or (s == base_slice and (not keep_base_lv)):
                        new_gt_data0 = np.zeros((r, c))
                        Image.fromarray(new_gt_data0.astype('uint8')).save(new_gt_file0)
                    elif (s == base_slice and keep_base_lv):
                        gt_data0 = np.array(gt0)
                        lv0 = np.where(np.logical_or(gt_data0 == 50, gt_data0 == 100), 
                                       np.ones_like(gt_data0), np.zeros_like(gt_data0))
                        new_gt_data0 = lv0 * gt_data0
                        Image.fromarray(new_gt_data0.astype('uint8')).save(new_gt_file0)
                    else:
                        command = 'cp {} {}'.format(gt_file0, new_gt_file0)
                        os.system(command)


            # Adaptation on cropped 2D ground-truth
            else:
                for s in range(slices):
                    gt_file0 = os.path.join(crop_2D_path, "crop_2D_gt_{}_{}.png".format(str(s).zfill(2), str(t).zfill(2)) )
                    new_gt_file0 = os.path.join(crop_2D_path, "crop_2D_gt2_{}_{}.png".format(str(s).zfill(2), str(t).zfill(2)) )
                    
                    gt0 = Image.open(gt_file0)
                    c, r = gt0.size
                    if (s < base_slice) or (s == base_slice and (not keep_base_lv)):
                        new_gt_data0 = np.zeros((r, c))
                        Image.fromarray(new_gt_data0.astype('uint8')).save(new_gt_file0)
                    elif (s == base_slice and keep_base_lv):
                        gt_data0 = np.array(gt0)
                        lv0 = np.where(np.logical_or(gt_data0 == 50, gt_data0 == 100), 
                                       np.ones_like(gt_data0), np.zeros_like(gt_data0))
                        new_gt_data0 = lv0 * gt_data0
                        Image.fromarray(new_gt_data0.astype('uint8')).save(new_gt_file0)
                    else:
                        command = 'cp {} {}'.format(gt_file0, new_gt_file0)
                        os.system(command)

        written += '\n'
        if adapt_original:
            base_slices.write(written)

    
    if adapt_original:
        base_slices.close()



if __name__ == '__main__':
    adapt_ground_truth()











