""" Statistics of the UK Biobank cases """

import sys
sys.path.append('..')

import os
import numpy as np
from PIL import Image
from scipy import interpolate
import nibabel as nib

import config



def statistics():

    data_path = config.data_root
    code_path = config.code_root

    statistics_file = os.path.join(code_path, 'Preprocessing', 'statistics_record.txt')
    statistics = open(statistics_file, 'w')

    eids = sorted([x for x in os.listdir(data_path) \
        if os.path.isdir(os.path.join(data_path,x))])

    print('There are {} eids in total'.format(len(eids)))

    useful_eid_count = 0

    # For each case
    for eid in eids:
        print('Processing eid = {}  (# {})'.format(eid, eids.index(eid)) )

        # Define the paths
        eid_path = os.path.join(data_path, eid)
        original_2D_path = os.path.join(eid_path, 'original_2D')

        if not os.path.exists(original_2D_path):
            os.makedirs(original_2D_path)

        sa_zip_file = os.path.join(eid_path, 'sa.nii.gz')
        gt_sa_zip_file = os.path.join(eid_path, 'label_sa.nii.gz')

   
        # Indicators of file existence
        if os.path.isfile(sa_zip_file):
            has_sa = 1
        else:
            has_sa = 0

        if os.path.isfile(gt_sa_zip_file):
            has_gt_sa = 1
        else:
            has_gt_sa = 0

        rows = 0
        columns = 0
        slices = 0
        times = 0

        # ED and ES instants
        ed_es_instant0 = -1
        ed_es_instant1 = -1
        ed_es_instants = []

        # The min/max indices of the slices on which the ground-truth segmentation is present
        ed_es_instant0_min_slice = -1
        ed_es_instant0_max_slice = -1
        ed_es_instant1_min_slice = -1
        ed_es_instant1_max_slice = -1

        spacing_x = -1
        spacing_y = -1
        spacing_z = -1
        spacing_t = -1


        if (has_sa == 1):
            img = nib.load(sa_zip_file)
            (spacing_x, spacing_y, spacing_z, spacing_t) = img.header.get_zooms()
            data = img.get_data()

            rows = data.shape[0]
            columns = data.shape[1]
            slices = data.shape[2]
            times = data.shape[3]

            print('rows = {}, columns = {}, slices = {}, times = {}'.format(rows, columns, slices, times))    


            if (has_gt_sa == 1):
                useful_eid_count += 1

                gt_img = nib.load(gt_sa_zip_file)
                gt_data = gt_img.get_data()
                gt_data_np = np.array(gt_data)

                # The first two instants on which the ground-truth is present are considered
                # as ED and ES respectively. This is the convention followed in UK Biobank.
                for t in range(times):
                    if (gt_data_np[:, :, :, t].max() > 0):
                        ed_es_instants += [t]
                    if (len(ed_es_instants) == 2):
                        break
            
                # There are the exceptional cases identified visually with incomplete
                # ground-truth. Incomplete ground-truth is considered as no ground-truth:

                # For these cases, the ground-truth at ES is incomplete
                if eid in ['1045027', '1081685', '1690388', '1714038', 
                           '2171212', '2809657', '2862965', '2912168', 
                           '3269043', '3448426', '5229454', '5609673', 
                           '5691129']:
                    ed_es_instants = [ed_es_instants[0]]

                # For these cases, the ground-truth at ED is incomplete
                if eid in ['1058604', '3487392']:
                    ed_es_instants = [ed_es_instants[1]]

                # For these cases, the ground-truths at ED and ES are incomplete
                if eid in ['1169331', '2383275', '3043433', '3445606', 
                           '3685624', '4080576', '4139453', '4146720',
                           '4328753', '4408843', '4771786', '4851833',
                           '4911283', '4992385', '5049135', '5354483', 
                           '5682805']:
                    ed_es_instants = []


                if (len(ed_es_instants) > 0):
                    ed_es_instant0 = ed_es_instants[0]
                    for s in range(slices):
                        if (gt_data_np[:, :, s, ed_es_instant0].max() > 0):
                            if (ed_es_instant0_min_slice < 0):
                                ed_es_instant0_min_slice = s
                            ed_es_instant0_max_slice = max(ed_es_instant0_max_slice, s)

                if (len(ed_es_instants) > 1):
                    ed_es_instant1 = ed_es_instants[1]
                    for s in range(slices):
                        if (gt_data_np[:, :, s, ed_es_instant1].max() > 0):
                            if (ed_es_instant1_min_slice < 0):
                                ed_es_instant1_min_slice = s
                            ed_es_instant1_max_slice = max(ed_es_instant1_max_slice, s)
                    
    

        written = '{0} {1} {2} {3} {4} {5} {6} {7} {8} {9} {10} {11} {12} {13} {14} {15} {16}\n'\
            .format(eid, has_sa, has_gt_sa, \
                    rows, columns, slices, \
                    times, ed_es_instant0, ed_es_instant1, \
                    ed_es_instant0_min_slice, ed_es_instant0_max_slice, ed_es_instant1_min_slice,\
                    ed_es_instant1_max_slice, spacing_x, spacing_y, \
                    spacing_z, spacing_t)

        statistics.write(written)



    statistics.close()

    print('useful_eid_count = {}'.format(useful_eid_count) )   

  



if __name__ == '__main__':
    statistics()





