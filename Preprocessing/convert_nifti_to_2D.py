""" Convert the NIfTI images and ground-truth to groups of 2D PNG files """

import sys
sys.path.append('..')

import os
import numpy as np
from PIL import Image
from scipy import interpolate
import nibabel as nib

import config



def convert_nifti_to_2D():

    data_path = config.data_root

    eids = sorted([x for x in os.listdir(data_path) \
        if os.path.isdir(os.path.join(data_path,x))])

    print('There are {} eids in total'.format(len(eids)))

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

        # If the short-axis image file exists, read the data and perform the conversion
        if os.path.isfile(sa_zip_file):
            img = nib.load(sa_zip_file)
            data = img.get_data()
            data_np = np.array(data)

            max_pixel_value = data_np.max()

            if max_pixel_value > 0:
                multiplier = 255.0 / max_pixel_value
            else:
                multiplier = 1.0

            print('max_pixel_value = {},  multiplier = {}'.format(max_pixel_value, multiplier))

            rows = data.shape[0]
            columns = data.shape[1]
            slices = data.shape[2]
            times = data.shape[3]

            for t in range(times):
                for s in range(slices):
                    s_t_image_file = os.path.join(original_2D_path, 'original_2D_{}_{}.png'.format(str(s).zfill(2), str(t).zfill(2)) )
                    Image.fromarray((np.rot90(data[:, ::-1, s, t], 1) * multiplier).astype('uint8')).save(s_t_image_file)

        else:
            print('There is no SA image file for {}'.format(eid))


        # If the ground-truth file exists, read the data and perform the conversion
        if os.path.isfile(gt_sa_zip_file):
            gt_img = nib.load(gt_sa_zip_file)
            gt_data = gt_img.get_data()
            gt_data_np = np.array(gt_data)

            for t in range(times):
                if (gt_data_np[:, :, :, t].max() > 0):
                    for s in range(slices):
                        s_t_image_gt_file = os.path.join(original_2D_path, 'original_gt_2D_{}_{}.png'.format(str(s).zfill(2), str(t).zfill(2)) )
                        # After be multiplied by 50, the pixel values for background, LVC, 
                        # LVM, and RVC are 0, 50, 100 and 150 respectively.
                        Image.fromarray((np.rot90(gt_data[:, ::-1, s, t], 1) * 50).astype('uint8')).save(s_t_image_gt_file)

        else:
            print('There is no SA label image file for {}'.format(eid))
    



if __name__ == '__main__':
    convert_nifti_to_2D()

    




