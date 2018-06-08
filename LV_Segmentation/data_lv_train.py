""" A function to generate the lists of files for LV-net training"""

import sys
sys.path.append('..')

import os

import config

def ukbiobank_data():

    data_dir = config.data_root
    code_dir = config.code_root

    statistics_file = os.path.join(code_dir, 'Preprocessing', 'statistics_record.txt')
    doubtful_case_file = os.path.join(code_dir, 'Preprocessing', 'doubtful_segmentation_cases2.txt')
    base_slices_file = os.path.join(code_dir, 'Preprocessing', 'base_slices.txt')

    # Read the basal slice index information
    with open(base_slices_file) as b_file:
        base_slices = b_file.readlines()

    base_slices = [x.strip() for x in base_slices]
    base_slices = [[int(z) for z in y.split()] for y in base_slices]

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

    # Separation of training and testing sets according to the result of eid modulo 5.
    train_statistics =  [x for x in used_statistics if (x[0] % 5 != 2)]
    test_statistics =  [x for x in used_statistics if (x[0] % 5 == 2)]
    

    train_img_list0 = []
    train_img_list1 = []
    train_gt_list0 = []
    train_gt_list1 = []

    test_img_list0 = []
    test_img_list1 = []
    test_gt_list0 = []
    test_gt_list1 = []


    # Training set
    train_subject_count = 0
    for k in train_statistics:
        eid = k[0]
        slices = k[5]
        ed_es_instant0 = k[7]
        ed_es_instant1 = k[8]
        ed_es_instant0_min_slice = k[9]
        ed_es_instant0_max_slice = k[10]
        ed_es_instant1_min_slice = k[11]
        ed_es_instant1_max_slice = k[12]

        base_slice_list = [x for x in base_slices if x[0] == eid][0][1:]

        train_subject_count +=1

        crop_2D_path = os.path.join(data_dir, str(eid), 'crop_2D')

        used_instants = []
        if (ed_es_instant0 >= 0):
            used_instants += [ed_es_instant0]
        if (ed_es_instant1 >= 0):
            used_instants += [ed_es_instant1]

        for idx, t in enumerate(used_instants):
            base_slice_t = base_slice_list[idx]
            first_slice_idx = max(base_slice_t + 1, 0)
            for s in range(max(base_slice_t, 0), slices):
                s_t_image_file0 = os.path.join(crop_2D_path, 'crop_2D_{}_{}.png'.format(str(s-1).zfill(2), str(t).zfill(2)) )
                s_t_image_file1 = os.path.join(crop_2D_path, 'crop_2D_{}_{}.png'.format(str(s).zfill(2), str(t).zfill(2)) )

                if s != max(base_slice_t, 0):
                    s_t_image_gt_file0 = os.path.join(crop_2D_path, 'crop_2D_gt2_{}_{}.png'.format(str(s-1).zfill(2), str(t).zfill(2)) )
                else:
                    s_t_image_gt_file0 = os.path.join(crop_2D_path, 'crop_2D_gt2_{}_{}.png'.format(str(-1).zfill(2), str(t).zfill(2)) )
                
                s_t_image_gt_file1 = os.path.join(crop_2D_path, 'crop_2D_gt2_{}_{}.png'.format(str(s).zfill(2), str(t).zfill(2)) )

                train_img_list0.append(s_t_image_file0)
                train_img_list1.append(s_t_image_file1)
                train_gt_list0.append(s_t_image_gt_file0)
                train_gt_list1.append(s_t_image_gt_file1)



    # Testing set
    test_subject_count = 0
    for k in test_statistics:
        eid = k[0]
        slices = k[5]
        ed_es_instant0 = k[7]
        ed_es_instant1 = k[8]
        ed_es_instant0_min_slice = k[9]
        ed_es_instant0_max_slice = k[10]
        ed_es_instant1_min_slice = k[11]
        ed_es_instant1_max_slice = k[12]

        base_slice_list = [x for x in base_slices if x[0] == eid][0][1:]

        test_subject_count +=1

        crop_2D_path = os.path.join(data_dir, str(eid), 'crop_2D')

        used_instants = []
        if (ed_es_instant0 >= 0):
            used_instants += [ed_es_instant0]
        if (ed_es_instant1 >= 0):
            used_instants += [ed_es_instant1]

        for idx, t in enumerate(used_instants):
            base_slice_t = base_slice_list[idx]
            first_slice_idx = max(base_slice_t + 1, 0)
            for s in range(max(base_slice_t, 0), slices):
                s_t_image_file0 = os.path.join(crop_2D_path, 'crop_2D_{}_{}.png'.format(str(s-1).zfill(2), str(t).zfill(2)) )
                s_t_image_file1 = os.path.join(crop_2D_path, 'crop_2D_{}_{}.png'.format(str(s).zfill(2), str(t).zfill(2)) )

                if s != max(base_slice_t, 0):
                    s_t_image_gt_file0 = os.path.join(crop_2D_path, 'crop_2D_gt2_{}_{}.png'.format(str(s-1).zfill(2), str(t).zfill(2)) )
                else:
                    s_t_image_gt_file0 = os.path.join(crop_2D_path, 'crop_2D_gt2_{}_{}.png'.format(str(-1).zfill(2), str(t).zfill(2)) )
                
                s_t_image_gt_file1 = os.path.join(crop_2D_path, 'crop_2D_gt2_{}_{}.png'.format(str(s).zfill(2), str(t).zfill(2)) )

                test_img_list0.append(s_t_image_file0)
                test_img_list1.append(s_t_image_file1)
                test_gt_list0.append(s_t_image_gt_file0)
                test_gt_list1.append(s_t_image_gt_file1)



    print('train_subject_count = {}'.format(train_subject_count) )
    print('test_subject_count = {}'.format(test_subject_count) )

    print('train_image_count = {}'.format(len(train_img_list0)) )
    print('test_image_count = {}'.format(len(test_img_list0)) )


    return train_img_list0, train_img_list1, train_gt_list0, train_gt_list1, \
           test_img_list0, test_img_list1, test_gt_list0, test_gt_list1 





