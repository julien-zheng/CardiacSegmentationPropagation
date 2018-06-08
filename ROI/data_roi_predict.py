""" A function to generate the lists of files for ROI-net inference"""

import sys
sys.path.append('..')

import os

import config

def ukbiobank_data():
    
    data_dir = config.data_root
    code_dir = config.code_root

    statistics_file = os.path.join(code_dir, 'Preprocessing', 'statistics_record.txt')
    doubtful_case_file = os.path.join(code_dir, 'Preprocessing', 'doubtful_segmentation_cases2.txt')

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

    train_img_list = []
    train_gt_list = []
    test_img_list = []
    test_gt_list = []


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

        train_subject_count +=1

        original_2D_path = os.path.join(data_dir, str(eid), 'original_2D')

        # Prediction on the ED stacks only
        used_instants = []
        if (ed_es_instant0 >= 0):
            used_instants += [ed_es_instant0]
        
        for idx, t in enumerate(used_instants):
            for s in range(int(round(slices * 0.2 + 0.001)), int(round(slices * 0.6 + 0.001))):
                s_t_image_file = os.path.join(original_2D_path, 'original_2D_{}_{}.png'.format(str(s).zfill(2), str(t).zfill(2)) )
                # The adapted ground-truth
                s_t_image_gt_file = os.path.join(original_2D_path, 'original_gt2_2D_{}_{}.png'.format(str(s).zfill(2), str(t).zfill(2)) )

                train_img_list.append(s_t_image_file)
                train_gt_list.append(s_t_image_gt_file)



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

        test_subject_count +=1

        original_2D_path = os.path.join(data_dir, str(eid), 'original_2D')

        # Prediction on the ED stacks only
        used_instants = []
        if (ed_es_instant0 >= 0):
            used_instants += [ed_es_instant0]

        for idx, t in enumerate(used_instants):
            for s in range(int(round(slices * 0.2 + 0.001)), int(round(slices * 0.6 + 0.001))):
                s_t_image_file = os.path.join(original_2D_path, 'original_2D_{}_{}.png'.format(str(s).zfill(2), str(t).zfill(2)) )
                # The adapted ground-truth
                s_t_image_gt_file = os.path.join(original_2D_path, 'original_gt2_2D_{}_{}.png'.format(str(s).zfill(2), str(t).zfill(2)) )

                test_img_list.append(s_t_image_file)
                test_gt_list.append(s_t_image_gt_file)


    print('train_subject_count = {}'.format(train_subject_count) )
    print('test_subject_count = {}'.format(test_subject_count) )

    print('train_image_count = {}'.format(len(train_img_list)) )
    print('test_image_count = {}'.format(len(test_img_list)) )

    return train_img_list, train_gt_list, test_img_list, test_gt_list





