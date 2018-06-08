""" The main file to launch the inference of LV-net """

import sys
sys.path.append('..')

import os
import copy
import numpy as np
from itertools import izip
from scipy.misc import imresize
from PIL import Image as pil_image
import tensorflow as tf

from keras.models import Model
from keras.optimizers import Adam
from keras.utils import plot_model
from keras import backend as K

K.set_image_data_format('channels_last')  # TF dimension ordering in this code

from helpers import (
    dice_coef5_0,
    dice_coef5_1,
    dice_coef5_2,
    dice_coef6,
    dice_coef6_loss,
    mean_variance_normalization5,
    elementwise_multiplication2,
    keep_largest_components,
    touch_length_count,
)

from image2 import (
    array_to_img,
    ImageDataGenerator2
)

from data_seg_predict import ukbiobank_data

from module_lv_net import net_module

import config


def predict_lv_net():

    code_path = config.code_root

    initial_lr = config.lv_net_initial_lr
    input_img_size = config.lv_net_imput_img_size
    epochs = config.lv_net_epochs
    batch_size = 1

    database = 'ukbiobank'

    ###########
    # The model
    model = net_module(input_shape=(input_img_size, input_img_size, 1), num_outputs=3)
    print('Loading model')
    model.load_weights(filepath=os.path.join(code_path, 'LV_Segmentation', 'model_lv_net_epoch{}.h5'.format(str(epochs).zfill(3))) )
    model.compile(optimizer=Adam(lr=initial_lr),loss=dice_coef6_loss, 
        metrics=[dice_coef6, dice_coef5_0, dice_coef5_1, dice_coef5_2])

    print('This model has {} parameters'.format(model.count_params()) )




    # Load data lists
    if (database == 'ukbiobank'):
        train_img_list, train_gt_list, \
        train_first_slice_list, train_end_slice_list, train_base_list, \
        test_img_list, test_gt_list, \
        test_first_slice_list, test_end_slice_list, test_base_list = ukbiobank_data()

        '''
        predict_img_list = train_img_list + test_img_list
        predict_gt_list = train_gt_list + test_gt_list
        predict_first_slice_list = train_first_slice_list + test_first_slice_list
        predict_end_slice_list = train_end_slice_list + test_end_slice_list
        predict_base_list = train_base_list + test_base_list
        '''

        predict_img_list = test_img_list
        predict_gt_list = test_gt_list
        predict_first_slice_list = test_first_slice_list
        predict_end_slice_list = test_end_slice_list
        predict_base_list = test_base_list


    predict_sample = len(predict_img_list)

    # we create two instances with the same arguments for random transformation
    img_data_gen_args = dict(featurewise_center=False, 
                    samplewise_center=False,
                    featurewise_std_normalization=False, 
                    samplewise_std_normalization=False,
                    zca_whitening=False, 
                    zca_epsilon=1e-6,
                    rotation_range=0.,
                    width_shift_range=0., 
                    height_shift_range=0.,
                    shear_range=0., 
                    zoom_range=0.,
                    channel_shift_range=0.,
                    fill_mode='constant', 
                    cval=0.,
                    horizontal_flip= False, 
                    vertical_flip=False,
                    rescale=None, 
                    preprocessing_function=mean_variance_normalization5,
                    data_format=K.image_data_format())

    # deep copy is necessary
    mask_data_gen_args = copy.deepcopy(img_data_gen_args)
    mask_data_gen_args['preprocessing_function'] = elementwise_multiplication2

    #########################
    # Generators for training
    print('Creating generators for prediction')
    image_minus_datagen = ImageDataGenerator2(**img_data_gen_args)
    image_datagen = ImageDataGenerator2(**img_data_gen_args)
    mask_minus_datagen = ImageDataGenerator2(**mask_data_gen_args)

    # Provide the same seed and keyword arguments to the fit and flow methods
    seed = 1
    image_minus_datagen.fit(np.zeros((1,1,1,1)), augment=False, rounds=0, seed=seed)
    image_datagen.fit(np.zeros((1,1,1,1)), augment=False, rounds=0, seed=seed)
    mask_minus_datagen.fit(np.zeros((1,1,1,1)), augment=False, rounds=0, seed=seed)


    print('Start prediction')
    print('There will be {} slice-stacks'.format(predict_sample) )

    
    for i in range(predict_sample):
        print('slice-stack {}'.format(i) )

        # The lists fot the stack
        img = predict_img_list[i]
        gt = predict_gt_list[i]
        gt = img.replace('crop_2D', 'predict_2D', 2)
        if database == 'ukbiobank':
            gt = gt.replace('predict_2D_', 'predict_lv_2D_', 1)
        first_slice = predict_first_slice_list[i]
        end_slice = predict_end_slice_list[i]
        slices = end_slice - first_slice
        base = predict_base_list[i]

        if database == 'ukbiobank':
            img_minus_sub_stack = [img[:-9] + str(x-1).zfill(2) + img[-7:] for x in \
                range(max(base, 0), end_slice)]

            img_sub_stack = [img[:-9] + str(x).zfill(2) + img[-7:] for x in \
                range(max(base, 0), end_slice)]

            gt_minus_sub_stack = [gt[:-9] + str(x-1).zfill(2) + gt[-7:] for x in \
                range(max(base, 0), end_slice)]

            gt_minus_sub_stack[0] = ''

            gt_sub_stack = [gt[:-9] + str(x).zfill(2) + gt[-7:] for x in \
                range(max(base, 0), end_slice)]


        image_minus_generator = image_minus_datagen.flow_from_path_list(
            path_list=img_minus_sub_stack,
            target_size=(input_img_size, input_img_size), 
            pad_to_square=True,
            resize_mode='nearest', 
            histogram_based_preprocessing=False,
            clahe=False,
            color_mode='grayscale',
            class_list=None,
            class_mode=None,
            batch_size=batch_size,
            shuffle=False,
            seed=seed,
            save_to_dir=None,
            save_prefix='',
            save_format='png',
            save_period=500,
            follow_links=False)

        image_generator = image_datagen.flow_from_path_list(
            path_list=img_sub_stack,
            target_size=(input_img_size, input_img_size), 
            pad_to_square=True,
            resize_mode='nearest', 
            histogram_based_preprocessing=False,
            clahe=False,
            color_mode='grayscale',
            class_list=None,
            class_mode=None,
            batch_size=batch_size,
            shuffle=False,
            seed=seed,
            save_to_dir=None,
            save_prefix='',
            save_format='png',
            save_period=500,
            follow_links=False)

        mask_minus_generator = mask_minus_datagen.flow_from_path_list(
            path_list=gt_minus_sub_stack,
            target_size=(input_img_size, input_img_size), 
            pad_to_square=True,
            resize_mode='nearest', 
            histogram_based_preprocessing=False,
            clahe=False,
            color_mode='grayscale',
            class_list=None,
            class_mode=None,
            batch_size=batch_size,
            shuffle=False,
            seed=seed,
            save_to_dir=None,
            save_prefix='',
            save_format='png',
            save_period=500,
            follow_links=False)


        # Combine generators into one which yields image and masks
        predict_generator = izip(image_minus_generator, image_generator, mask_minus_generator)

        
        img_size = pil_image.open(img_sub_stack[0]).size
        size = img_size[0]

        for j in range(len(img_sub_stack)):
            
            img_up, img_down, mask_up = next(predict_generator)
            masks = model.predict([img_up, img_down, mask_up], 
                batch_size=batch_size, verbose=0)

            masks = np.reshape(masks, newshape=(input_img_size, input_img_size, 3))
            masks_resized = np.zeros((size, size, 3))
            for c in range(3):
                masks_resized[:, :, c] = imresize(masks[:, :, c], (size, size), interp='bilinear')
            prediction_resized = np.argmax(masks_resized, axis=-1)
            prediction_resized = np.reshape(prediction_resized, newshape=(size, size, 1))

            # Check whether the prediction is successful
            have_lvc = (1 in prediction_resized)
            have_lvm = (2 in prediction_resized)
            lvc_touch_background_length = touch_length_count(prediction_resized, size, size, 1, 0)
            lvc_touch_lvm_length = touch_length_count(prediction_resized, size, size, 1, 2)
            lvc_touch_rvc_length = touch_length_count(prediction_resized, size, size, 1, 3)

            if database == 'ukbiobank':
                success = have_lvm and \
                    ((lvc_touch_background_length + lvc_touch_rvc_length) <= 0.5 * lvc_touch_lvm_length)

            if not success:
                prediction_resized = 0 * prediction_resized

            

            # save txt file
            prediction_path = gt_sub_stack[j]   
            prediction_txt_path = prediction_path.replace('.png', '.txt', 1)
            np.savetxt(prediction_txt_path, prediction_resized, fmt='%.6f')

            # save image
            prediction_img = array_to_img(prediction_resized * 50.0,
                                          data_format=None, 
                                          scale=False)
            prediction_img.save(prediction_path)
            


    K.clear_session()



if __name__ == '__main__':
    predict_lv_net()




