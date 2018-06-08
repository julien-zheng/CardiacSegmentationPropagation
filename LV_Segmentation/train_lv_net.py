""" The main file to launch the training of LV-net """

import sys
sys.path.append('..')

import os
import copy
import math
import numpy as np
from itertools import izip
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
    save_layer_output,
    print_model_weights_gradients,
    mean_variance_normalization5,
    elementwise_multiplication2
)

from image2 import ImageDataGenerator2

from data_lv_train import ukbiobank_data

from module_lv_net import net_module

import config



def train_lv_net():

    code_path = config.code_root

    initial_lr = config.lv_net_initial_lr
    decay_rate = config.lv_net_decay_rate
    batch_size = config.lv_net_batch_size
    input_img_size = config.lv_net_imput_img_size
    epochs = config.lv_net_epochs

    current_epoch = 0
    new_start_epoch = current_epoch

    ###########
    # The model
    model = net_module(input_shape=(input_img_size, input_img_size, 1), num_outputs=3)
    # Train from scratch
    if current_epoch == 0:
        print('Building model')
    # Finetune
    else:
        print('Loading model')
        model.load_weights(filepath=os.path.join(code_path, 'LV_Segmentation', 'model_lv_net_epoch{}.h5'.format(str(current_epoch).zfill(3))) )

    model.compile(optimizer=Adam(lr=initial_lr), loss=dice_coef6_loss, 
        metrics=[dice_coef6, dice_coef5_0, dice_coef5_1, dice_coef5_2])
    #plot_model(model, to_file='model_lv_net.png', show_shapes=True, show_layer_names=True)

    print('This model has {} parameters'.format(model.count_params()) )

    # Load data lists
    train_img_list0, train_img_list1, train_gt_list0, train_gt_list1, \
    test_img_list0, test_img_list1, test_gt_list0, test_gt_list1 = ukbiobank_data()

    training_sample = len(train_img_list0)

    # we create two instances with the same arguments for random transformation
    img_data_gen_args = dict(featurewise_center=False, 
                    samplewise_center=False,
                    featurewise_std_normalization=False, 
                    samplewise_std_normalization=False,
                    zca_whitening=False, 
                    zca_epsilon=1e-6,
                    rotation_range=180.,
                    width_shift_range=0.15, 
                    height_shift_range=0.15,
                    shear_range=0., 
                    zoom_range=0.15,
                    channel_shift_range=0.,
                    fill_mode='constant', 
                    cval=0.,
                    horizontal_flip=True, 
                    vertical_flip=True,
                    rescale=None, 
                    preprocessing_function=mean_variance_normalization5,
                    data_format=K.image_data_format())

    # deep copy is necessary
    mask_data_gen_args = copy.deepcopy(img_data_gen_args)
    mask_data_gen_args['preprocessing_function'] = elementwise_multiplication2

    #########################
    # Generators for training
    print('Creating generators for training')
    image_datagen0 = ImageDataGenerator2(**img_data_gen_args)
    image_datagen1 = ImageDataGenerator2(**img_data_gen_args)
    mask_datagen0 = ImageDataGenerator2(**mask_data_gen_args)
    mask_datagen1 = ImageDataGenerator2(**mask_data_gen_args)

    # Provide the same seed and keyword arguments to the fit and flow methods
    seed = 1
    image_datagen0.fit(np.zeros((1,1,1,1)), augment=False, rounds=0, seed=seed)
    image_datagen1.fit(np.zeros((1,1,1,1)), augment=False, rounds=0, seed=seed)
    mask_datagen0.fit(np.zeros((1,1,1,1)), augment=False, rounds=0, seed=seed)
    mask_datagen1.fit(np.zeros((1,1,1,1)), augment=False, rounds=0, seed=seed)

    image_generator0 = image_datagen0.flow_from_path_list(
        path_list=train_img_list0,
        target_size=(input_img_size, input_img_size), 
        pad_to_square=True,
        resize_mode='nearest', 
        histogram_based_preprocessing=False,
        clahe=False,
        color_mode='grayscale',
        class_list=None,
        class_mode=None,
        batch_size=batch_size,
        shuffle=True,
        seed=seed,
        save_to_dir=None,
        save_prefix='',
        save_format='png',
        save_period=500,
        follow_links=False)

    image_generator1 = image_datagen1.flow_from_path_list(
        path_list=train_img_list1,
        target_size=(input_img_size, input_img_size), 
        pad_to_square=True,
        resize_mode='nearest', 
        histogram_based_preprocessing=False,
        clahe=False,
        color_mode='grayscale',
        class_list=None,
        class_mode=None,
        batch_size=batch_size,
        shuffle=True,
        seed=seed,
        save_to_dir=None,
        save_prefix='',
        save_format='png',
        save_period=500,
        follow_links=False)

    mask_generator0 = mask_datagen0.flow_from_path_list(
        path_list=train_gt_list0,
        target_size=(input_img_size, input_img_size), 
        pad_to_square=True,
        resize_mode='nearest', 
        histogram_based_preprocessing=False,
        clahe=False,
        color_mode='grayscale',
        class_list=None,
        class_mode=None,
        batch_size=batch_size,
        shuffle=True,
        seed=seed,
        save_to_dir=None,
        save_prefix='',
        save_format='png',
        save_period=500,
        follow_links=False)

    mask_generator1 = mask_datagen1.flow_from_path_list(
        path_list=train_gt_list1,
        target_size=(input_img_size, input_img_size), 
        pad_to_square=True,
        resize_mode='nearest', 
        histogram_based_preprocessing=False,
        clahe=False,
        color_mode='grayscale',
        class_list=None,
        class_mode=None,
        batch_size=batch_size,
        shuffle=True,
        seed=seed,
        save_to_dir=None,
        save_prefix='',
        save_format='png',
        save_period=500,
        follow_links=False)

    # Combine generators into one which yields image and masks
    train_generator = izip(image_generator0, image_generator1, 
                           mask_generator0, mask_generator1)

    
    ###########################
    # Generators for validation
    print('Creating generators for validation')
    val_image_datagen0 = ImageDataGenerator2(**img_data_gen_args)
    val_image_datagen1 = ImageDataGenerator2(**img_data_gen_args)
    val_mask_datagen0 = ImageDataGenerator2(**mask_data_gen_args)
    val_mask_datagen1 = ImageDataGenerator2(**mask_data_gen_args)

    # Provide the same seed and keyword arguments to the fit and flow methods
    val_seed = 2
    val_image_datagen0.fit(np.zeros((1,1,1,1)), augment=False, rounds=0, seed=val_seed)
    val_image_datagen1.fit(np.zeros((1,1,1,1)), augment=False, rounds=0, seed=val_seed)
    val_mask_datagen0.fit(np.zeros((1,1,1,1)), augment=False, rounds=0, seed=val_seed)
    val_mask_datagen1.fit(np.zeros((1,1,1,1)), augment=False, rounds=0, seed=val_seed)

    val_image_generator0 = val_image_datagen0.flow_from_path_list(
        path_list=test_img_list0,
        target_size=(input_img_size, input_img_size), 
        pad_to_square=True,
        resize_mode='nearest', 
        histogram_based_preprocessing=False,
        clahe=False,
        color_mode='grayscale',
        class_list=None,
        class_mode=None,
        batch_size=batch_size,
        shuffle=True,
        seed=val_seed,
        save_to_dir=None,
        save_prefix='',
        save_format='png',
        save_period=1,
        follow_links=False)

    val_image_generator1 = val_image_datagen1.flow_from_path_list(
        path_list=test_img_list1,
        target_size=(input_img_size, input_img_size), 
        pad_to_square=True,
        resize_mode='nearest', 
        histogram_based_preprocessing=False,
        clahe=False,
        color_mode='grayscale',
        class_list=None,
        class_mode=None,
        batch_size=batch_size,
        shuffle=True,
        seed=val_seed,
        save_to_dir=None,
        save_prefix='',
        save_format='png',
        save_period=1,
        follow_links=False)

    val_mask_generator0 = val_mask_datagen0.flow_from_path_list(
        path_list=test_gt_list0,
        target_size=(input_img_size, input_img_size), 
        pad_to_square=True,
        resize_mode='nearest',
        histogram_based_preprocessing=False,
        clahe=False, 
        color_mode='grayscale',
        class_list=None,
        class_mode=None,
        batch_size=batch_size,
        shuffle=True,
        seed=val_seed,
        save_to_dir=None,
        save_prefix='',
        save_format='png',
        save_period=1,
        follow_links=False)

    val_mask_generator1 = val_mask_datagen1.flow_from_path_list(
        path_list=test_gt_list1,
        target_size=(input_img_size, input_img_size), 
        pad_to_square=True,
        resize_mode='nearest', 
        histogram_based_preprocessing=False,
        clahe=False,
        color_mode='grayscale',
        class_list=None,
        class_mode=None,
        batch_size=batch_size,
        shuffle=True,
        seed=val_seed,
        save_to_dir=None,
        save_prefix='',
        save_format='png',
        save_period=1,
        follow_links=False)


    # Combine generators into one which yields image and masks
    validation_generator = izip(val_image_generator0, val_image_generator1, 
                                val_mask_generator0, val_mask_generator1)


    ###############
    # Train the model
    print('Start training')
    steps = int(math.ceil(float(training_sample) / batch_size))
    print('There will be {} epochs with {} steps in each epoch'.format(epochs, steps) )


    total_step = 0
    for epoch in range(new_start_epoch + 1, new_start_epoch + epochs + 1):
        print('\n\n##########\nEpoch {}\n##########'.format(epoch) )

        for step in range(steps):
            print('\n****** Epoch {} Step {} ******'.format(epoch, step) )
            batch_img0, batch_img1, batch_mask0, batch_mask1 = next(train_generator)
            print(model.train_on_batch([batch_img0, batch_img1, batch_mask0], 
                                       batch_mask1, sample_weight=None, class_weight=None))

            # Save_layer_output and print_model_weights_gradients are useful if we
            # want to monitor the training process
            '''
            # save output
            if (total_step % 500 == 0):
                save_layer_output(model, [batch_img0, batch_img1, batch_mask0],
                                  layer_name='output', 
                                  save_path_prefix='record/output')

            # print weights
            if (total_step % 500 == 0):
                print_model_weights_gradients(model, [batch_img0, batch_img1, batch_mask0],
                                              batch_mask1)
            '''

            # perform test
            if (total_step % 500 == 0):
                val_batch_img0, val_batch_img1, \
                val_batch_mask0, val_batch_mask1 = next(validation_generator)
                print('test:')
                print(model.test_on_batch([val_batch_img0, val_batch_img1, val_batch_mask0], 
                                          val_batch_mask1, sample_weight=None))

            total_step += 1


        # adjust learning rate
        if (epoch % 10 == 0):
            old_lr = float(K.get_value(model.optimizer.lr))
            new_lr = initial_lr * (decay_rate**(epoch//10))
            K.set_value(model.optimizer.lr, new_lr)
            print("learning rate is reset to %.8f" % (new_lr))

        # save the model
        if (epoch % 5 == 0):
            model.save_weights(os.path.join(code_path, 'LV_Segmentation', 'model_lv_net_epoch{}.h5'.format(str(epoch).zfill(3))) )


    print('Training is done!')


if __name__ == '__main__':
    train_lv_net()




