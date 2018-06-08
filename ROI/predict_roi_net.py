""" The main file to launch the inference of ROI-net """


import os
import numpy as np
import scipy
import math
from PIL import Image as pil_image
import tensorflow as tf

from keras.models import (
    Model,
    load_model
)
from keras.optimizers import Adam
from keras import backend as K

K.set_image_data_format('channels_last')  # TF dimension ordering in this code

from helpers import (
    dice_coef2,
    dice_coef2_loss,
    mean_variance_normalization5
)
from image2 import (
    array_to_img,
    ImageDataGenerator2
)
from data_roi_predict import ukbiobank_data

from module_roi_net import net_module

import config



def predict_roi_net():

    code_path = config.code_root

    initial_lr = config.roi_net_initial_lr
    batch_size = config.roi_net_batch_size
    input_img_size = config.roi_net_imput_img_size

    epochs = config.roi_net_epochs

    ###########
    # The model
    model = net_module(input_shape=(input_img_size, input_img_size, 1), num_outputs=1)

    model.load_weights(filepath=os.path.join(code_path, 'ROI', 'model_roi_net_epoch{}.h5'.format(str(epochs).zfill(3))) )

    model.compile(optimizer=Adam(lr=initial_lr), loss=dice_coef2_loss, 
        metrics=[dice_coef2])

    ######
    # Data
    train_img_list, train_gt_list, test_img_list, test_gt_list = ukbiobank_data()
    predict_img_list = train_img_list + test_img_list
    predict_gt_list = train_gt_list + test_gt_list

    predict_img_list = sorted(predict_img_list)
    predict_gt_list = sorted(predict_gt_list)

    predict_sample = len(predict_img_list)

    # we create two instances with the same arguments
    data_gen_args = dict(featurewise_center=False, 
                    samplewise_center=False,
                    featurewise_std_normalization=False, 
                    samplewise_std_normalization=False,
                    zca_whitening=False, 
                    zca_epsilon=1e-6,
                    rotation_range=0.0,
                    width_shift_range=0.0, 
                    height_shift_range=0.0,
                    shear_range=0., 
                    zoom_range=0.0,
                    channel_shift_range=0.,
                    fill_mode='constant', 
                    cval=0.,
                    horizontal_flip=False, 
                    vertical_flip=False,
                    rescale=None, 
                    preprocessing_function=mean_variance_normalization5,
                    data_format=K.image_data_format())


    ###########################
    # Generators for predicting
    image_datagen = ImageDataGenerator2(**data_gen_args)

    seed = 1
    image_datagen.fit(np.zeros((1,1,1,1)), augment=False, rounds=0, seed=seed)

    image_generator = image_datagen.flow_from_path_list(
            path_list=predict_img_list,
            target_size=(input_img_size, input_img_size), 
            pad_to_square=True,
            resize_mode='nearest', 
            histogram_based_preprocessing=True,
            clahe=True,
            color_mode='grayscale',
            class_list=None,
            class_mode=None,
            batch_size=batch_size,
            shuffle=False,
            seed=seed,
            save_to_dir=None,
            save_prefix='',
            save_format='png',
            save_period=100,
            follow_links=False)


    print('Start prediction')
    print('There will be {} batches with batch-size {}'.format(int(math.ceil(float(predict_sample) / batch_size)), batch_size) )

    for i in range(int(math.ceil(float(predict_sample) / batch_size)) ):
        print('batch {}'.format(i) )
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, predict_sample)
        img_list_batch = predict_img_list[start_idx:end_idx]

        batch_img = next(image_generator) 

        predict_masks = model.predict(batch_img, 
                                      batch_size=batch_size, 
                                      verbose=0)
        binarized_predict_masks = np.where(predict_masks >= 0.5, 1.0, 0.0)

        for j in range(len(img_list_batch)):
            img_path = img_list_batch[j]
            #print(img_path)
            img_size = pil_image.open(img_path).size
            h = img_size[0]
            w = img_size[1]
            size = max(h, w)

            # reshape and crop the predicted mask to the original size
            mask = np.reshape(binarized_predict_masks[j], newshape=(input_img_size, input_img_size))
            resized_mask = scipy.misc.imresize(mask, size=(size, size), interp='nearest')/255.0
            cropped_resized_mask = resized_mask[((size-w)//2):((size-w)//2 + w), 
                                                ((size-h)//2):((size-h)//2 + h)]
            cropped_resized_mask = np.reshape(cropped_resized_mask, newshape=(w, h, 1))

            predicted_mask_path = img_path.replace('original_2D', 'mask_original_2D', 2)

            # save txt file
            predicted_mask_txt_path = predicted_mask_path.replace('.png', '.txt', 1)
            np.savetxt(predicted_mask_txt_path, cropped_resized_mask, fmt='%.6f')

            # save image
            cropped_resized_mask_img = array_to_img(cropped_resized_mask,
                                                    data_format=None, 
                                                    scale=True)
            cropped_resized_mask_img.save(predicted_mask_path)



if __name__ == '__main__':
    predict_roi_net()


