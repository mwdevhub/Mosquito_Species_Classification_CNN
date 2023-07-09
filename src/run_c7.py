import sys
import os
import time
import csv
import cv2
import numpy as np
import random as random
# import matplotlib as mpl
import matplotlib.pyplot as plt
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from captum.attr import IntegratedGradients
from captum.attr import GuidedGradCam
from captum.attr import LayerGradCam

from nets import Net256_Conv5_Fc3_B_C7, Net256_Conv5_Fc3_B_RGB_C7

import utility as uty
import pre_processing as prep
import training as train
import testing as test


print('\n')
print(f'PYTHON VERSION: {sys.version}')
print(f'CV2 VERSION: {cv2.__version__}')
print(f'NUMPY VERSION: {cv2.__version__}')
print(f'PYTORCH VERSION: {torch.__version__}')

if torch.cuda.is_available():
    device = torch.device('cuda:0')
    print('DEVICE: GPU\n')
else:
    device = torch.device('cpu')
    print('DEVICE: CPU\n')


class Run():

    def __init__(self, changes):

        self.CREATING_DATASETS = False
        self.TRAINING_MODEL = True
        self.TESTING_MODEL = True
        self.TESTING_MODEL_WITH_GRADCAM = False

        self.INPUT_DIRECTORY = '01_input_images'
        self.DATASET_DIRECTORY = '02_datasets'
        self.MODEL_DIRECTROY = '03_trained_models'
        self.OUTPUT_DIRECTORY = '04_output'

        self.PERCENT_FOR_TRAINING = 80
        self.PERCENT_FOR_VALIDATION = 10
        self.PERCENT_FOR_TESTING = 10

        # Use only if CREATING_DATASETS == False else it will be overriden dataset
        self.TRAINING_DATA_USED = '2023_7_9_12-56-7_training_28644.npy'#2022_2_12_12-8-5_training_297.npy'
        self.VALIDATION_DATA_USED = '2023_7_9_12-56-29_validation_112.npy'#'2022_2_12_12-8-5_validation_3.npy'
        self.TESTING_DATA_USED = '2023_7_9_12-56-29_testing_119.npy' #'2022_5_26_20-51-11_testing_119.npy'

        # Use only if TRAINING_MODEL == False else it will be overriden with new model
        self.MODEL_NAME = '' #'2022_5_26_23-28-9_Net256_Conv5_Fc3_B_RGB_C7_RELU_Adam_MSELoss_e15_b100_lr0.00015.pt'

        self.FINAL_RESOLUTION = 256
        self.CROPPINGX = 300
        self.CROPPINGY = 0

        self.NUMBER_OF_AUGMENTATION = changes['num_aug']
        self.MAXIMUM_ROTATION = 15
        self.MAXIMUM_SHIFT = 20

        self.EPOCHS = changes['epochs']
        self.BATCH_SIZE = changes['batch_size']
        self.LEARNING_RATE = changes['learning_rate']

        self.NET = Net256_Conv5_Fc3_B_RGB_C7() 
        self.NET.to(device)

        print('\n')
        input_names = ['input']
        output_names = ['output']

        self.OPTIMZER = optim.Adam(self.NET.parameters(), lr=self.LEARNING_RATE)
        self.LOSS_FUNCTION = nn.MSELoss()

        self.log = self.create_log_dict()


    def create_log_dict(self):
        timestamp = uty.timestamp()
        log = {
                'NAME': f'{timestamp}_mcc',
                'START_TIME': timestamp,
                'END_TIME': 0,
                'CREATING_DATASETS': self.CREATING_DATASETS,
                'TRAINING_MODEL': self.TRAINING_MODEL,
                'CREATING_DATASETS': self.CREATING_DATASETS,
                'TRAINING_MODEL': self.TRAINING_MODEL,
                'TESTING_MODEL': self.TESTING_MODEL,
                'TESTING_MODEL_WITH_GRADCAM': self.TESTING_MODEL_WITH_GRADCAM,

                'INPUT_DIRECTORY': self.INPUT_DIRECTORY,
                'DATASET_DIRECTORY': self.DATASET_DIRECTORY,
                'MODEL_DIRECTROY': self.MODEL_DIRECTROY,
                'OUTPUT_DIRECTORY': self.OUTPUT_DIRECTORY,
                'OUTPUT_PATH': os.path.join(self.OUTPUT_DIRECTORY, f'{timestamp}_mcc_{self.NET.__class__.__name__}'),

                'PERCENT_FOR_TRAINING': 'k.a.',
                'PERCENT_FOR_VALIDATION': 'k.a.',
                'PERCENT_FOR_TESTING': 'k.a.',

                'TRAINING_DATA_USED': self.TRAINING_DATA_USED,
                'VALIDATION_DATA_USED': self.VALIDATION_DATA_USED,
                'TESTING_DATA_USED': self.TESTING_DATA_USED,
                'NUMBER_OF_CLASSES': 0,
                'CLASS_LABELS': [],
                'DATA_DISTRIBUTION_PER_CLASS': [],

                'NUMBER_OF_TRAINING_SAMPLES': 0,
                'NUMBER_OF_AUGMENTED_TRAINING_SAMPLES': 0,
                'NUMBER_OF_VALIDATION_SAMPLES': 0,
                'NUMBER_OF_TESTING_SAMPLES': 0,

                'TRAINING_ACCURACCY_AFTER_EPOCH': [],
                'TRAINING_LOSS_AFTER_EPOCH': [],

                'VALIDATION_ACCURACCY_AFTER_EPOCH': [],
                'VALIDATION_LOSS_AFTER_EPOCH': [],

                'TRAINING_START_TIME': 0,
                'TRAINING_END_TIME': 0,

                'NET': self.NET.__class__.__name__,
                'ARCHITECTURE': str(self.NET).replace('\n', ' ').replace('\r', ''),
                'OPTIMZER': 'k.a',
                'LOSS_FUNCTION': 'k.a.',

                'MODEL_NAME': self.MODEL_NAME,

                'FINAL_RESOLUTION': 'k.a.',
                'CROPPINGX': 'k.a.',
                'CROPPINGY': 'k.a.',

                'NUMBER_OF_AUGMENTATION': 'k.a.',
                'MAXIMUM_ROTATION': 'k.a.',
                'MAXIMUM_SHIFT': 'k.a.',

                'EPOCHS': 'k.a.',
                'BATCH_SIZE': 'k.a.',
                'LEARNING_RATE': 'k.a.',

                'NUMBER_OF_TESTS': 0,

                'CONFUSION_MATRIX': [],

                'TEST_ACCURACY': 0,
                }
        return log


def run_test(changes):

    print(f'\n\nRUN {changes}')

    run = Run(changes)
    print(run.NET.__class__.__name__)

    uty.create_output_folder(run.log)

    uty.import_labels(run.log, directory=run.log['INPUT_DIRECTORY'])

    if run.log['CREATING_DATASETS'] == True:

        print('DATA IMPORT')
        #input_data = prep.import_input_data_gray(run.log, directory=run.log['INPUT_DIRECTORY'])
        input_data = prep.import_input_data_rgb(run.log, directory=run.log['INPUT_DIRECTORY'])

        #print('SHOW IMPOTED IMAGE')
        #prep.show_output_images(training_data[0][0][0], training_data[0][0][3] + ' : ' + str(training_data[0][0][1]))

        print('DATASET CREATION')
        training_data, validation_data, testing_data = prep.split_input_data(run.log,
                                                                             input_data,
                                                                             training=run.PERCENT_FOR_TRAINING,
                                                                             validation=run.PERCENT_FOR_VALIDATION,
                                                                             testing=run.PERCENT_FOR_TESTING)

        uty.double_check_data(training_data, validation_data, testing_data)

        print('DATA PREPARATION')
        training_data=prep.prepare_images(run.log,
                                          training_data,
                                          cropping_x=run.CROPPINGX,
                                          cropping_y=run.CROPPINGY,
                                          resolution=run.FINAL_RESOLUTION)



        #print('SHOW PREPED IMAGE')
        #prep.show_output_images(training_data[0][0], training_data[0][3] + ' : ' + str(training_data[0][1]))

        validation_data=prep.prepare_images(run.log,
                                          validation_data,
                                          cropping_x=run.CROPPINGX,
                                          cropping_y=run.CROPPINGY,
                                          resolution=run.FINAL_RESOLUTION)

        testing_data=prep.prepare_images(run.log,
                                          testing_data,
                                          cropping_x=run.CROPPINGX,
                                          cropping_y=run.CROPPINGY,
                                          resolution=run.FINAL_RESOLUTION)
        print('DATA AUGMENTATION')
        training_data=prep.data_augmentation(run.log,
                                         training_data,
                                         num_of_augmentations=run.NUMBER_OF_AUGMENTATION,
                                         max_shift=run.MAXIMUM_SHIFT,
                                         max_rotation=run.MAXIMUM_ROTATION)

        training_data=prep.shuffle_dataset(training_data)
        validation_data=prep.shuffle_dataset(validation_data)

        #print('SHOW AUGMENTED IMAGE')
        #prep.show_output_images(training_data[0][0], training_data[0][3] + ' : ' + str(training_data[0][1]))

        print('SAVING DATASETS')
        uty.save_dataset(run.log, training_data, kind='training', directory=run.log['DATASET_DIRECTORY'])
        uty.save_dataset(run.log, validation_data, kind='validation', directory=run.log['DATASET_DIRECTORY'])
        uty.save_dataset(run.log, testing_data, kind='testing', directory=run.log['DATASET_DIRECTORY'])


    if run.log['TRAINING_MODEL'] == True:

        if run.log['CREATING_DATASETS'] == False:
            training_data = uty.load_dataset(run.log['TRAINING_DATA_USED'], directory=run.log['DATASET_DIRECTORY'])
            validation_data = uty.load_dataset(run.log['VALIDATION_DATA_USED'], directory=run.log['DATASET_DIRECTORY'])

        print(f"USING DATASET: {run.log['TRAINING_DATA_USED']}")
        print(f"USING DATASET: {run.log['VALIDATION_DATA_USED']}")
        print('MODEL GETS TRAINED')
        #train.train_cnn_gray(run, training_data, validation_data)
        train.train_cnn_rgb(run, training_data, validation_data)

        print('TRAINING COMPLETED \n')


    if run.log['TESTING_MODEL'] == True:

        if run.log['CREATING_DATASETS'] == False:
            testing_data = uty.load_dataset(run.log['TESTING_DATA_USED'], directory=run.log['DATASET_DIRECTORY'])

        if run.log['TESTING_MODEL_WITH_GRADCAM'] == True:
            print('START TESTING WITH GRAD CAM')
            #test.testing_cnn_gradcam_gray(run, testing_data)
            test.testing_cnn_gradcam_rgb(run, testing_data)
        else:
            print('START TESTING')
            #test.testing_cnn_gray(run, testing_data)
            test.testing_cnn_rgb(run, testing_data)

        print('END TESTING')

    uty.save_log_file(run.log)



def run_tests():

    change_list=[
                 {'epochs': 3, 'num_aug': 30, 'learning_rate': 0.00015, 'batch_size': 100},
                 #{'epochs': 5, 'num_aug': 30, 'learning_rate': 0.0002, 'batch_size': 100},
                 #{'epochs': 5, 'num_aug': 30, 'learning_rate': 0.00025, 'batch_size': 100},
                ]

    for changes in change_list:
        run_test(changes)
        torch.cuda.empty_cache()



if __name__ == '__main__':
    run_tests()