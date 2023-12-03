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

from nets import Net256_Conv4_Fc2_B_RGB_C7, Net256_Conv4_Fc1_B_GRAY_C7

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

    def __init__(self, changes = {'epochs': 3, 'num_aug': 30, 'learning_rate': 0.00015, 'weight_decay': 0.0005, 'batch_size': 100}):
        # Set the parameter for a test run using the "run_test" function. The parameter "changes" can be used to run several test with varying parameters see "run_tests" function.
        # To run one test use "run_tests" with one entry.
        
        self.CREATING_DATASETS = True
        self.TRAINING_MODEL = True
        self.TESTING_MODEL = True
        self.TESTING_MODEL_WITH_GRADCAM = True
        self.COLOR_FLAG = cv2.IMREAD_COLOR #cv2.IMREAD_GRAYSCALE #cv2.IMREAD_COLOR

        self.INPUT_DIRECTORY = '01_input_split'
        self.DATASET_DIRECTORY = '02_datasets'
        self.MODEL_DIRECTROY = '03_trained_models'
        self.OUTPUT_DIRECTORY = '04_output'

        self.PERCENT_FOR_TRAINING = 70
        self.PERCENT_FOR_VALIDATION = 15
        self.PERCENT_FOR_TESTING = 15

        self.CLASS_LABELS = ['Ae albopictus right', 'Ae cinereus', 'Ae communis', 'Ae punctor', 'Ae rusticus', 'Ae sticticus', 'Ae vexans']

        # Use only if "CREATING_DATASETS == False" else the dataset will be overriden.
        self.TRAINING_DATA_USED = '' #'2023_10_26_20-2-12_training_28644.npy' 
        self.VALIDATION_DATA_USED = '' #'2023_10_26_20-4-34_validation_112.npy'
        self.TESTING_DATA_USED = ''

        # Use only if "TRAINING_MODEL == False" else it the model will be overriden with new model.
        self.MODEL_NAME = ''

        self.FINAL_RESOLUTION = 256
        self.CROPPINGX = 620
        self.CROPPINGY = 0

        self.MAX_ZOOM = 2.0
        self.NUMBER_OF_AUGMENTATION = changes['num_aug']
        self.MAXIMUM_ROTATION = 15
        self.MAXIMUM_SHIFT = 20
        self.MAXIMUM_CROP = 0.9

        self.EPOCHS = changes['epochs']
        self.BATCH_SIZE = changes['batch_size']
        self.LEARNING_RATE = changes['learning_rate']
        self.WEIGHT_DECAY = changes['weight_decay']

        self.NET = Net256_Conv4_Fc1_B_RGB_C7() #Net256_Conv4_Fc1_B_GRAY_C7()
        self.NET.to(device)

        print('\n')
        input_names = ['input']
        output_names = ['output']

        #Regarding reviewer 1 comment: L2 regularization WEIGHT_DECAY
        self.OPTIMZER = optim.Adam(self.NET.parameters(), lr=self.LEARNING_RATE, weight_decay=self.WEIGHT_DECAY) 
        self.LOSS_FUNCTION = nn.MSELoss()

        self.log = self.create_log_dict()


    def create_log_dict(self):
        # Creates a log dictionary to track settings, training and testing steps. The log is updated during a run and saved afterwards.
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
                'COLOR FLAG' : self.COLOR_FLAG,

                'PERCENT_FOR_TRAINING': 'k.a.',
                'PERCENT_FOR_VALIDATION': 'k.a.',
                'PERCENT_FOR_TESTING': 'k.a.',

                'TRAINING_DATA_USED': self.TRAINING_DATA_USED,
                'VALIDATION_DATA_USED': self.VALIDATION_DATA_USED,
                'TESTING_DATA_USED': self.TESTING_DATA_USED,
                'NUMBER_OF_CLASSES': len(self.CLASS_LABELS),
                'CLASS_LABELS': self.CLASS_LABELS,
                'DATA_DISTRIBUTION_PER_CLASS': {'train' : [], 'testing' : [], 'validation' : []},

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
                'MAX_ZOOM' : 'k.a.',
                'MAXIMUM_ROTATION': 'k.a.',
                'MAXIMUM_SHIFT': 'k.a.',
                'MAXIMUM_CROP' : 'k.a.',

                'EPOCHS': 'k.a.',
                'BATCH_SIZE': 'k.a.',
                'LEARNING_RATE': 'k.a.',
                'WEIGHT_DECAY': 'k.a.',

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

    
    if run.log['CREATING_DATASETS'] == True:

        print('DATA IMPORT')
 
        testing_data = prep.import_input_data_new(run.log, directory=run.log['INPUT_DIRECTORY'], folder='testing', flag=run.COLOR_FLAG)
        training_data = prep.import_input_data_new(run.log, directory=run.log['INPUT_DIRECTORY'], folder='train', flag=run.COLOR_FLAG)
        validation_data = prep.import_input_data_new(run.log, directory=run.log['INPUT_DIRECTORY'], folder='validation', flag=run.COLOR_FLAG)

        run.log['NUMBER_OF_TESTING_SAMPLES'] = len(testing_data)
        run.log['NUMBER_OF_TRAINING_SAMPLES'] = len(training_data)
        run.log['NUMBER_OF_VALIDATION_SAMPLES'] = len(validation_data)

        run.log['PERCENT_FOR_TESTING'] = len(testing_data) / (len(testing_data) +  len(training_data) + len(validation_data))
        run.log['PERCENT_FOR_TRAINING'] = len(training_data) / (len(testing_data) +  len(training_data) + len(validation_data))
        run.log['PERCENT_FOR_VALIDATION'] = len(validation_data) / (len(testing_data) +  len(training_data) + len(validation_data))
  
        print('DATASET CREATION')

        uty.double_check_data(training_data, validation_data, testing_data)

        print('DATA PREPARATION')
        training_data=prep.prepare_images(run.log,
                                          training_data,
                                          cropping_x=run.CROPPINGX,
                                          cropping_y=run.CROPPINGY,
                                          resolution=run.FINAL_RESOLUTION)

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
                                         max_zoom=run.MAX_ZOOM,
                                         max_shift=run.MAXIMUM_SHIFT,
                                         max_rotation=run.MAXIMUM_ROTATION,
                                         max_crop=run.MAXIMUM_CROP,
                                         resolution=run.FINAL_RESOLUTION)

        training_data=prep.shuffle_dataset(training_data)
        validation_data=prep.shuffle_dataset(validation_data)

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


        if run.COLOR_FLAG == 1:
            train.train_cnn_rgb(run, training_data, validation_data)
        else:
            train.train_cnn_gray(run, training_data, validation_data)

        print('TRAINING COMPLETED \n')


    if run.log['TESTING_MODEL'] == True:

        if run.log['CREATING_DATASETS'] == False:
            testing_data = uty.load_dataset(run.log['TESTING_DATA_USED'], directory=run.log['DATASET_DIRECTORY'])

        print(f"USING DATASET: {run.log['TESTING_DATA_USED']}")

        if run.log['TESTING_MODEL_WITH_GRADCAM'] == True:
            print('START TESTING WITH GRAD CAM')
            if run.COLOR_FLAG == 1:
                test.testing_cnn_gradcam_rgb(run, testing_data)
            else:
                test.testing_cnn_gradcam_gray(run, testing_data)
        else:
            print('START TESTING')
            if run.COLOR_FLAG == 1:
                test.testing_cnn_rgb(run, testing_data)
            else:
                test.testing_cnn_gray(run, testing_data)
            
        print('END TESTING')

    print(list(run.log.keys()), sep=',')
    uty.save_log_file(run.log)



def run_tests():
    # Define the number of runs and parameter that should change here. Each dict in the list represets one test run.
    change_list=[
                 {'epochs': 20, 'num_aug': 30, 'learning_rate': 0.00015, 'weight_decay': 0.0005, 'batch_size': 100},
#                 {'epochs': 20, 'num_aug': 30, 'learning_rate': 0.00015, 'weight_decay': 0.0005, 'batch_size': 100},
#                 {'epochs': 20, 'num_aug': 30, 'learning_rate': 0.00015, 'weight_decay': 0.0005, 'batch_size': 100},
#                 {'epochs': 20, 'num_aug': 30, 'learning_rate': 0.00015, 'weight_decay': 0.0005, 'batch_size': 100},
#                 {'epochs': 20, 'num_aug': 30, 'learning_rate': 0.00015, 'weight_decay': 0.0005, 'batch_size': 100},
#                 {'epochs': 25, 'num_aug': 30, 'learning_rate': 0.00015, 'weight_decay': 0.0005, 'batch_size': 100},
#                 {'epochs': 15, 'num_aug': 20, 'learning_rate': 0.00015, 'weight_decay': 0.0005, 'batch_size': 100},
#                 {'epochs': 20, 'num_aug': 20, 'learning_rate': 0.00015, 'weight_decay': 0.0005, 'batch_size': 100},
#                 {'epochs': 25, 'num_aug': 20, 'learning_rate': 0.00015, 'weight_decay': 0.0005, 'batch_size': 100},
                ]

    for changes in change_list:
        run_test(changes)
        torch.cuda.empty_cache()

if __name__ == '__main__':
    run_tests()
