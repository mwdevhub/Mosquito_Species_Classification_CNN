import sys
import os
import time
import shutil
from datetime import datetime
import csv
import numpy as np
import random as random


def timestamp():
    dto = datetime.now()
    ts = f'{dto.year}_{dto.month}_{dto.day}_{dto.hour}-{dto.minute}-{dto.second}'
    return ts


def pick_images():

    main_dir = os.getcwd()
    src_dir = os.path.join(main_dir, '01_input_images')
    dst_dir = os.path.join(main_dir, '00_Input_Known_Unknown')
    dst_dir_path = os.path.join(dst_dir, 'K')
    max_count = 554
    num_to_pick = max_count // len(os.listdir(src_dir))

    for sub_dir in os.listdir(src_dir):
        src_sub_dir = os.path.join(src_dir, sub_dir)

        picked = []
        while len(picked) < num_to_pick:

            index_to_pick = random.randrange(0, len(os.listdir(src_sub_dir)))
            if index_to_pick not in picked:
                file_name = os.listdir(src_sub_dir)[index_to_pick]
                picked.append(index_to_pick)
                image_src_path = os.path.join(src_sub_dir, file_name)
                image_dst_path = os.path.join(dst_dir_path, file_name)
                print(image_dst_path)
                with open(image_src_path, 'rb') as fsrc:
                    with open(image_dst_path, 'wb') as fdst:
                        shutil.copyfile(image_src_path, image_dst_path)
    print('IMAGES PICKED')


def create_output_folder(log):
    newpath = log['OUTPUT_PATH']
    if not os.path.exists(newpath):
        os.makedirs(newpath)
    return newpath


def save_log_file(log):
    log['END_TIME'] = timestamp()
    csv_columns = log.keys()
    single_csv_file = os.path.join(log['OUTPUT_PATH'], str(log['NAME'] + '.csv'))

    try:
        with open(single_csv_file, 'w') as f:
            writer = csv.DictWriter(f, fieldnames=csv_columns)
            writer.writeheader()
            writer.writerow(log)
    except IOError:
        print("I/O error")

    global_csv_file = os.path.join(log['OUTPUT_DIRECTORY'], 'global_log.csv')

    try:
        with open(global_csv_file, 'a') as f:
            writer = csv.DictWriter(f, fieldnames=csv_columns)
            writer.writerow(log)
    except IOError:
        print("I/O error")


def import_labels(log, directory='01_input_images'):
    log['CLASS_LABELS'] = [folder_name for folder_name in os.listdir(directory)]
    log['NUMBER_OF_CLASSES'] = len(log['CLASS_LABELS'])
    print(os.listdir(directory))


def print_final_report(log):
    for keys, values in log.items():
        print(keys, ': ', values)


def double_check_data(training_data, validation_data, testing_data):
    print(f'TOTAL NUMBER OF OF TRAINING SAMPLES {len(training_data)}')
    print(f'TOTAL NUMBER OF OF VALIDATION SAMPLES {len(validation_data)}')
    print(f'TOTAL NUMBER OF OF TESTING SAMPLES {len(testing_data)}')
    print('\n')
    training_set = set([i[-1] for i in training_data])
    validation_set = set([i[-1] for i in validation_data])
    testing_set = set([i[-1] for i in testing_data])
    print(f'SAMPLES IN TRAINING AND VALIDATION: {len(training_set.intersection(validation_set))} (SHOULD BE 0)')
    print(f'SAMPLES IN TRAINING AND TESTING: {len(set(training_set).intersection(testing_set))} (SHOULD BE 0)')
    print(f'SAMPLES VALIDATION AND TESTING: {len(validation_set.intersection(testing_set))} (SHOULD BE 0)')
    print('\n')


def save_dataset(log, datasets, kind='training', directory='02_datasets'):
    name = f'{timestamp()}_{kind}_{len(datasets)}.npy'
    path = os.path.join(directory, name)

    datasets = np.array(datasets, dtype="object")
    np.save(path, datasets)

    if kind == 'training':
        log['TRAINING_DATA_USED'] = name
    elif kind == 'validation':
        log['VALIDATION_DATA_USED'] = name
    elif kind == 'testing':
        log['TESTING_DATA_USED'] = name


def load_dataset(name, directory='02_datasets'):
    dataset_path = os.path.join(directory, name)
    try:
        dataset = np.load(dataset_path, allow_pickle=True)
    except:
        print(f'CAN NOT LOAD DATASET NAMED: {name}')
        return None
    print(f'LOADED DATASET: {name}')
    return dataset
