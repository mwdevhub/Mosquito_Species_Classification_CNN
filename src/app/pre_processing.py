import os
import random
import cv2
import numpy as np
import matplotlib.pyplot as plt


def import_input_data_gray(log, directory='01_input_images'):
    input_data = []
    class_count = {}
    hitmatrix = np.eye(len(log['CLASS_LABELS']))
    num_samples = []
    smallest_sample_num = 0

    for label_index, label in enumerate(log['CLASS_LABELS']):
        path = os.path.join(directory, label)
        num_samples.append(len(os.listdir(path)))

    num_samples.sort()
    smallest_sample_num = num_samples[0]

    for label_index, label in enumerate(log['CLASS_LABELS']):
        path = os.path.join(directory, label)
        sample_class = []
        class_count[label] = 0
        for image_index in range(20):  # range(smallest_sample_num):  # range(20))
            image_name = os.listdir(path)[image_index]
            image_path = os.path.join(path, image_name)
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            sample_class.append([image, hitmatrix[label_index], label, image_name])
            class_count[label] += 1
        input_data.append(sample_class)

    print(class_count)
    print('\n')
    return input_data


def import_input_data_rgb(log, directory='01_input_images'):
    input_data = []
    class_count = {}
    hitmatrix = np.eye(len(log['CLASS_LABELS']))
    num_samples = []
    smallest_sample_num = 0

    for label_index, label in enumerate(log['CLASS_LABELS']):
        path = os.path.join(directory, label)
        num_samples.append(len(os.listdir(path)))

    num_samples.sort()
    smallest_sample_num = num_samples[0]

    for label_index, label in enumerate(log['CLASS_LABELS']):
        path = os.path.join(directory, label)
        sample_class = []
        class_count[label] = 0
        for image_index in range(smallest_sample_num):  # range(20))
            image_name = os.listdir(path)[image_index]
            image_path = os.path.join(path, image_name)
            image = cv2.imread(image_path)
            sample_class.append([image, hitmatrix[label_index], label, image_name])
            class_count[label] += 1
        input_data.append(sample_class)

    print(class_count)
    print('\n')
    return input_data


def split_input_data(log, input_data, training=70, validation=15, testing=15):
    training_data = []
    validation_data = []
    testing_data = []
    print(f'ALL SAMPLES {sum([len(i) for i in input_data])}')

    for sample_class in list(input_data):

        sample_class_copy = sample_class.copy()

        random.shuffle(sample_class_copy)

        train = int(len(sample_class_copy) * (training / 100))
        validate = int(len(sample_class_copy) * (validation / 100))
        test = int(len(sample_class_copy) * (testing / 100))

        training_data += sample_class_copy[:train]
        validation_data += sample_class_copy[train:train + validate]
        testing_data += sample_class_copy[train + validate:]

        log['DATA_DISTRIBUTION_PER_CLASS'].append([sample_class_copy[0][2],
                                                   'train: ', len(sample_class_copy[:train]),
                                                   'validate: ', len(sample_class_copy[train:train + validate]),
                                                   'test: ', len(sample_class_copy[train + validate:])])
    log['PERCENT_FOR_TRAINING'] = training
    log['PERCENT_FOR_VALIDATION'] = validation
    log['PERCENT_FOR_TESTING'] = testing

    log['NUMBER_OF_TRAINING_SAMPLES'] = len(training_data)
    log['NUMBER_OF_VALIDATION_SAMPLES'] = len(validation_data)
    log['NUMBER_OF_TESTING_SAMPLES'] = len(testing_data)

    return training_data, validation_data, testing_data


def prepare_images(log, dataset, cropping_x=300, cropping_y=0, resolution=128):

    for sample in dataset:
        image = sample[0]

        image = image[cropping_x:image.shape[0] - cropping_x, cropping_y:image.shape[1] - cropping_y]
        image = cv2.resize(image, (resolution, resolution))
        sample[0] = image

    log['FINAL_RESOLUTION'] = resolution
    log['CROPPINGX'] = cropping_x
    log['CROPPINGY'] = cropping_y
    return dataset


def rotateImage(image, angle):
    image_center = tuple(np.array(image.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
    result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
    return result


def data_augmentation(log, dataset, num_of_augmentations=10, max_shift=20, max_rotation=15):
    print(f'TOTAL NUMBER OF TRAINING SAMPLES BEFORE AUGMENTATION: {len(dataset)}')
    augmented_dataset = []
    for sample in dataset:
        for i in range(num_of_augmentations):
            image = sample[0]
            M = np.float32([[1, 0, random.randint(max_shift * -1, max_shift)], [0, 1, random.randint(max_shift * -1, max_shift)]])
            aug_image = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]))
            aug_image = rotateImage(aug_image, random.uniform(max_rotation * -1, max_rotation))
            augmented_dataset.append([aug_image, sample[1], sample[2], sample[3][:-4] + f' Aug {i}' + sample[3][-4:]])
    augmented_dataset += dataset
    print(f'TOTAL NUMBER OF TRAINING SAMPLES AFTER AUGMENTATION: {len(augmented_dataset)}')

    log['NUMBER_OF_AUGMENTED_TRAINING_SAMPLES'] = len(augmented_dataset)
    log['NUMBER_OF_AUGMENTATION'] = num_of_augmentations
    log['MAXIMUM_SHIFT'] = max_shift
    log['MAXIMUM_ROTATION'] = max_rotation
    return augmented_dataset


def shuffle_dataset(dataset):
    random.shuffle(dataset)
    return dataset


def show_output_images(image, name):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    plt.title(name)
    plt.imshow(image)
    plt.show()


def prepare_output_image_gray(attribution):
    image = np.array(attribution.cpu().detach()).squeeze()
    image = np.maximum(image, 0)
    image /= np.amax(image)
    return image


def prepare_output_image_rgb(attribution):
    attribution = attribution.permute(0, 2, 3, 1)
    image = np.array(attribution.cpu().detach()).squeeze()
    image = np.maximum(image, 0)
    image /= np.amax(image)
    return image


def save_output_image(log, image, name):
    heatmap = image
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    path = os.path.join(log['OUTPUT_PATH'], name)
    cv2.imwrite(path, heatmap)
