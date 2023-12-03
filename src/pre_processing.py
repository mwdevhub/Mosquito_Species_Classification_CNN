import os
import random
import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision

#Regarding reviewer 1 comment: Random Seed
# ToDo: ADD RANDOM SEED

random.seed(14)

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


def zoomImage(image, zoom_factor, resolution):
    rand_zoom = random.uniform(1.0, zoom_factor)
    image = cv2.resize(image, None, fx=rand_zoom, fy=rand_zoom)
    # Get new image center
    image_center = image.shape[0] // 2
    # Get lower and upper boundary to crop image back to final resolution
    lower = image_center - (resolution // 2)
    upper = image_center + (resolution // 2)
    # Apply the cropping to the image
    image = image[lower:upper, lower:upper]   #cv2.resize(image, (resolution, resolution))
    return image


def shiftImage(image, max_shift):
    M = np.float32([[1, 0, random.randint(max_shift * -1, max_shift)], [0, 1, random.randint(max_shift * -1, max_shift)]])
    image = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]))
    return image


def cropImage(image, crop_factor):
    transform = torchvision.transforms.ToTensor()
    image_tensor = transform(image)
    rand_factor = random.uniform(crop_factor * -1, crop_factor) #random.uniform(crop_factor * -1, crop_factor)

    og_x = image_tensor.shape[1]
    og_y = image_tensor.shape[2]
    crop_x = int(og_x - (og_x * rand_factor))
    crop_y = int(og_y - (og_y * rand_factor))

    # Create a transform to crop and apply it
    transform = torchvision.transforms.RandomCrop((crop_x, crop_y), padding=1, pad_if_needed=True, padding_mode='constant')
    cropped_image_tensor = transform(image_tensor)

    # Create a transform to resize back to input
    transform = torchvision.transforms.Resize((og_x, og_y), antialias=True)
    cropped_image_tensor = transform(cropped_image_tensor)
    cropped_image_tensor = cropped_image_tensor.permute(1, 2, 0)

    image = np.array(cropped_image_tensor.cpu().detach()).squeeze()
    image = np.uint8(255 * image)
    #plt.imshow(image)
    #plt.show()
    return image
    

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


def save_output_image_no_heatmap(log, image, name):
    path = os.path.join(log['OUTPUT_PATH'], name)
    cv2.imwrite(path, image)


def import_input_data_new(log, directory='01_input_images', folder='train', flag=cv2.IMREAD_GRAYSCALE):
    input_data = []
    class_count = {}
    hitmatrix = np.eye(len(log['CLASS_LABELS']))
    num_samples = []
    smallest_sample_num = 0

    dir = os.path.join(directory, folder)

    if folder in ['train', 'testing', 'validation']:
        for label_index, label in enumerate(log['CLASS_LABELS']):
            path = os.path.join(dir, label)
            num_samples.append(len(os.listdir(path)))
            num_samples.sort()
            smallest_sample_num = num_samples[0]

        for label_index, label in enumerate(log['CLASS_LABELS']):
            path = os.path.join(dir, label)
            sample_class = []
            class_count[label] = 0
            for image_index in range(smallest_sample_num):# #range(smallest_sample_num): #range(smallest_sample_num):  # range(20))
                image_name = os.listdir(path)[image_index]
                image_path = os.path.join(path, image_name)
                image = cv2.imread(image_path, flag)
                sample_class.append([image, hitmatrix[label_index], label, image_name])
                class_count[label] += 1
                input_data.append([image, hitmatrix[label_index], label, image_name])

    random.shuffle(input_data)
    print(f'FOLDER {folder}: ', class_count)
    log['DATA_DISTRIBUTION_PER_CLASS'][folder].append(class_count)
    print('\n')
    return input_data


def data_augmentation(log, dataset, num_of_augmentations=10, max_zoom=1.0, max_shift=20, max_rotation=15, max_crop=0.5, resolution=128):
    print(f'TOTAL NUMBER OF TRAINING SAMPLES BEFORE AUGMENTATION: {len(dataset)}')
    augmented_dataset = []
    for idx, sample in enumerate(dataset):
        for i in range(num_of_augmentations):
            image = sample[0]
            
            aug_image = zoomImage(image, max_zoom, resolution)
            aug_image = shiftImage(aug_image, max_shift)
            aug_image = cropImage(aug_image, max_crop)
            aug_image = rotateImage(aug_image, random.uniform(max_rotation * -1, max_rotation))
            
            #Save 6 augmented images from the file
            if idx <= 5 and i <= 5:
                image_name = f'{sample[3]}_Aug_{i}_{idx}.jpg'
                save_output_image_no_heatmap(log, aug_image, image_name)
                #plt.imshow(aug_image)
                #plt.show()
            augmented_dataset.append([aug_image, sample[1], sample[2], sample[3][:-4] + f' Aug {i}' + sample[3][-4:]])
    
    augmented_dataset += dataset
    print(f'TOTAL NUMBER OF TRAINING SAMPLES AFTER AUGMENTATION: {len(augmented_dataset)}')

    log['NUMBER_OF_AUGMENTED_TRAINING_SAMPLES'] = len(augmented_dataset)
    log['MAX_ZOOM'] = max_zoom
    log['NUMBER_OF_AUGMENTATION'] = num_of_augmentations
    log['MAXIMUM_SHIFT'] = max_shift
    log['MAXIMUM_ROTATION'] = max_rotation
    log['MAXIMUM_CROP'] = max_crop
    return augmented_dataset