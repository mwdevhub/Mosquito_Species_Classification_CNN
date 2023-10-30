import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import utility as uty
import pre_processing as prep

from captum.attr import IntegratedGradients
from captum.attr import GuidedGradCam
from captum.attr import LayerGradCam

if torch.cuda.is_available():
    device = torch.device('cuda:0')
    print('DEVICE: GPU\n')
else:
    device = torch.device('cpu')
    print('DEVICE: CPU\n')


def testing_cnn_gray(run, testing_data):

    if run.log['MODEL_NAME'] != '':
        path = os.path.join(run.log['MODEL_DIRECTROY'], run.log['MODEL_NAME'])
        state_dict = torch.load(path)['state_dict']
        run.NET.load_state_dict(state_dict)

    size_x, size_y = np.shape(testing_data[0][0])[0], np.shape(testing_data[0][0])[1]

    test_x = np.array([i[0] for i in testing_data])
    test_x = torch.Tensor(test_x)
    test_x = test_x / 255.0
    test_x = test_x.view(-1, size_x, size_y)

    test_y = np.array([i[1] for i in testing_data])
    test_y = torch.Tensor(test_y)

    test_labels = [i[2] for i in testing_data]
    test_image_names = [i[3] for i in testing_data]

    conf_matrix = {}
    for i in range(len(run.log['CLASS_LABELS'])):
        label = run.log['CLASS_LABELS'][i]
        conf_matrix[f'{i}: {label}'] = [0 for i in run.log['CLASS_LABELS']]

    number_of_tests = 0
    correct_prediction = 0
    incorrect_prediction = 0
    with torch.no_grad():
        for i in range(len(test_x)):

            real_class_idx = torch.argmax(test_y[i]).to(device)
            real_class_label = test_labels[i]

            output = run.NET(test_x[i].view(-1, 1, size_x, size_y).to(device))
            predicted_class_idx = torch.argmax(output)

            print(f'PREDICTION: {predicted_class_idx} REAL CLASS: {real_class_idx}')

            conf_matrix[f'{real_class_idx}: {real_class_label}'][predicted_class_idx] += 1

            if real_class_idx == predicted_class_idx:
                correct_prediction += 1
            else:
                incorrect_prediction += 1

            number_of_tests += 1

        total_accuracy = round((correct_prediction / number_of_tests) * 100, 3)
        print(f'CORRECT PREDICTION: {correct_prediction} INCORRECT PREDICTION: {incorrect_prediction}')
        print(f'TOTAL ACCURACY: {total_accuracy}')

    run.log['CONFUISION_MATRIX'] = conf_matrix
    run.log['NUMBER_OF_TESTS'] = number_of_tests

    if correct_prediction != 0 and incorrect_prediction != 0:
        run.log['TEST_ACCURACY'] = str(int(total_accuracy - (total_accuracy % 1))) + ',' + str(total_accuracy % 1)[2:]
    elif incorrect_prediction == 0:
        run.log['TEST_ACCURACY'] = 100
    else:
        run.log['TEST_ACCURACY'] = 0


def testing_cnn_gradcam_gray(run, testing_data):

    if run.log['MODEL_NAME'] != '':
        path = os.path.join(run.log['MODEL_DIRECTROY'], run.log['MODEL_NAME'])
        state_dict = torch.load(path)['state_dict']
        run.NET.load_state_dict(state_dict)

    size_x, size_y = np.shape(testing_data[0][0])[0], np.shape(testing_data[0][0])[1]

    test_x = np.array([i[0] for i in testing_data])
    test_x = torch.Tensor(test_x)
    test_x = test_x / 255.0
    test_x = test_x.view(-1, size_x, size_y)

    test_y = np.array([i[1] for i in testing_data])
    test_y = torch.Tensor(test_y)

    test_labels = [i[2] for i in testing_data]
    test_image_names = [i[3] for i in testing_data]

    conf_matrix = {}
    for i in range(len(run.log['CLASS_LABELS'])):
        label = run.log['CLASS_LABELS'][i]
        conf_matrix[f'{i}: {label}'] = [0 for i in run.log['CLASS_LABELS']]

    number_of_tests = 0
    correct_prediction = 0
    incorrect_prediction = 0

    for i in range(0, len(test_x)):  # range(len(test_x))

        real_class_idx = torch.argmax(test_y[i]).to(device)
        real_class_label = test_labels[i]

        output = run.NET(test_x[i].view(-1, 1, size_x, size_y).to(device))
        predicted_class_idx = torch.argmax(output)

        ggc = GuidedGradCam(run.NET, run.NET.conv4)
        input = test_x[i].view(-1, 1, size_x, size_y).to(device)

        conf_matrix[f'{real_class_idx}: {real_class_label}'][predicted_class_idx] += 1

        print(f'PREDICTION: {predicted_class_idx} REAL CLASS: {real_class_idx}')

        if real_class_idx == predicted_class_idx:
            correct_prediction += 1
        else:
            incorrect_prediction += 1

        input.requires_grad = True

        guided_class_idx = predicted_class_idx
        attribution = ggc.attribute(input, guided_class_idx)
        image = prep.prepare_output_image_gray(attribution)
        image_name = f'GGC_{test_image_names[i][:-4]}_P_{predicted_class_idx}_R_{real_class_idx}_GC_{guided_class_idx}.jpg'
        prep.save_output_image(run.log, image, image_name)

        lgc = LayerGradCam(run.NET, run.NET.conv1)
        attribution = lgc.attribute(input, guided_class_idx)

        image = prep.prepare_output_image_gray(attribution)
        image_name = f'LGC_Conv1_{test_image_names[i][:-4]}_P_{predicted_class_idx}_R_{real_class_idx}_GC_{guided_class_idx}.jpg'
        prep.save_output_image(run.log, image, image_name)

        lgc = LayerGradCam(run.NET, run.NET.conv2)
        attribution = lgc.attribute(input, guided_class_idx)

        image = prep.prepare_output_image_gray(attribution)
        image_name = f'LGC_Conv2_{test_image_names[i][:-4]}_P_{predicted_class_idx}_R_{real_class_idx}_GC_{guided_class_idx}.jpg'
        prep.save_output_image(run.log, image, image_name)

        lgc = LayerGradCam(run.NET, run.NET.conv3)
        attribution = lgc.attribute(input, guided_class_idx)

        image = prep.prepare_output_image_gray(attribution)
        image_name = f'LGC_Conv3_{test_image_names[i][:-4]}_P_{predicted_class_idx}_R_{real_class_idx}_GC_{guided_class_idx}.jpg'
        prep.save_output_image(run.log, image, image_name)

        lgc = LayerGradCam(run.NET, run.NET.conv4)
        attribution = lgc.attribute(input, guided_class_idx)

        image = prep.prepare_output_image_gray(attribution)
        image_name = f'LGC_Conv4_{test_image_names[i][:-4]}_P_{predicted_class_idx}_R_{real_class_idx}_GC_{guided_class_idx}.jpg'
        prep.save_output_image(run.log, image, image_name)

        lgc = LayerGradCam(run.NET, run.NET.conv4)
        attribution = lgc.attribute(input, guided_class_idx)

        #image = prep.prepare_output_image_gray(attribution)
        #image_name = f'LGC_Conv5_{test_image_names[i][:-4]}_P_{predicted_class_idx}_R_{real_class_idx}_GC_{guided_class_idx}.jpg'
        #prep.save_output_image(run.log, image, image_name)

        number_of_tests += 1

    total_accuracy = round((correct_prediction / number_of_tests) * 100, 3)
    print(f'CORRECT PREDICTION: {correct_prediction} INCORRECT PREDICTION: {incorrect_prediction}')
    print(f'TOTAL ACCURACY: {total_accuracy}')

    run.log['CONFUISION_MATRIX'] = conf_matrix
    run.log['NUMBER_OF_TESTS'] = number_of_tests

    if correct_prediction != 0 and incorrect_prediction != 0:
        run.log['TEST_ACCURACY'] = str(int(total_accuracy - (total_accuracy % 1))) + ',' + str(total_accuracy % 1)[2:]
    elif incorrect_prediction == 0:
        run.log['TEST_ACCURACY'] = 100
    else:
        run.log['TEST_ACCURACY'] = 0


def testing_cnn_rgb(run, testing_data):

    number_of_tests = 0
    if run.log['MODEL_NAME'] != '':
        path = os.path.join(run.log['MODEL_DIRECTROY'], run.log['MODEL_NAME'])
        state_dict = torch.load(path)['state_dict']
        run.NET.load_state_dict(state_dict)

    size_x, size_y = np.shape(testing_data[0][0])[0], np.shape(testing_data[0][0])[1]

    test_x = np.array([i[0] for i in testing_data])
    test_x = torch.Tensor(test_x)
    test_x = test_x / 255.0
    test_x = test_x.permute(0, 3, 1, 2)

    test_y = np.array([i[1] for i in testing_data])
    test_y = torch.Tensor(test_y)

    test_labels = [i[2] for i in testing_data]
    test_image_names = [i[3] for i in testing_data]

    conf_matrix = {}
    for i in range(len(run.log['CLASS_LABELS'])):
        label = run.log['CLASS_LABELS'][i]
        conf_matrix[f'{i}: {label}'] = [0 for i in run.log['CLASS_LABELS']]

    correct_prediction = 0
    incorrect_prediction = 0
    with torch.no_grad():
        for i in range(len(test_x)):

            real_class_idx = torch.argmax(test_y[i]).to(device)
            real_class_label = test_labels[i]

            output = run.NET(test_x[i].view(-1, 3, size_x, size_y).to(device))
            predicted_class_idx = torch.argmax(output)

            print(f'PREDICTION: {predicted_class_idx} Real Class: {real_class_idx}')

            conf_matrix[f'{real_class_idx}: {real_class_label}'][predicted_class_idx] += 1

            if real_class_idx == predicted_class_idx:
                correct_prediction += 1
            else:
                incorrect_prediction += 1

            number_of_tests += 1

        total_accuracy = round((correct_prediction / number_of_tests) * 100, 3)
        print(f'CORRECT PREDICTION: {correct_prediction} incorrect prediction: {incorrect_prediction}')
        print(f'TOTAL ACCURACY: {total_accuracy}')

    run.log['CONFUISION_MATRIX'] = conf_matrix
    run.log['NUMBER_OF_TESTS'] = number_of_tests

    if correct_prediction != 0 and incorrect_prediction != 0:
        run.log['TEST_ACCURACY'] = str(int(total_accuracy - (total_accuracy % 1))) + ',' + str(total_accuracy % 1)[2:]
    elif incorrect_prediction == 0:
        run.log['TEST_ACCURACY'] = 100
    else:
        run.log['TEST_ACCURACY'] = 0


def testing_cnn_gradcam_rgb(run, testing_data):

    if run.log['MODEL_NAME'] != '':
        path = os.path.join(run.log['MODEL_DIRECTROY'], run.log['MODEL_NAME'])
        state_dict = torch.load(path)['state_dict']
        run.NET.load_state_dict(state_dict)

    size_x, size_y = np.shape(testing_data[0][0])[0], np.shape(testing_data[0][0])[1]

    test_x = np.array([i[0] for i in testing_data])
    test_x = torch.Tensor(test_x)
    test_x = test_x / 255.0
    test_x = test_x.permute(0, 3, 1, 2)

    test_y = np.array([i[1] for i in testing_data])
    test_y = torch.Tensor(test_y)

    test_labels = [i[2] for i in testing_data]
    test_image_names = [i[3] for i in testing_data]

    conf_matrix = {}
    for i in range(len(run.log['CLASS_LABELS'])):
        label = run.log['CLASS_LABELS'][i]
        conf_matrix[f'{i}: {label}'] = [0 for i in run.log['CLASS_LABELS']]

    number_of_tests = 0
    correct_prediction = 0
    incorrect_prediction = 0

    for i in range(0, len(test_x)):

        real_class_idx = torch.argmax(test_y[i]).to(device)
        real_class_label = test_labels[i]

        #print(test_x[i].view(-1, 3, size_x, size_y).to(device).size())

        output = run.NET(test_x[i].view(-1, 3, size_x, size_y).to(device))

        #print(output.to(device).size())

        predicted_class_idx = torch.argmax(output)

        ggc = GuidedGradCam(run.NET, run.NET.conv4)
        input = test_x[i].view(-1, 3, size_x, size_y).to(device)

        conf_matrix[f'{real_class_idx}: {real_class_label}'][predicted_class_idx] += 1

        print(f'PREDICTION: {predicted_class_idx} REAL CLASS: {real_class_idx}')

        if real_class_idx == predicted_class_idx:
            correct_prediction += 1
        else:
            incorrect_prediction += 1

        input.requires_grad = True

        guided_class_idx = predicted_class_idx
        attribution = ggc.attribute(input, guided_class_idx)
        image = prep.prepare_output_image_rgb(attribution)
        image_name = f'GGC_{test_image_names[i][:-4]}_P_{predicted_class_idx}_R_{real_class_idx}_GC_{guided_class_idx}.jpg'
        prep.save_output_image(run.log, image, image_name)

        lgc = LayerGradCam(run.NET, run.NET.conv1)
        attribution = lgc.attribute(input, guided_class_idx)

        image = prep.prepare_output_image_rgb(attribution)
        image_name = f'LGC_Conv1_{test_image_names[i][:-4]}_P_{predicted_class_idx}_R_{real_class_idx}_GC_{guided_class_idx}.jpg'
        prep.save_output_image(run.log, image, image_name)

        lgc = LayerGradCam(run.NET, run.NET.conv2)
        attribution = lgc.attribute(input, guided_class_idx)

        image = prep.prepare_output_image_rgb(attribution)
        image_name = f'LGC_Conv2_{test_image_names[i][:-4]}_P_{predicted_class_idx}_R_{real_class_idx}_GC_{guided_class_idx}.jpg'
        prep.save_output_image(run.log, image, image_name)

        lgc = LayerGradCam(run.NET, run.NET.conv3)
        attribution = lgc.attribute(input, guided_class_idx)

        image = prep.prepare_output_image_rgb(attribution)
        image_name = f'LGC_Conv3_{test_image_names[i][:-4]}_P_{predicted_class_idx}_R_{real_class_idx}_GC_{guided_class_idx}.jpg'
        prep.save_output_image(run.log, image, image_name)

        lgc = LayerGradCam(run.NET, run.NET.conv4)
        attribution = lgc.attribute(input, guided_class_idx)

        image = prep.prepare_output_image_rgb(attribution)
        image_name = f'LGC_Conv4_{test_image_names[i][:-4]}_P_{predicted_class_idx}_R_{real_class_idx}_GC_{guided_class_idx}.jpg'
        prep.save_output_image(run.log, image, image_name)

        #lgc = LayerGradCam(run.NET, run.NET.conv5)
        #attribution = lgc.attribute(input, guided_class_idx)

        #image = prep.prepare_output_image_rgb(attribution)
        #image_name = f'LGC_Conv5_{test_image_names[i][:-4]}_P_{predicted_class_idx}_R_{real_class_idx}_GC_{guided_class_idx}.jpg'
        #prep.save_output_image(run.log, image, image_name)

        number_of_tests += 1

    total_accuracy = round((correct_prediction / number_of_tests) * 100, 3)
    print(f'CORRECT PREDICTION: {correct_prediction} INCORRECT PREDICTION: {incorrect_prediction}')
    print(f'TOTAL ACCURACY: {total_accuracy}')

    run.log['CONFUISION_MATRIX'] = conf_matrix
    run.log['NUMBER_OF_TESTS'] = number_of_tests

    if correct_prediction != 0 and incorrect_prediction != 0:
        run.log['TEST_ACCURACY'] = str(int(total_accuracy - (total_accuracy % 1))) + ',' + str(total_accuracy % 1)[2:]
    elif incorrect_prediction == 0:
        run.log['TEST_ACCURACY'] = 100
    else:
        run.log['TEST_ACCURACY'] = 0
