import os
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import utility as uty
import pre_processing as prep

if torch.cuda.is_available():
    device = torch.device('cuda:0')
    print('DEVICE: GPU\n')
else:
    device = torch.device('cpu')
    print('DEVICE: CPU\n')


def train_cnn_rgb(run, training_data, validation_data):

    log = run.log
    net = run.NET
    epochs = run.EPOCHS
    batch_size = run.BATCH_SIZE
    LEARNING_RATE = run.LEARNING_RATE
    WEIGHT_DECAY = run.WEIGHT_DECAY

    OPTIMZER = run.OPTIMZER
    LOSS_FUNCTION = run.LOSS_FUNCTION

    log['TRAINING_START_TIME'] = uty.timestamp()

    print(uty.timestamp())

    r = 6

    size_x, size_y = np.shape(training_data[0][0])[0], np.shape(training_data[0][0])[1]

    train_x = np.array([i[0] for i in training_data])
    train_x = torch.Tensor(train_x)
    train_x = train_x / 255.0
    train_x = train_x.permute(0, 3, 1, 2)

    train_y = np.array([i[1] for i in training_data])
    train_y = torch.Tensor(train_y)

    validate_x = np.array([i[0] for i in validation_data])
    validate_x = torch.Tensor(validate_x)
    validate_x = validate_x / 255.0
    validate_x = validate_x.permute(0, 3, 1, 2)

    validate_y = np.array([i[1] for i in validation_data])
    validate_y = torch.Tensor(validate_y)

    epoch_count = 0

    print(epochs)
    for epoch in range(epochs):
        print(f'EPOCH {epoch} STARTED AT {uty.timestamp()}')

        for i in range(0, len(train_x), batch_size):
            batch_x = train_x[i:i + batch_size].view(-1, 3, size_x, size_y).to(device)
            batch_y = train_y[i:i + batch_size].to(device)
            acc, loss, outputs = forward_pass_training(run, batch_x, batch_y)

        if epoch % 1 == 0:
            print(f'Loss training {round(float(loss), r)} Accuracy training {round(float(acc), r)} ')
            log['TRAINING_LOSS_AFTER_EPOCH'].append(round(float(loss), r))
            log['TRAINING_ACCURACCY_AFTER_EPOCH'].append(round(float(acc), r))

        if epoch % 1 == 0:
            validate_x = validate_x.view(-1, 3, size_x, size_y).to(device)
            validate_y = validate_y.to(device)
            acc, loss, outputs = forward_pass_validation(run, validate_x, validate_y)

            print(f'Loss validation {round(float(loss), r)} Accuracy validation {round(float(acc), r)} ')
            log['VALIDATION_LOSS_AFTER_EPOCH'].append(round(float(loss), r))
            log['VALIDATION_ACCURACCY_AFTER_EPOCH'].append(round(float(acc), r))

        epoch_count += 1

    log['TRAINING_END_TIME'] = uty.timestamp()

    name = f'{uty.timestamp()}_{run.NET.__class__.__name__}_{OPTIMZER.__class__.__name__}_{LOSS_FUNCTION.__class__.__name__}_e{epochs}_b{batch_size}_lr{LEARNING_RATE}.pt'
    path = os.path.join(log['MODEL_DIRECTROY'], name)

    torch.save({'state_dict': net.state_dict()}, path)

    log['MODEL_NAME'] = name
    log['OPTIMZER'] = OPTIMZER.__class__.__name__
    log['LOSS_FUNCTION'] = LOSS_FUNCTION.__class__.__name__

    log['EPOCHS'] = epochs
    log['BATCH_SIZE'] = batch_size
    log['LEARNING_RATE'] = LEARNING_RATE
    log['WEIGHT_DECAY'] = WEIGHT_DECAY


def train_cnn_gray(run, training_data, validation_data):

    log = run.log
    net = run.NET
    epochs = run.EPOCHS
    batch_size = run.BATCH_SIZE
    LEARNING_RATE = run.LEARNING_RATE

    OPTIMZER = run.OPTIMZER
    LOSS_FUNCTION = run.LOSS_FUNCTION

    log['TRAINING_START_TIME'] = uty.timestamp()

    r = 6

    size_x, size_y = np.shape(training_data[0][0])[0], np.shape(training_data[0][0])[1]

    train_x = np.array([i[0] for i in training_data])
    train_x = torch.Tensor(train_x).view(-1, size_x, size_y)
    train_x = train_x / 255.0
    train_y = np.array([i[1] for i in training_data])
    train_y = torch.Tensor(train_y)

    validate_x = np.array([i[0] for i in validation_data])
    validate_x = torch.Tensor(validate_x).view(-1, size_x, size_y)
    validate_x = validate_x / 255.0
    validate_y = np.array([i[1] for i in validation_data])
    validate_y = torch.Tensor(validate_y)

    epoch_count = 0

    print(epochs)
    for epoch in range(epochs):
        print(f'EPOCH {epoch} STARTED')

        for i in range(0, len(train_x), batch_size):

            batch_x = train_x[i:i + batch_size].view(-1, 1, size_x, size_y).to(device)
            batch_y = train_y[i:i + batch_size].to(device)
            acc, loss, outputs = forward_pass_training(run, batch_x, batch_y)

        if epoch % 1 == 0:
            print(f'LOSS TRAINING {round(float(loss), r)} ACCURACY TRAINING {round(float(acc), r)} ')
            log['TRAINING_LOSS_AFTER_EPOCH'].append(round(float(loss), r))
            log['TRAINING_ACCURACCY_AFTER_EPOCH'].append(round(float(acc), r))

        if epoch % 1 == 0:
            validate_x = validate_x.view(-1, 1, size_x, size_y).to(device)
            validate_y = validate_y.to(device)
            acc, loss, outputs = forward_pass_validation(run, validate_x, validate_y)

            print(f'LOSS VALIDATION {round(float(loss), r)} ACCURACY VALIDATION {round(float(acc), r)} ')
            log['VALIDATION_LOSS_AFTER_EPOCH'].append(round(float(loss), r))
            log['VALIDATION_ACCURACCY_AFTER_EPOCH'].append(round(float(acc), r))

        epoch_count += 1

    log['TRAINING_END_TIME'] = uty.timestamp()

    name = f'{uty.timestamp()}_{run.NET.__class__.__name__}_{OPTIMZER.__class__.__name__}_{LOSS_FUNCTION.__class__.__name__}_e{epochs}_b{batch_size}_lr{LEARNING_RATE}.pt'
    path = os.path.join(log['MODEL_DIRECTROY'], name)

    torch.save({'state_dict': net.state_dict()}, path)

    log['MODEL_NAME'] = name
    log['OPTIMZER'] = OPTIMZER.__class__.__name__
    log['LOSS_FUNCTION'] = LOSS_FUNCTION.__class__.__name__

    log['EPOCHS'] = epochs
    log['BATCH_SIZE'] = batch_size
    log['LEARNING_RATE'] = LEARNING_RATE


def forward_pass_training(run, batch_x, batch_y):
    run.NET.zero_grad()

    outputs = run.NET(batch_x)
    loss = run.LOSS_FUNCTION(outputs, batch_y)
    loss.backward()
    run.OPTIMZER.step()

    matches = [torch.argmax(i) == torch.argmax(j) for i, j in zip(outputs, batch_y)]
    acc = matches.count(True) / len(matches)

    return acc, loss, outputs


def forward_pass_validation(run, validate_x, validate_y):
    with torch.no_grad():
        outputs = run.NET(validate_x)
        loss = run.LOSS_FUNCTION(outputs, validate_y)
        matches = [torch.argmax(i) == torch.argmax(j) for i, j in zip(outputs, validate_y)]
        acc = matches.count(True) / len(matches)

    return acc, loss, outputs
