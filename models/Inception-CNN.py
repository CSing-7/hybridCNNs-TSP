import keras
import keras.backend as K
from keras.layers import concatenate, Input, Dense, Conv2D, MaxPool2D, AveragePooling2D, Flatten, BatchNormalization
from keras.models import Model
import numpy as np  
import matplotlib.pyplot as plt
import pandas as pd
import format_io

def inception_cnn(input_length, output_length, epochs=200, batch_size=256):
    loops = 35
    input_num = input_length // 5
    output_num = output_length // 5
    (x_train, y_train), (x_validation, y_validation), (x_test, y_test) = format_io.get_data(input_length, output_length)

    if K.image_data_format() == 'channels_first':
        x_train = x_train.reshape(x_train.shape[0], 1, x_train.shape[1], x_train.shape[2])
        y_train = y_train.reshape(y_train.shape[0], -1)
        x_validation = x_validation.reshape(x_validation.shape[0], 1, x_validation.shape[1], x_validation.shape[2])
        y_validation = y_validation.reshape(y_validation.shape[0], -1)
        x_test = x_test.reshape(x_test.shape[0], 1, x_test.shape[1], x_test.shape[2])
        y_test = y_test.reshape(y_test.shape[0], -1)
        input_shape = (1, loops, input_num)
    else:
        x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], x_train.shape[2], 1)
        y_train = y_train.reshape(y_train.shape[0], -1)
        x_validation = x_validation.reshape(x_validation.shape[0], x_validation.shape[1], x_validation.shape[2], 1)
        y_validation = y_validation.reshape(y_validation.shape[0], -1)
        x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], x_test.shape[2], 1)
        y_test = y_test.reshape(y_test.shape[0], -1)
        input_shape = (loops, input_num, 1)

    inp = Input(input_shape, name = 'input')
    bn = BatchNormalization(name = 'bn1')(inp)
    branch3x3x3x3 = Conv2D(64, kernel_size = (3, 3), activation = 'relu', padding = 'same', strides = (1, 1), name = '1brach1_1')(bn)
    branch3x3x3x3 = Conv2D(64, kernel_size = (3, 3), activation = 'relu', padding = 'same', strides = (1, 1), name = '1branch1_2')(branch3x3x3x3)
    branch3x3 = Conv2D(96, kernel_size = (3, 3), activation = 'relu', padding = 'same', strides = (1, 1), name = '1branch2')(bn)
    branch1x1 = Conv2D(64, kernel_size = (1, 1), activation = 'relu', padding = 'same', strides = (1, 1), name = '1branch3')(bn)
    branchpool = AveragePooling2D(pool_size = (3, 3), padding = 'same', strides = (1, 1), name = '1branch4_1')(bn)
    branchpool = Conv2D(32, kernel_size = (1, 1), activation = 'relu', padding = 'same', strides = (1, 1), name = '1branch4_2')(branchpool)
    x = concatenate([branch3x3x3x3, branch3x3, branch1x1, branchpool], name = 'concatenate1')
    branch1x3x3x1x1x3x3x1 = Conv2D(128, kernel_size = (1, 3), activation = 'relu', padding = 'valid', strides = (1, 1), name = '2brach1_1')(x)
    branch1x3x3x1x1x3x3x1 = Conv2D(128, kernel_size = (3, 1), activation = 'relu', padding = 'valid', strides = (1, 1), name = '2brach1_2')(branch1x3x3x1x1x3x3x1)
    branch1x3x3x1x1x3x3x1 = Conv2D(128, kernel_size = (1, 3), activation = 'relu', padding = 'same', strides = (2, 1), name = '2brach1_3')(branch1x3x3x1x1x3x3x1)
    branch1x3x3x1x1x3x3x1 = Conv2D(128, kernel_size = (3, 1), activation = 'relu', padding = 'same', strides = (1, 1), name = '2brach1_4')(branch1x3x3x1x1x3x3x1)
    branch1x3x3x1 = Conv2D(192, kernel_size = (1, 3), activation = 'relu', padding = 'valid', strides = (1, 1), name = '2branch2_1')(x)
    branch1x3x3x1 = Conv2D(192, kernel_size = (3, 1), activation = 'relu', padding = 'valid', strides = (2, 1), name = '2branch2_2')(branch1x3x3x1)
    branchpool = AveragePooling2D(pool_size = (2, 2), padding = 'same', strides = (1, 1), name = '2branch3_1')(x)
    branchpool = Conv2D(64, kernel_size = (3, 3), activation = 'relu', padding = 'valid', strides = (2, 1), name = '2branch3_2')(branchpool)
    x = concatenate([branch1x3x3x1x1x3x3x1, branch1x3x3x1, branchpool], name = 'concatenate2')  # 第二层
    x = MaxPool2D(pool_size = (2, 2), padding = 'same', name = 'maxpool1')(x)
    x = Flatten(name = 'flatten')(x)
    x = Dense(1024, activation = 'relu', name = 'fc1')(x)
    x = Dense(512, activation = 'relu', name = 'fc2')(x)
    x = Dense(y_train.shape[1], name = 'output')(x)
    model = Model(input = inp, output = x)

    model.compile(loss = keras.losses.mean_absolute_error, optimizer = keras.optimizers.Adam(), metrics = [keras.metrics.mean_absolute_percentage_error])
    lrreduce = keras.callbacks.ReduceLROnPlateau(monitor = 'val_loss', factor = 0.5, min_delta = 0, patience = 6, verbose = 0)
    earlystop = keras.callbacks.EarlyStopping(monitor = 'val_loss', min_delta = 0, patience = 15, verbose = 0)
    history = model.fit(x_train, y_train, batch_size = batch_size, epochs = epochs, verbose = 0, validation_data = (x_validation, y_validation), callbacks = [lrreduce, earlystop])
    model.save('inception-cnn.h5')

if __name__=='__main__':
    inception_cnn(30, 5)
