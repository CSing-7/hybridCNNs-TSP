from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Dense, Conv2DTranspose, Flatten, Dropout
from keras.models import Model
import keras.backend as K
import keras
import numpy as np  
import pandas as pd
import format_io

def ae_cnn(input_length, output_length, epochs=200, batch_size=256):
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

    input_img = Input(shape=input_shape)
    x = keras.layers.ZeroPadding2D(padding=((0,1),(1,1)))(input_img)
    x = Conv2D(256, kernel_size = (3, 3), strides = (1, 1), activation='relu', padding = 'same')(x)
    x = MaxPooling2D(pool_size = (2, 2), padding = 'same')(x)  
    x = Conv2D(512, kernel_size = (3, 3), strides = (1, 1), activation='relu', padding = 'same')(x)
    encoded = MaxPooling2D(pool_size = (2, 2), padding = 'same')(x)
    x = UpSampling2D((2,2))(encoded)
    x = Conv2DTranspose(256, kernel_size = (3, 3), strides = (1, 1), activation = 'relu', padding = 'same')(x)
    x = UpSampling2D((2,2))(x)
    x = Conv2DTranspose(1, kernel_size = (3, 3), strides = (1, 1), activation = 'relu', padding = 'same')(x)
    decoded = keras.layers.Cropping2D(cropping=((0, 1), (1, 1)))(x)
    autoencoder = Model(inputs = input_img, outputs = decoded)

    autoencoder.compile(loss = keras.losses.mean_absolute_error, optimizer = keras.optimizers.Adam(), metrics = [keras.metrics.mean_absolute_percentage_error])
    lrreduce = keras.callbacks.ReduceLROnPlateau(monitor = 'val_loss', factor = 0.5, min_delta = 0, patience = 6, verbose = 0)
    earlystop = keras.callbacks.EarlyStopping(monitor = 'val_loss', min_delta = 0, patience = 15, verbose = 0)
    history = autoencoder.fit(x_train, x_train, batch_size = batch_size, epochs = epochs, verbose = 0, validation_data = (x_validation, x_validation), callbacks = [lrreduce, earlystop])

    flatten = Flatten()(encoded)
    fc = Dense(1024, activation = 'relu' ,kernel_regularizer = keras.regularizers.l2(0.1))(flatten)
    fc = (Dense(512, activation = 'relu' ,kernel_regularizer = keras.regularizers.l2(0.1))(fc)
    out = Dense(35)(fc)
    encoder = Model(inputs = input_img, outputs = out)

    encoder.compile(loss = keras.losses.mean_absolute_error, optimizer = keras.optimizers.Adam(0.5e-4), metrics = [keras.metrics.mean_absolute_percentage_error])
    lrreduce = keras.callbacks.ReduceLROnPlateau(monitor = 'val_loss', factor = 0.5, min_delta = 0, patience = 6, verbose = 0)
    earlystop = keras.callbacks.EarlyStopping(monitor = 'val_loss', min_delta = 0, patience = 15, verbose = 0)
    history = encoder.fit(x_train, y_train, batch_size = batch_size, epochs = epochs, verbose = 0, validation_data = (x_validation, y_validation), callbacks = [lrreduce, earlystop])
    encoder.save('ae-cnn.h5')

if __name__=='__main__':
    ae_cnn(30, 5)