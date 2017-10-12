'''

Chair Identifier convolutional neural network

'''

from __future__ import print_function
import numpy as np
np.random.seed(1000)

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.utils import np_utils
from keras import backend as K
#import theano
#import sys
#from kers.callbacks import EarlyStopping, TensorBoard

class ChairIdentifier(object):

    def __init__(self, test_mod_num, batch_size, nb_classes, nb_epoch):

        self.batch_size = batch_size
        self.nb_classes = nb_classes
        self.nb_epoch = nb_epoch

        if test_mod_num == 1:
            self.model = self._model_1()
        else:
            self.model = self._model_2()

    def _model_1(self):

        # number of convolutional filters to use
        nb_filters = 32
        # size of pooling area for max pooling
        pool_size = (2, 2)
        # convolution kernel size
        kernel_size = (10, 10)

        test_model = Sequential()
        test_model.add(Conv2D(nb_filters, kernel_size, padding='valid',
                                input_shape=input_shape))
        test_model.add(Activation('relu'))

        test_model.add(MaxPooling2D(pool_size=pool_size))
        test_model.add(Dropout(0.5))

        test_model.add(Conv2D(nb_filters, kernel_size))
        test_model.add(Activation('relu'))

        test_model.add(Conv2D(nb_filters, kernel_size))
        test_model.add(Activation('relu'))

        # transition to an mlp
        test_model.add(Flatten())
        test_model.add(Dense(200))
        test_model.add(Activation('relu'))
        test_model.add(Dense(125))
        test_model.add(Activation('relu'))
        test_model.add(Dense(100))
        test_model.add(Activation('relu'))
        test_model.add(Dropout(0.5))
        test_model.add(Dense(nb_classes))
        test_model.add(Activation('softmax'))

        test_model.compile(loss='categorical_crossentropy',
                      optimizer='adadelta',
                      metrics=['accuracy'])

        return test_model

    def _model_2(self):

        # number of convolutional filters to use
        nb_filters = 32
        # size of pooling area for max pooling
        pool_size = (2, 2)
        # convolution kernel size
        # kernel_size = (10, 10)

        test_model2 = Sequential()
        test_model2.add(Conv2D(nb_filters, (9, 9), padding='valid',
                                input_shape=input_shape))
        test_model2.add(Activation('relu'))

        test_model2.add(MaxPooling2D(pool_size=pool_size))
        test_model2.add(Dropout(0.15))

        test_model2.add(Conv2D(nb_filters, (7,7)))
        test_model2.add(Activation('relu'))
        test_model2.add(Dropout(0.15))

        # test_model2.add(MaxPooling2D(pool_size=pool_size))
        # test_model2.add(Dropout(0.15))
        #
        # test_model2.add(Conv2D(nb_filters, (3,3)))
        # test_model2.add(Activation('relu'))

        # transition to an mlp
        test_model2.add(Flatten())
        test_model2.add(Dense(128))
        test_model2.add(Activation('relu'))
        test_model2.add(Dropout(0.15))
        test_model2.add(Dense(self.nb_classes))
        test_model2.add(Activation('softmax'))

        test_model2.compile(loss='categorical_crossentropy',
                      optimizer='adadelta',
                      metrics=['accuracy'])

        return test_model2

    def fit(self, X_train, y_train, X_test, y_test):

        self.model.fit(X_train, y_train, batch_size = self.batch_size, epochs = self.nb_epoch, \
                  verbose=1, validation_data=(X_test, y_test))

    def score(self, X_test, y_test):

        return self.model.evaluate(X_test, y_test, verbose = 0)

if __name__ == '__main__':

    # input image dimensions
    img_rows, img_cols = 66, 66

    X_train = np.load('X_chair_images_train.npy')
    X_test = np.load('X_chair_images_test.npy')
    y_train = np.load('y_chair_labels_train.npy')
    y_test = np.load('y_chair_labels_test.npy')


    if K.image_dim_ordering() == 'th':
        X_train = X_train.reshape(X_train.shape[0], 1, img_rows, img_cols)
        X_test = X_test.reshape(X_test.shape[0], 1, img_rows, img_cols)
        input_shape = (1, img_rows, img_cols)
    else:
        X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, 1)
        X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, 1)
        input_shape = (img_rows, img_cols, 1)

    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    print('X_train shape:', X_train.shape)
    print(X_train.shape[0], 'train samples')
    print(X_test.shape[0], 'test samples')

    batch_size = 100
    nb_classes = 2
    nb_epoch = 3

    # convert class vectors to binary class matrices
    Y_train = np_utils.to_categorical(y_train, nb_classes)
    Y_test = np_utils.to_categorical(y_test, nb_classes)

    #earlyStopping = EarlySTopping()
    #tbCallBack = TensorBoard

    # model1 = ChairIdentifier(1, batch_size, nb_classes, nb_epoch)
    # model1.fit(X_train, Y_train, X_test, Y_test)
    # score1 = model1.score(X_test, Y_test)
    # print('Test score:', score1[0])
    # print('Test accuracy:', score1[1])

    model2 = ChairIdentifier(2, batch_size, nb_classes, nb_epoch)
    model2.fit(X_train, Y_train, X_test, Y_test)
    score2 = model2.score(X_test, Y_test)
    print('Test score:', score2[0])
    print('Test accuracy:', score2[1])

    #model.save('trained_tables_convnet.h5')
