from __future__ import absolute_import

import pytest

from hdf5generator.generator import HDF5ImageGenerator
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import to_categorical
from keras.models import Sequential, Model
from keras.layers import Dense, Input, SeparableConv2D, Flatten
from sklearn.preprocessing import LabelEncoder
import numpy as np
import h5py as h5
import cv2

class Resizer(object):
    def __init__(self,
                 shape=(128, 128),
                 interp=cv2.INTER_AREA):
        
        self.shape = shape
        self.interp = interp

    def preprocess(self, image):
        return cv2.resize(image, self.shape, interpolation=self.interp)
        
def create_generator(
    num_classes=2,
    batch_size=16,
    labels_encoding_mode='smooth',
    smooth_factor=0.1,
    shape=(227, 227)):
    myPreprocessor = Resizer(shape)
    
    myAugmenter = ImageDataGenerator(
        rotation_range=8,
        zoom_range = 0.2, 
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=False,
        fill_mode='nearest')
    
    gen = HDF5ImageGenerator(
        src= '../../storage/datasets/c.h5',
        num_classes=num_classes,
        scaler='std',
        labels_encoding=labels_encoding_mode,
        smooth_factor=smooth_factor,
        batch_size=batch_size,
        augmenter=myAugmenter,
        processors=[myPreprocessor])

    return gen
    
def create_sequential_model(num_classes=2, shape=(227, 227, 3)):    
    model = Sequential()
    model.add(SeparableConv2D(32,  (5, 5), input_shape=shape, padding='same', activation='relu'))
    model.add(Flatten())
    model.add(Dense(32, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))
    
    return model

def labels_smoothing(y, factor=0.1):
    y *= (1 - factor)
    y += (factor / y.shape[1])

    return y

def test_labels_encoding():
    classes = ['Cat', 'Dog', 'Panda', 'Bird', 'Zebra']
    y = [np.random.choice(classes) for _ in range(16)]
    le = LabelEncoder()
    scalar_labels = le.fit_transform(y)

    gen = create_generator(num_classes=len(classes))
    binary_matrix = gen.apply_labels_encoding(scalar_labels)
    smooth_binary_matrix = gen.apply_labels_encoding(scalar_labels, 0.1)
    
    assert np.array_equal(
        to_categorical(scalar_labels, num_classes=len(classes)),
        binary_matrix)
    'generated binary matrix should be equal to the output of to_categorical'
    
    assert np.array_equal(
        labels_smoothing(to_categorical(scalar_labels,num_classes=len(classes)), 0.1),
        smooth_binary_matrix)
    'same with labels smoothing'

    
def test_normalization():
    X = np.array([
        [100, 200, 127],
        [75, 225, 127],
        [50, 250, 127],
        [25, 255, 127]
    ])
    test_X = X.astype('float32') / 255.0
        
    gen = create_generator()
    normalized_X = gen.apply_normalization(X)
    
    assert np.array_equal(normalized_X, test_X), 'normalized_X is in the range [0, 1]'
    
def test_standardization():
    X = np.array([
        [100, 200, 127],
        [75, 225, 127],
        [50, 250, 127],
        [25, 255, 127]
    ])
    
    gen = create_generator()
    std_X = gen.apply_standardization(X)
    
    X  = X.astype('float32') 
    X -= np.mean(X, keepdims=True)
    X /= (np.std(X, keepdims=True) + 1e-6)
    
    assert np.array_equal(std_X, X), 'std is in the range [-1, 1]'
    
def tbd_test_len():
    file = h5.File('../../storage/datasets/test.h5', 'r')
    len_file_X = len(file['images'])
    
    batch_size = 32
    num_batches = len_file_X // batch_size
    gen = create_generator(batch_size=batch_size)
    
    assert num_batches == len(gen), 'len() returns the right number of batches'
    
def test_get_next_batch():
    gen = create_generator(batch_size=32)
    
    (X, y) = gen[np.random.randint(10)]
        
    assert X.shape == (32, 227, 227, 3), 'equals to 32, 227x227x3 images'
    assert y.shape == (32, 2),           'equals to 32 labels (2 classes)'

def test_generator():
    train_gen   = create_generator(batch_size=32, shape=(28, 28))
    val_gen     = create_generator(batch_size=32, shape=(28, 28))
    model       = create_sequential_model(shape=(28, 28, 3))
        
    model.compile(
        loss='categorical_crossentropy',
        metrics=['accuracy'],
        optimizer='rmsprop')
    
    model.fit_generator(
        train_gen,
        validation_data=val_gen,
        steps_per_epoch=len(train_gen),
        validation_steps=len(val_gen),
        workers=10,
        use_multiprocessing=True,
        verbose=1,
        epochs=1,
      )
    
    assert True

if __name__ == '__main__':
    pytest.main([__file__])
