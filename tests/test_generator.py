from __future__ import absolute_import

import pytest

from hdf5generator.generator import HDF5ImageGenerator
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import to_categorical
from keras.models import Sequential, Model
from keras.layers import Dense, Input, SeparableConv2D, Flatten
from tensorflow.keras.optimizers import RMSprop
from sklearn.preprocessing import LabelEncoder
import numpy as np
import h5py as h5
import cv2

class Resizer(object):
    def __init__(self,
                 width=250,
                 height=250,
                 interp=cv2.INTER_AREA):
        
        self.width = width
        self.height = height
        self.interp = interp

    def preprocess(self, image):
        return cv2.resize(image,
            (self.width, self.height),
            interpolation=self.interp)

def create_generator(num_classes=2, batch_size=16):
    myPreprocessor = Resizer(227, 227)
    
    aug = ImageDataGenerator(
        rotation_range=8,
        zoom_range = 0.2, 
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=False,
        fill_mode='nearest'
    )
    
    gen = HDF5ImageGenerator(
        src= '../../storage/datasets/test.h5',
        num_classes=num_classes,
        batch_size=batch_size,
        augmenter=aug,
        processors=[myPreprocessor]
    )

    return gen
    
def create_sequential_model(num_classes=2, input_shape=(227, 227, 3)):    
    model = Sequential()
    model.add(SeparableConv2D(32,  (5, 5), input_shape=input_shape, padding='same', activation='relu'))
    model.add(Flatten())
    model.add(Dense(32, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))
    
    return model

def test_label_hot_encode():
    classes = ['Cat', 'Dog', 'Panda', 'Bird', 'Zebra']
    y = [np.random.choice(classes) for _ in range(16)]
    le = LabelEncoder()
    scalar_labels = le.fit_transform(y)

    gen = create_generator(num_classes=len(classes))
    binary_matrix = gen.label_hot_encode(scalar_labels)
    
    assert np.array_equal(
        to_categorical(scalar_labels, num_classes=len(classes)),
        binary_matrix)
    'generated binary matrix should be equal to the output of to_categorical'

    
def test_normalize():
    X = np.array([
        [100, 200, 127],
        [75, 225, 127],
        [50, 250, 127],
        [25, 255, 127]
    ])
    test_X = X.astype('float32') / 255.0
        
    gen = create_generator()
    normalized_X = gen.apply_normalize(X)
    
    assert np.array_equal(normalized_X, test_X), 'normalied_X is in the range [0, 1]'
    
def test_len():
    file = h5.File('../../storage/datasets/test.h5', 'r')
    len_file_X = len(file['images'])
    
    batch_size = 32
    num_batches = len_file_X // batch_size
    gen = create_generator(batch_size=batch_size)
    
    assert num_batches == len(gen), 'len() returns the right number of batches'
    
def test_get_next_batch():
    gen = create_generator() # num_batches is 16
    
    (X, y) = gen[np.random.randint(10)]
        
    assert X.shape == (16, 227, 227, 3), 'equals to 16, 227x227x3 images'
    assert y.shape == (16, 2),           'equals to 16 labels (2 classes)'

def tbd_test_generator():
    train_gen   = create_generator()
    val_gen     = create_generator()
    model       = create_sequential_model()
        
    model.compile(
        loss='categorical_crossentropy',
        metrics=['accuracy'],
        optimizer=RMSprop(learning_rate=0.001, rho=0.9))
    
    model.fit_generator(
        train_gen,
        validation_data=val_gen,
        steps_per_epoch=len(train_gen),
        validation_steps=len(val_gen),
        verbose=0,
        epochs=1,
      )
    
    assert True

if __name__ == '__main__':
    pytest.main([__file__])
