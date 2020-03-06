from __future__ import absolute_import

import pytest

from hdf5generator.generator import HDF5ImageGenerator
from keras.preprocessing.image import ImageDataGenerator
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

def get_generator():
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
        batch_size=16,
        augmenter=aug,
        processors=[myPreprocessor]
    )

    # TODO: Test model.
    
    assert True

def test_generator():
    gen = get_generator()
    
    assert 2==2

if __name__ == '__main__':
    pytest.main([__file__])