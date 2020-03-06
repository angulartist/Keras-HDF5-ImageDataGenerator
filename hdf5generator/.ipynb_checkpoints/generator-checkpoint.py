from keras.utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import Sequence
import h5py as h5
import numpy as np

import cv2

class HDF5ImageGenerator(Sequence):
    """Just a simple custom Keras ImageDataGenerator that generates
     batches of tensor images from HDF5 files with (optional) real-time
     data augmentation and preprocessing on-the-fly.
     
    # Arguments
        src: Description...
        num_classes: Description...
        X_key: Description...
        y_key: Description...
        batch_size: Description...
        shuffle: Description...
        normalize: Description...
        hot_encoding: Description...
        augmenter: Description...
        processors: Description...
        
    # Examples
    Example of usage:
    ```python
    # Python example here...
    ```
    """
    def __init__(self,
                 src,
                 num_classes=2,
                 X_key='images',
                 y_key='labels',
                 batch_size=32,
                 shuffle=True,
                 normalize=True,
                 hot_encoding=True,
                 augmenter=None,
                 processors=None):
                
        self.file = h5.File(src, 'r')
        self.X_key = X_key
        self.y_key = y_key
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.hot_encoding = hot_encoding
        self.augmenter = augmenter
        self.processors = processors
        self.shuffle = shuffle
        
        self.indices = np.arange(self.file[self.X_key].shape[0])

    def __len__(self):
        return int(np.floor(self.file[self.X_key].shape[0] / self.batch_size))

    def preprocess(self, batch_X):
        """Takes a batch of image tensors, applies preprocessing
         and returns preprocessed images. A preprocessor is a class 
         that implements a preprocess method.
     
        # Arguments
            batch_X: Batch of image tensors
        
        # Examples
        Example of preprocessor:
        
        ```python
        class SimpleResizer(object):
            def __init__(self, width, height, inter=cv2.INTER_AREA):
                self.width = width
                self.height = height
                self.inter = inter
                
            def preprocess(self, image):
                return cv2.resize(image, (self.width, self.height), interpolation=self.inter)
        ```
        """
        X_processed = []

        for x in batch_X:
            for p in self.processors:
                x = p.preprocess(x)

            X_processed.append(x)

        return np.array(X_processed)

    def hot_encode(self, batch_y):
        """Converts a class vector (integers) to binary class matrix.
         See Keras to_categorical utils function.
         
        # Arguments
            batch_y: Batch of integer labels
            
        # Example
        1 => [1000]
        2 => [0100]
        3 => [0001]
        """
        return to_categorical(batch_y, num_classes=self.num_classes)
    
    def normalize(self, X):
        return X.astype('float32') / 255.0

    def __getitem__(self, index):        
        inds = np.sort(self.indices[index * self.batch_size:(index + 1) * self.batch_size])
    
        batch_X = self.file[self.X_key][inds]
        batch_y = self.file[self.y_key][inds]
        
        if self.hot_encoding:
            batch_y = self.hot_encode(batch_y)
        
        if self.processors is not None:
            batch_X = self.preprocess(batch_X)
                    
        if self.augmenter is not None:
            (batch_X, batch_y) = next(self.augmenter.flow(batch_X, batch_y, batch_size=self.batch_size))
        
        batch_X = self.normalize(batch_X)
        
        return (batch_X, batch_y)
    
    def on_epoch_end(self):
        """Triggered once at the very beginning as well as 
         at the end of each epoch. If the shuffle parameter 
         is set to True, image tensors will be shuffled.
         
         This will eventually makes the model more robust.
        """
        if self.shuffle:
            np.random.shuffle(self.indices)