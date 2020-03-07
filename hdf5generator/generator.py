from __future__ import absolute_import

from keras.utils import Sequence
from keras.utils import to_categorical
import h5py as h5
import numpy as np
import cv2

class HDF5ImageGenerator(Sequence):
    """Just a simple custom Keras ImageDataGenerator that generates
     batches of tensor images from HDF5 files with (optional) real-time
     data augmentation and preprocessing on-the-fly.
     
    # Arguments
        src: <String>
            Path of the hdf5 source file.
        num_classes: <Int>
            Total number of classes.
            Default is 2.
        X_key: <String>
            Key of the h5 file image tensors dataset.
            Default is "images".
        y_key: <String>
            Key of the h5 file labels dataset.
            Default is "labels".
        batch_size: <Int>
            Size of each batch, must be a power of two.
            (16, 32, 64, 128, 256, ...)
            Default is 32.
        shuffle: <Boolean>
            Shuffle images at the end of each epoch.
            Default is True.
        normalize: <Boolean>
            Normalize the pixel intensities
            to the range [0, 1].
            Default is True.
        label_hot_encoding: <Boolean>
            Convert integer labels vector
            to binary class matrix (to_categorical).
            Default is True.
        augmenter: <ImageDataGenerator(object)>
            An ImageDataGenerator to apply
            various image transformations
            (augmented data).
            Default is None.
        processors: <Array>
            List of preprocessor classes that implements
            a preprocess method to apply to each batch
            sample before the final output.
            Default is None.
        
    # Examples
    Example of usage:
    ```python
    # Example of a simple imgge resizer pre-processor.
    class Resizer(object):
        def __init__(self, width, height):
            self.width = width
            self.height = height

        # It must implement a preprocess method.
        def preprocess(self, image):
        return cv2.resize(image,
            (self.width, self.height),
            interpolation=cv2.INTER_AREA)
    
    # Optional: Instanciate preprocessors.
    myPreprocessor = Resizer(227, 227)
    
    # Optional: Declare a data augmenter.
    aug = ImageDataGenerator(
        rotation_range=8,
        zoom_range = 0.2, 
        width_shift_range=0.1,
        height_shift_range=0.1,
        vertical_flip=False,
        fill_mode='nearest'
    )

    # Create the generator.
    train_gen = HDF5ImageGenerator('path/to/my/file.h5',
                      augmenter=aug,
                      processors=[myPreprocessor])
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
                 label_hot_encoding=True,
                 augmenter=None,
                 processors=None):
        
        self.src = src
        self.num_classes = num_classes
        self.X_key = X_key
        self.y_key = y_key
        self.batch_size = batch_size
        self.normalize = normalize
        self.label_hot_encoding = label_hot_encoding
        self.augmenter = augmenter
        self.processors = processors
        self.shuffle = shuffle
        
        self.indices = np.arange(self.get_dataset_shape(self.X_key, 0))
        
    def get_dataset_shape(self, dataset, index):
        """Get a h5py dataset shape.
        
        # Arguments
            dataset: <String>, dataset key.
            index: <Int>, dataset index.
         
        # Returns
            An tuple of array dimensions.
        """
        with h5.File(self.src, 'r') as file:
            return file[dataset].shape[index]
        
    def get_dataset_items(self, dataset, indices):
        """Get a h5py dataset items.
        
        # Arguments
            dataset: <String>, dataset key.
            indices: <List>, list of indices.
         
        # Returns
            An batch of elements.
        """
        with h5.File(self.src, 'r') as file:
            return file[dataset][indices]

    def __len__(self):
        """Denotes the number of batches
         per epoch.
         
        # Returns
            An integer.
        """
        return int(np.floor(self.get_dataset_shape(self.X_key, 0) / self.batch_size))

    def preprocess(self, batch_X):
        """Takes a batch of image tensors, applies preprocessing
         and returns preprocessed images. A preprocessor is a class 
         that implements a preprocess method.
     
        # Arguments
            batch_X: Batch of image tensors.
        
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
        
        # Returns
            A numpy array of preprocessed
            image tensors.
        """
        X_processed = []

        for x in batch_X:
            for p in self.processors:
                x = p.preprocess(x)

            X_processed.append(x)

        return np.array(X_processed)

    @staticmethod
    def apply_labels_smoothing(batch_y, factor):
        """Applies labels smoothing to the original
         labels binary matrix.
         
        # Arguments
            batch_y: Vector (batch) of integer labels.
            factor: Int or Float, smoothing factor.
        
        # Returns
            A binary class matrix.
        """
        batch_y *= (1 - factor)
        batch_y += (factor / batch_y.shape[1])

        return batch_y

    def apply_labels_encoding(self, batch_y, smooth_factor=0):
        """Converts a class vector (integers) to binary class matrix.
         See Keras to_categorical utils function.
         
        # Arguments
            batch_y: Vector (batch) of integer labels.
            smooth_factor: Int or Float
                applies labels smoothing if > 0.
            Default is 0.
            
        # Examples
            Outputs:
            1 => [1000]
            2 => [0100]
            3 => [0001]
        
        # Returns
            A binary class matrix.
        """
        batch_y = to_categorical(batch_y, num_classes=self.num_classes)
        
        if smooth_factor > 0:
            batch_y = self.apply_labels_smoothing(batch_y, factor=smooth_factor)
            
        return batch_y
    
    @staticmethod
    def apply_normalize(X):
        """Normalize the pixel intensities
         to the range [0, 1].
         
        # Arguments
            X: Batch of image tensors to be
            normalized.
        
        # Returns
            A batch of normalized image tensors.
        """
        return X.astype('float32') / 255.0

    def __getitem__(self, index): 
        """Generates one batch of data.
        
        # Arguments
            index: index for the current batch.
            
        # Returns
            A tuple containing a batch of image tensors
            and their associated labels.
        """
        
        # Indices for the current batch.
        inds = np.sort(self.indices[index * self.batch_size: (index + 1) * self.batch_size])

        # Grab corresponding images from the HDF5 source file.
        batch_X = self.get_dataset_items(self.X_key, inds)
        
        # Grab corresponding labels from the HDF5 source file.
        batch_y = self.get_dataset_items(self.y_key, inds)
        
        # Shall we apply labels one hot encoding?
        if self.label_hot_encoding:
            batch_y = self.apply_labels_encoding(batch_y)
        
        # Shall we apply any preprocessor?
        if self.processors is not None:
            batch_X = self.preprocess(batch_X)
         
        # Shall we apply any data augmentation?
        if self.augmenter is not None:
            (batch_X, batch_y) = next(self.augmenter.flow(batch_X, batch_y, batch_size=self.batch_size))
        
        # Shall we normalize to range [0, 1]?
        if self.normalize:
            batch_X = self.apply_normalize(batch_X)
        
        return (batch_X, batch_y)
    
    def on_epoch_end(self):
        """Triggered once at the very beginning as well as 
         at the end of each epoch. If the shuffle parameter 
         is set to True, image tensors will be shuffled.
        """
        if self.shuffle:
            np.random.shuffle(self.indices)