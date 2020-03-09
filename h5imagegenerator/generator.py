from __future__ import absolute_import

import warnings

import h5py as h5

from keras.utils import Sequence
from keras.utils import to_categorical
import numpy as np

class HDF5ImageGenerator(Sequence):
    """Just a simple custom Keras ImageDataGenerator that generates
     batches of tensor images from HDF5 files with (optional) real-time
     data augmentation.
     
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
        scaler: "std", "norm" or False
            "std" mode means standardization to range [-1, 1]
            with 0 mean and unit variance.
            "norm" mode means normalization to range [0, 1].
            Default is "std".
        labels_encoding: "hot", "smooth" or False
            "hot" mode means classic one hot encoding.
            "smooth" mode means smooth hot encoding.
            Default is "hot".
        smooth_factor: <Int> or <Float>
            smooth factor used by smooth
            labels encoding.
            Default is 0.1.
        augmenter: albumentations Compose([]) Pipeline
            An albumentations transformations pipeline
            to apply to each sample.
            (data augmentation).
            Default is None.
        
    # Examples
    Example of usage:
    ```python
    my_augmenter = Compose([
        HorizontalFlip(p=0.5),
        RandomContrast(limit=0.2, p=0.5),
        RandomGamma(gamma_limit=(80, 120), p=0.5),
        RandomBrightness(limit=0.2, p=0.5),
        Resize(227, 227, cv2.INTER_AREA)
    ])

    # Create the generator.
    train_gen = HDF5ImageGenerator(
        'path/to/my/file.h5',
         augmenter=my_augmenter)
    ```
    """
    def __init__(self,
                 src,
                 num_classes=2,
                 X_key='images',
                 y_key='labels',
                 batch_size=32,
                 shuffle=True,
                 scaler=True,
                 labels_encoding='hot',
                 smooth_factor=0.1,
                 augmenter=None):
        
        if labels_encoding not in {'hot', 'smooth', False}:
            raise ValueError(
                '`labels_encoding` should be `"hot"` '
                '(classic binary matrix) or '
                '`"smooth"` (smooth encoding) or '
                'False (no labels encoding).'
                'Received: %s' % labels_encoding)
        self.labels_encoding = labels_encoding
        
        if self.labels_encoding == 'smooth' and not (0 < smooth_factor <= 1):
            raise ValueError('`"smooth"` labels encoding'
                             'must use a `"smooth_factor"`'
                             '< 0 smooth_factor <= 1')
        
        self.src = src
        self.num_classes = num_classes
        self.X_key = X_key
        self.y_key = y_key
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.scaler = scaler
        self.smooth_factor = smooth_factor
        self.augmenter = augmenter
        
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
        return int(np.ceil(self.get_dataset_shape(self.X_key, 0) / float(self.batch_size)))

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
    
    # TODO: Deprecated. 
    @staticmethod
    def apply_standardization(batch_X):
        """Scale the pixel intensities
         to the range [-1, 1], 0 mean and unit variance.
         z = (x - u) / s
         
        # Arguments
            batch_X: Batch of image tensors to be
            standardized.
        
        # Returns
            A batch of standardized image tensors.
        """
        batch_X  = batch_X.astype('float32')
        batch_X -= np.mean(batch_X, keepdims=True)
        batch_X /= (np.std(batch_X, keepdims=True) + 1e-6)
                
        return batch_X
    
    @staticmethod
    def apply_normalization(batch_X):
        """Normalize the pixel intensities
         to the range [0, 1].
         
        # Arguments
            batch_X: Batch of image tensors to be
            normalized.
        
        # Returns
            A batch of normalized image tensors.
        """
        return batch_X.astype('float32') / 255.0

    def __getitem__(self, index): 
        """Generates one batch of data.
        
        # Arguments
            index: index for the current batch.
            
        # Returns
            A tuple containing a batch of image tensors
            and their associated labels.
        """
        
        # Indices for the current batch.
        inds = np.sort(self.indices[index * self.batch_size : (index + 1) * self.batch_size])

        # Grab corresponding images from the HDF5 source file.
        batch_X = self.get_dataset_items(self.X_key, inds)
        
        # Grab corresponding labels from the HDF5 source file.
        batch_y = self.get_dataset_items(self.y_key, inds)
         
        # Shall we apply any data augmentation?
        if self.augmenter is not None:
            batch_X = np.stack([
                self.augmenter(image=x)['image'] for x in batch_X
            ], axis=0)
                                        
        # Shall we rescale features?
        if self.scaler:
            batch_X = self.apply_normalization(batch_X)
            
        # Shall we apply labels encoding?
        if self.labels_encoding:
            batch_y = self.apply_labels_encoding(
                batch_y,
                smooth_factor = self.smooth_factor
                    if self.labels_encoding == 'smooth'
                    else 0)
                                        
        return (batch_X, batch_y)
    
    def on_epoch_end(self):
        """Triggered once at the very beginning as well as 
         at the end of each epoch. If the shuffle parameter 
         is set to True, image tensor indices will be shuffled.
        """
        if self.shuffle:
            np.random.shuffle(self.indices)