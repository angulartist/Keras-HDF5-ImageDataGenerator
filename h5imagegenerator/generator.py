from __future__ import absolute_import

from typing import Tuple, Union, Optional

import h5py as h5
from keras.utils import Sequence
from keras.utils import to_categorical

from albumentations import Compose
import numpy as np

available_modes = {'train', 'test'}
available_labels_encoding = {'hot', 'smooth', False}
    
class HDF5ImageGenerator(Sequence):
    """Just a simple custom Keras HDF5 ImageDataGenerator.
    
    Custom Keras ImageDataGenerator that generates
    batches of tensor images from HDF5 files with (optional) real-time
    data augmentation.
     
    Arguments
    ---------
    src : str
        Path of the hdf5 source file.
    X_key : str
        Key of the h5 file image tensors dataset.
        Default is "images".
    y_key : str
        Key of the h5 file labels dataset.
        Default is "labels".
    batch_size : int
        Size of each batch, must be a power of two.
        (16, 32, 64, 128, 256, ...)
        Default is 32.
    shuffle : bool
        Shuffle images at the end of each epoch.
        Default is True.
    scaler : "std", "norm" or False
        "std" mode means standardization to range [-1, 1]
        with 0 mean and unit variance.
        "norm" mode means normalization to range [0, 1].
        Default is "std".
    labels_encoding : "hot", "smooth" or False
        "hot" mode means classic one hot encoding.
        "smooth" mode means smooth hot encoding.
        Default is "hot".
    smooth_factor : int or float
        smooth factor used by smooth
        labels encoding.
        Default is 0.1.
    augmenter : albumentations Compose([]) Pipeline or False
        An albumentations transformations pipeline
        to apply to each sample.
        Default is False.
    mode : str "train" or "test"
        Model generator type. "train" is used for
        fit_generator() and evaluate_generator.
        "test" is used for predict_generator().
        Default is "train".
        
    Notes
    -----
    Turn off scaler (scaler=False) if using the
    ToFloat(max_value=255) transformation from
    albumentations.
        
    Examples
    --------
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
                 X_key='images',
                 y_key='labels',
                 batch_size=32,
                 shuffle=True,
                 scaler=True,
                 labels_encoding='hot',
                 smooth_factor=0.1,
                 augmenter=False,
                 mode='train'):
        
        if mode not in available_modes:
            raise ValueError(
                '`mode` should be `"train"`'
                '(fit_generator() and evaluate_generator()) or'
                '`"test"` (predict_generator().'
                'Received: %s' % mode)
        self.mode = mode
                
        if labels_encoding not in available_labels_encoding:
            raise ValueError(
                '`labels_encoding` should be `"hot"` '
                '(classic binary matrix) or '
                '`"smooth"` (smooth encoding) or '
                'False (no labels encoding).'
                'Received: %s' % labels_encoding)
        self.labels_encoding = labels_encoding
        
        if (self.labels_encoding == 'smooth') and not (0 < smooth_factor <= 1):
            raise ValueError(
                '`"smooth"` labels encoding'
                'must use a `"smooth_factor"`'
                '< 0 smooth_factor <= 1')
        
        if augmenter and not isinstance(augmenter, Compose):
             raise ValueError(
                 '`augmenter` argument'
                 'must be an instance of albumentations'
                 '`Compose` class.'
                 'Received type: %s' % type(augmenter))
        self.augmenter = augmenter
            
        self.src: str = src
        self.X_key: str = X_key
        self.y_key: str = y_key
        self.batch_size: int = batch_size
        self.shuffle: bool = shuffle
        self.scaler: bool = scaler
        self.smooth_factor: float = smooth_factor
        
        self.indices = np.arange(self.__get_dataset_shape(self.X_key, 0))
        
    def __get_dataset_shape(self, dataset: str, index: int) -> Tuple[int, ...]:
        """Get an h5py dataset shape.
        
        Arguments
        ---------
        dataset : str
            The dataset key.
        index : int
            The dataset index.
         
        Returns
        -------
        tuple of ints
            A tuple of array dimensions.
        """
        with h5.File(self.src, 'r', libver='latest', swmr=True) as file:
            return file[dataset].shape[index]
        
    def __get_dataset_items(self,
                            indices: np.ndarray,
                            dataset: Optional[str] = None) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """Get an HDF5 dataset items.
        
        Arguments
        ---------
        indices : ndarray, 
            The list of current batch indices.
        dataset : str or None
            The dataset key. If None, returns
            a batch of (image tensors, labels).
         
        Returns
        -------
        np.ndarray or a tuple of ndarrays
            A batch of samples.
        """
        with h5.File(self.src, 'r', libver='latest', swmr=True) as file:
            if dataset is not None:
                return file[dataset][indices]
            else:
                return (file[self.X_key][indices], file[self.y_key][indices])

    def __len__(self):
        """Denotes the number of batches per epoch.
         
        Returns
        -------
        int
            The number of batches per epochs.
        """
        return int(np.ceil(self.__get_dataset_shape(self.X_key, 0) / float(self.batch_size)))

    @staticmethod
    def apply_labels_smoothing(batch_y: np.ndarray, factor: float) -> np.ndarray:
        """Applies labels smoothing to the original
         labels binary matrix.
         
        Arguments
        ---------
        batch_y : np.ndarray
            Current batch integer labels.
        factor : float
            Smoothing factor.
        
        Returns
        -------
        np.ndarray
            A binary class matrix.
        """
        batch_y *= (1 - factor)
        batch_y += (factor / batch_y.shape[1])

        return batch_y

    def apply_labels_encoding(self,
                              batch_y: np.ndarray,
                              smooth_factor: float = 0.0) -> np.ndarray:
        """Converts a class vector (integers) to binary class matrix.
         See Keras to_categorical utils function.
         
        Arguments
        ---------
        batch_y : np.ndarray
            Current batch integer labels.
        smooth_factor : Int or Float
            Smooth factor.
        
        Returns
        -------
        np.ndarray
            A binary class matrix.
        """
        batch_y = to_categorical(batch_y)
        
        if smooth_factor > 0:
            batch_y = self.apply_labels_smoothing(batch_y, factor=smooth_factor)
            
        return batch_y
    
    @staticmethod
    def apply_normalization(batch_X: np.ndarray) -> np.ndarray:
        """Normalize the pixel intensities. 
        
        Normalize the pixel intensities to the range [0, 1].
         
        Arguments
        ---------
        batch_X : np.ndarray
            Batch of image tensors to be normalized.
        
        Returns
        -------
        np.ndarray
            A batch of normalized image tensors.
        """
        return batch_X.astype('float32') / 255.0
    
    def __next_batch_test(self, indices: np.ndarray) -> np.ndarray:
        """Generates a batch of test data for the given indices.
        
        Arguments
        ---------
        index : int
            The index for the batch.
            
        Returns
        -------
        ndarray
            4D tensor (num_samples, height, width, depth).
        """
        # Grab corresponding images from the HDF5 source file.
        batch_X = self.__get_dataset_items(indices, self.X_key)
                                        
        # Shall we rescale features?
        if self.scaler:
            batch_X = self.apply_normalization(batch_X)
                                        
        return batch_X
        
    def __next_batch(self, indices: np.ndarray) -> Tuple[np.ndarray, np.ndarray]: 
        """Generates a batch of train/val data for the given indices.
        
        Arguments
        ---------
        index : int
            The index for the batch.
            
        Returns
        -------
        tuple of ndarrays
            A tuple containing a batch of image tensors
            and their associated labels.
        """
        # Grab samples (tensors, labels) HDF5 source file.
        (batch_X, batch_y) = self.__get_dataset_items(indices)
        
        # Shall we apply any data augmentation?
        if self.augmenter:
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

    def __getitem__(self, index: int) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """Generates a batch of data for the given index.
        
        Arguments
        ---------
        index : int
            The index for the current batch.
            
        Returns
        -------
        tuple of ndarrays or ndarray
            A tuple containing a batch of image tensors
            and their associated labels (train) or
            a tuple of image tensors (predict).
        """
        # Indices for the current batch.
        indices = np.sort(self.indices[index * self.batch_size : (index + 1) * self.batch_size])

        return {
            'train': self.__next_batch,
            'test' : self.__next_batch_test
        }[self.mode](indices)
    
    def __shuffle_indices(self):
        """If the shuffle parameter is set to True,
         dataset will be shuffled (in-place).
         (not available in test 'mode').
        """
        if (self.mode == 'train') and self.shuffle:
            np.random.shuffle(self.indices)
    
    def on_epoch_end(self):
        """Triggered once at the very beginning as well as 
         at the end of each epoch.
        """
        self.__shuffle_indices()