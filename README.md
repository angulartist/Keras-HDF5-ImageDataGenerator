Keras HDF5 ImageDataGenerator
===============================

A blazing fast HDF5 Image Generator for Keras :zap:

Overview
--------

Sometimes you'd like to work with large scale image datasets that cannot fit into the memory. Luckily, Keras provides various data generators to feed your network with mini-batch of data directly from a directory, simply by passing the source path. But this method is terribly inefficient. During training, the model has to deal with massive I/Os operations on disk which introduces huge latency.

A more efficient way is to take advantage of HDF5 data structure which is optimized for I/O operations. The idea is to (1) store your raw images and their labels to an HDF5 file, and to (2) create a generator that will load and preprocess mini-batches in real-time.

This image generator is built on top of Keras `Sequence` class and it's safe for multiprocessing. It's also using the super-fast image-processing albumentations library.

Installation / Usage
--------------------

To install use pip:

    $ pip install h5imagegenerator
    
Dependencies
------------
* Keras
* Numpy
* Albumentations
* h5py
    
Contributing
------------

Feel free to PR any change/request. :grin:

Example
-------

First, import the image generator class:

```python
from h5imagegenerator import HDF5ImageGenerator
```

Then, create a new image generator:

```python
train_generator = HDF5ImageGenerator(
        src='path/to/train.h5',
        X_key='images,
        y_key='labels,
        scaler=True,
        labels_encoding='hot',
        batch_size=32,
        mode='train')
```

* **src**: the source HDF5 file
* **X_key**: the key of the image tensors dataset (default is `images`)
* **y_key**: the key of the labels dataset (default is `labels`)
* **scaler**: scale inputs to the range [0, 1] (basic normalization) (default is `True`)
* **labels_encoding**: set it to `hot` to convert integers labels to binary matrix (one hot encoding),
set it to `smooth` to perform smooth encoding (default is `hot`)
* **batch_size**: the number of samples to be generated at each iteration (default is `32`)
* **mode**: 'train' to generate tuples of image samples and labels, 'test' to generate image samples only (default is `'train'`)

Note: 

(1) When using `smooth` labels_encoding, you should provides a **smooth_factor** (defaults to `0.1`).

(2) Labels stored in the HDF5 file must be integers or list of lists/tuples of integers in case you're doing multi-labels classification. ie: `labels=[1, 2, 3, 6, 9] or labels=[(1, 2), (5, 9), (3, 9)]`...

Sometimes you'd like to perform some data augmentation on-the-fly, to flip, zoom, rotate or scale images. You can pass to the generator an [albumentations](https://github.com/albumentations-team/albumentations) transformation pipeline:

```python
my_augmenter = Compose([
        HorizontalFlip(p=0.5),
        RandomContrast(limit=0.2, p=0.5),
        RandomGamma(gamma_limit=(80, 120), p=0.5),
        RandomBrightness(limit=0.2, p=0.5),
        Resize(227, 227, cv2.INTER_AREA)])
    
train_generator = HDF5ImageGenerator(
        src='path/to/train.h5',
        X_key='images,
        y_key='labels,
        scaler=True,
        labels_encoding='hot',
        batch_size=32,
        augmenter=my_augmenter)
```

Note:

(1) albumentations offers a `ToFloat(max_value=255)` transformation which scales pixel intensities from [0, 255] to [0, 1]. Thus, when using it, you must turn off scaling: `scaler=False`.

(2) If you want to apply standardization (mean/std), you may want to use albumentations [Normalize](https://albumentations.readthedocs.io/en/latest/api/augmentations.html#albumentations.augmentations.transforms.Normalize) instead.

(3) Make sure to turn off data augmentation (`augmenter=False`) when using `evaluate_generator()` and `predict_generator()`.

Finally, pass the generator to your model:

```python
model.compile(
        loss='categorical_crossentropy',
        metrics=['accuracy'],
        optimizer='rmsprop')

# Example with fit:
model.fit_generator(
    train_generator,
    validation_data=val_generator,
    workers=10,
    use_multiprocessing=True,
    verbose=1,
    epochs=1)
    
    
# Example with evaluate:
model.evaluate_generator(
    eval_generator,
    workers=10,
    use_multiprocessing=True,
    verbose=1,
    epochs=1)
```
