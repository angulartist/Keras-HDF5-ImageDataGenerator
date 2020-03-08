Keras HDF5 ImageDataGenerator
===============================

A dead simple Keras HDF5 ImageDataGenerator.

Overview
--------

Sometimes you'd like to work with large scale image datasets that cannot fit into the memory. Keras provides data generators to feed your network with mini-batch of data directly from a directory, simply by passing the image paths. But this method is terribly inefficient because during training, the model has to deal with massive I/Os operations on disk which introduces huge latency.

A more efficient way is to take advantage of HDF5 data structure which is optimized for I/O operations. The idea is to (1) store your raw images (and their labels) to an HDF5 file, and to (2) create a generator that will load and preprocess mini-batches in real-time.

Installation / Usage
--------------------

To install use pip:

    $ pip install h5imagegenerator
    
Contributing
------------

michel @angulartist

Example
-------



TODO LIST
-------
* [x] Generator
* [] Docs
* [] 
* [] 
