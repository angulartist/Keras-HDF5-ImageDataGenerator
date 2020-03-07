import os

import h5py as h5
import cv2

from sklearn.preprocessing import LabelEncoder

class HDF5ImageWriter(object):
    def __init__(self,
                 src,
                 dims,
                 X_key='images',
                 y_key='labels',
                 buffer_size=1024):
        
        self.db = h5.File(src, 'x')
        self.images = self.db.create_dataset(
            X_key, dims, dtype='float32', compression='gzip')
        self.labels = self.db.create_dataset(
            y_key, (dims[0],), dtype='int', compression='gzip')
        self.buffer_size = buffer_size
        self.buffer = {'tmp_images': [], 'tmp_labels': []}
        self.index = 0
  
    def add(self, images, labels):
        self.buffer['tmp_images'].extend(images)
        self.buffer['tmp_labels'].extend(labels)

        if len(self.buffer['tmp_images']) >= self.buffer_size:
            self.flush()

    def flush(self):
        index = self.index + len(self.buffer['tmp_images'])
        self.images[self.index : index] = self.buffer['tmp_images']
        self.labels[self.index : index] = self.buffer['tmp_labels']
        self.index = index
        
        self.buffer = {'tmp_images': [], 'tmp_labels': []}

    def close(self):
        if len(self.buffer['tmp_images']) > 0:
            self.flush()

        self.db.close()
        
import imutils

class Resizer(object):
    def __init__(self, width, height, inter=cv2.INTER_AREA):
        self.width = width
        self.height = height
        self.inter = inter

    def preprocess(self, image):
        (h, w) = image.shape[:2]
        dW = 0
        dH = 0

        if w < h:
            image = imutils.resize(image, width=self.width, inter=self.inter)
            dH = int((image.shape[0] - self.height) / 2.0)
        else:
            image = imutils.resize(image, height=self.height, inter=self.inter)
            dW = int((image.shape[1] - self.width) / 2.0)

        (h, w) = image.shape[:2]
    
        image = image[dH:h - dH, dW:w - dW]

        return cv2.resize(image, (self.width, self.height), interpolation=self.inter)
    
X = os.listdir('../../datasets/images')
human_labels = [path.split(os.path.sep)[-1].split('.')[0] for path in X]
labels_encoder = LabelEncoder()
y = labels_encoder.fit_transform(human_labels)

resizer = Resizer(256, 256)
writer = HDF5ImageWriter('../../datasets/c.h5', (len(X), 256, 256, 3))

for (i, (filename, y)) in enumerate(zip(X, y)):
    if i % 100 == 0:
        print('Images written', i)
    x = cv2.imread(f'../../datasets/images/{filename}')
    x = resizer.preprocess(x)
    writer.add([x], [y])
    
writer.close()