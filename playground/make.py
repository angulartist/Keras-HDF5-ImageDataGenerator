import os

from keras.datasets import mnist
from albumentations import Compose, HorizontalFlip, RandomGamma, ToFloat, Resize
import h5py as h5
from concurrent.futures import ProcessPoolExecutor as PoolExecutor
from pytictoc import TicToc

(X_train, y_train), (X_test, y_test) = mnist.load_data()

X_train = X_train.reshape(-1, 28, 28, 1)
X_test = X_test.reshape(-1, 28, 28, 1)


class FileAlreadyOpenError(RuntimeError):
    pass


class HDF5ImageWriter(object):
    def __init__(self, src, dims, X_key="images", y_key="labels", buffer_size=512):

        self.src: str = src
        self.dims = dims
        self.X_key: str = X_key
        self.y_key: str = y_key
        self.db = None
        self.images = None
        self.labels = None
        self.buffer_size = buffer_size
        self.buffer = {"tmp_images": [], "tmp_labels": []}
        self._index = 0

    def __enter__(self):
        if self.db is not None:
            raise FileAlreadyOpenError("The HDF5 file is already open!")

        self.db = h5.File(self.src, "w")
        self.images = self.db.create_dataset(self.X_key, self.dims, dtype="float32")
        self.labels = self.db.create_dataset(self.y_key, (self.dims[0],), dtype="uint8")

        return self

    def __exit__(self, type_, value, traceback):
        self.__close()

    def add(self, images, labels):
        self.buffer["tmp_images"].extend(images)
        self.buffer["tmp_labels"].extend(labels)

        if len(self.buffer["tmp_images"]) >= self.buffer_size:
            self.__flush()

    def __flush(self):
        index = self._index + len(self.buffer["tmp_images"])
        self.images[self._index : index] = self.buffer["tmp_images"]
        self.labels[self._index : index] = self.buffer["tmp_labels"]
        self._index = index

        self.buffer = {"tmp_images": [], "tmp_labels": []}

    def __close(self):
        if len(self.buffer["tmp_images"]) > 0:
            self.__flush()

        self.db.close()


pipeline = Compose(
    [
        HorizontalFlip(p=0.5),
        RandomGamma(gamma_limit=(80, 120), p=0.5),
        HorizontalFlip(p=0.5),
        HorizontalFlip(p=0.5),
        HorizontalFlip(p=0.5),
        HorizontalFlip(p=0.5),
        HorizontalFlip(p=0.5),
        HorizontalFlip(p=0.5),
    ]
)


def transform(sample):
    x, y = sample

    return pipeline(image=x)["image"], y


h5_writer = HDF5ImageWriter(
    src="/storage/datasets/mnist_test.h5", dims=(len(X_train), 28, 28, 1)
)

with TicToc():
    with h5_writer as writer:
        with PoolExecutor(max_workers=os.cpu_count()) as executor:
            for image, label in executor.map(transform, zip(X_train, y_train)):
                print("Adding:", label)
                writer.add([image], [label])
