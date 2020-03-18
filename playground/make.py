from keras.datasets import mnist

import h5py as h5

(X_train, y_train), (X_test, y_test) = mnist.load_data()

X_train = X_train.reshape(-1, 28, 28, 1)
X_test = X_test.reshape(-1, 28, 28, 1)


class HDF5ImageWriter(object):
    def __init__(self, src, dims, X_key="images", y_key="labels", buffer_size=1024):

        self.db = h5.File(src, "x")
        self.images = self.db.create_dataset(X_key, dims, dtype="float32")
        self.labels = self.db.create_dataset(y_key, (dims[0],), dtype="uint8")
        self.buffer_size = buffer_size
        self.buffer = {"tmp_images": [], "tmp_labels": []}
        self.index = 0

    def add(self, images, labels):
        self.buffer["tmp_images"].extend(images)
        self.buffer["tmp_labels"].extend(labels)

        if len(self.buffer["tmp_images"]) >= self.buffer_size:
            self.flush()

    def flush(self):
        index = self.index + len(self.buffer["tmp_images"])
        self.images[self.index : index] = self.buffer["tmp_images"]
        self.labels[self.index : index] = self.buffer["tmp_labels"]
        self.index = index

        self.buffer = {"tmp_images": [], "tmp_labels": []}

    def close(self):
        if len(self.buffer["tmp_images"]) > 0:
            self.flush()

        self.db.close()


writer = HDF5ImageWriter(
    src="/storage/datasets/mnist_test.h5", dims=(len(X_test), 28, 28, 1)
)

for (i, (x, y)) in enumerate(zip(X_test, y_test)):
    writer.add([x], [y])

    if i % 1024 == 0:
        print("Images written", i)

writer.close()
