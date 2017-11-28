# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#  change the label algorithm   ljqu  2017.4.28
# ==============================================================================

"""Functions for downloading and reading MNIST data."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy
from six.moves import xrange  # pylint: disable=redefined-builtin

import csv
from glob import glob

import numpy as np
from config import cfg
from tensorflow.python.platform import gfile

from tensorflow.contrib.learn.python.learn.datasets import base
from tensorflow.python.framework import dtypes

DATALEN = cfg.data_period  # 125
LABELNUM = cfg.label_num  # 19
LBLPOSTNUM = cfg.label_post_num  # 20        #利用后面多少个数据来计算标签
loss_ratio = cfg.loss_ratio  # 0.05
profit_ratio = cfg.profit_ratio  # 0.20
image_size = cfg.image_size


def _read32(bytestream):
    dt = numpy.dtype(numpy.uint32).newbyteorder('>')
    return numpy.frombuffer(bytestream.read(4), dtype=dt)[0]


# %%
def extract_images(filequeue):
    """Extract the images into a 4D uint8 numpy array [index, y, x, depth].

    Args:
        f: A file object that can be passed into a gzip reader.

    Returns:
        train_iamges: A 4D float32 numpy array [index, x, y, depth].
        train_labels: A 1D int numpy  array [index]

    Raises:
        ValueError: If the bytestream does not start with 2051.

    """
    features_dtype = "float32"
    target_dtype = "int"
    for i, filename in enumerate(filequeue):
        print('Extracting', filename)
        train_image, train_label = get_csv_data1(filename, target_dtype, features_dtype, DATALEN)
        if i == 0:
            train_images = train_image
            train_labels = train_label
        else:
            train_images = np.concatenate((train_images, train_image), axis=0)
            train_labels = np.concatenate((train_labels, train_label), axis=0)
    return train_images, train_labels


# %%
def get_csv_data1(filename,
                  target_dtype,
                  features_dtype,
                  datalen=DATALEN,
                  fileout=False):
    """  get data from given file(.txt or .csv)
        Args:
            file: the given file (the full name)
            target_dtype:  datatype of the label
            features_dtype:  datatype of the features
            datalen:  the length of data
        Returns:
            train_images:  feature 4D array [Index,25,25,1]
            train_labels:  label 1D array  [Index]
    """
    idx = -1
    with gfile.Open(filename) as csv_file:
        # read data according to the file name
        data = numpy.loadtxt(open(filename), dtype=features_dtype, delimiter=",", skiprows=1, usecols=(2, 3, 4, 5, 6))
        n_samples = data.shape[0]  # int(header[0])
        n_features = data.shape[1]  # header.shape[0] - 2
        # print("data:", data.shape, ",n_sampes:",n_samples,",n_features:",n_features )
        train_images = np.zeros((n_samples - datalen - LBLPOSTNUM + 1, datalen * n_features), dtype=features_dtype)
        train_labels = np.zeros(n_samples - datalen - LBLPOSTNUM + 1, dtype=target_dtype)

        # deal the data one by one
        for i, d in enumerate(data):
            if i >= datalen and i <= data.shape[0] - LBLPOSTNUM:
                fArr1 = diverse_standard_comp(data[i - datalen:i, :], 0)  # open
                fArr2 = diverse_standard_comp(data[i - datalen:i, :], 1)  # high
                fArr3 = diverse_standard_comp(data[i - datalen:i, :], 2)  # low
                fArr4 = diverse_standard_comp(data[i - datalen:i, :], 3)  # close
                fArr5 = diverse_standard(data[i - datalen:i, 4])  # volume
                f1 = np.transpose(fArr1)
                f2 = np.transpose(fArr2)
                f3 = np.transpose(fArr3)
                f4 = np.transpose(fArr4)
                f5 = np.transpose(fArr5)

                # deal the target category (one-shot = ture)
                tArr = data[i - datalen:i + LBLPOSTNUM, :]  # the column to calculate the profit
                label_data = deal_label(tArr, datalen)

                idx += 1
                train_images[idx] = np.concatenate((f1, f2, f3, f4, f5), axis=1)
                train_labels[idx] = label_data
    if fileout:
        out_indi_data(filename, train_labels, "mc_label/", datalen)
    trn_images = train_images[:idx + 1, ]
    trn_images = trn_images.reshape([-1, image_size, image_size, 1])
    trn_labels = train_labels[:idx + 1, ]
    return trn_images, trn_labels


# %%
def diverse_standard_comp(x, k):
    x_max = max(x[:, 1])
    x_min = min(x[:, 2])
    if (x_max - x_min) < 1e-10:  # when in top limit or lower limit,
        # print(x_max,",",x_min)
        x1 = np.ones(x.shape[0])
    else:
        x1 = np.round((x[:, k] - x_min) / (x_max - x_min), 4)  # round the data to 1e-4
    x1 = x1 - 0.5  # uncentral
    x1 = x1.reshape(x.shape[0], 1)
    return x1


# %%
def diverse_standard(x):
    x_max = max(x)
    x_min = min(x)
    if (x_max - x_min) < 1e-10:  # when in top limit or lower limit,
        # print(x_max,",",x_min)
        x1 = np.ones(x.shape[0])
    else:
        x1 = np.round((x - x_min) / (x_max - x_min), 4)  # round the data to 1e-4
    x1 = x1 - 0.5  # uncentral
    x1 = x1.reshape(x.shape[0], 1)
    return x1


# %%
def deal_label(x, len1=DATALEN):
    """According to the pre 100 bars and post 20 bars
        to calculate the label.

    Args:
        x: An array that include the pre 100 and post 20 bars.
        len1: data length

    Returns:
        lbl: A int scalar. Represent the current bar's label

    """
    x1 = x[0:len1, :]
    x2 = x[len1 + 1:len1 + LBLPOSTNUM, :]  # the old value is :x2=x[len1,len1+20,:]   ,update 2017.5.13
    close = x[len1, 3]  # the old value is:close = x[len1 - 1, 3],  because of the sign is in the dataLen Bar,
    # so define the openprice is the close of the dataLen Bar. update 2017.5.13

    # calculate the 100 bars range
    x1_max = max(x1[:, 1])  # high
    x1_min = min(x1[:, 2])  # low
    x1_range = x1_max - x1_min

    x2_max = max(x2[:, 1])  # high
    x2_min = min(x2[:, 2])  # low
    x2_hc = max(x2[:, 3])  # highest close
    x2_lc = min(x2[:, 3])  # lowest close

    long_loss = close - x2_min
    long_profit = x2_hc - close

    long_loss_std = long_loss / x1_range
    long_profit_std = long_profit / x1_range

    short_loss = x2_max - close
    short_profit = close - x2_lc
    short_loss_std = short_loss / x1_range
    short_profit_std = short_profit / x1_range

    long_lbl = 0
    short_lbl = 0
    lbl = 0

    if long_loss_std >= loss_ratio:
        long_lbl = 0
    elif long_profit_std < profit_ratio:
        long_lbl = 0
    else:
        long_lbl = int((long_profit_std + 0.000001 - profit_ratio) / 0.2) + 1
    if long_lbl > 9:
        long_lbl = 9

    if short_loss_std >= loss_ratio:
        short_lbl = 0
    elif short_profit_std < profit_ratio:
        short_lbl = 0
    else:
        short_lbl = -int((short_profit_std + 0.000001 - profit_ratio) / 0.2) - 1
    if short_lbl < -9:
        short_lbl = -9

    if long_lbl >= abs(short_lbl):
        lbl = long_lbl
    else:
        lbl = abs(short_lbl) + 9
    return lbl


def dense_to_one_hot(labels_dense, num_classes):
    """Convert class labels from scalars to one-hot vectors."""
    num_labels = labels_dense.shape[0]
    index_offset = numpy.arange(num_labels) * num_classes
    labels_one_hot = numpy.zeros((num_labels, num_classes))
    labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
    return labels_one_hot


def extract_labels(labels, one_hot=False, num_classes=LABELNUM):
    """Extract the labels into a 1D uint8 numpy array [index].

    Args:
      labels: a 1D unit8 numpy array.
      one_hot: Does one hot encoding for the result.
      num_classes: Number of classes for the one hot encoding.

    Returns:
      labels: a 2D unit8 numpy array [index,19].

    Raises:
      ValueError: If the bystream doesn't start with 2049.
    """
    if one_hot:
        return dense_to_one_hot(labels, num_classes)
    return labels


class DataSet(object):
    def __init__(self,
                 images,
                 labels,
                 fake_data=False,
                 one_hot=False,
                 dtype=dtypes.float32,
                 reshape=True):
        """Construct a DataSet.
        one_hot arg is used only if fake_data is true.  `dtype` can be either
        `uint8` to leave the input as `[0, 255]`, or `float32` to rescale into
        `[0, 1]`.
        """
        dtype = dtypes.as_dtype(dtype).base_dtype
        if dtype not in (dtypes.uint8, dtypes.float32):
            raise TypeError('Invalid image dtype %r, expected uint8 or float32' %
                            dtype)
        if fake_data:
            self._num_examples = 10000
            self.one_hot = one_hot
        else:
            assert images.shape[0] == labels.shape[0], (
                'images.shape: %s labels.shape: %s' % (images.shape, labels.shape))
            self._num_examples = images.shape[0]

            # Convert shape from [num examples, rows, columns, depth]
            # to [num examples, rows*columns] (assuming depth == 1)
            if reshape:
                assert images.shape[3] == 1
                images = images.reshape(images.shape[0],
                                        images.shape[1] * images.shape[2])
            if dtype == dtypes.float32:
                # Convert from [0, 255] -> [0.0, 1.0].
                images = images.astype(numpy.float32)
                # images = numpy.multiply(images, 1.0 / 255.0)
        self._images = images
        self._labels = labels
        self._epochs_completed = 0
        self._index_in_epoch = 0

    @property
    def images(self):
        return self._images

    @property
    def labels(self):
        return self._labels

    @property
    def num_examples(self):
        return self._num_examples

    @property
    def epochs_completed(self):
        return self._epochs_completed

    def next_batch(self, batch_size, fake_data=False):
        """Return the next `batch_size` examples from this data set."""
        if fake_data:
            fake_image = [1] * 500
            if self.one_hot:
                fake_label = [1] + [0] * 9
            else:
                fake_label = 0
            return [fake_image for _ in xrange(batch_size)], [
                fake_label for _ in xrange(batch_size)
            ]
        start = self._index_in_epoch
        self._index_in_epoch += batch_size
        if self._index_in_epoch > self._num_examples:
            # Finished epoch
            self._epochs_completed += 1
            # Shuffle the data
            perm = numpy.arange(self._num_examples)
            numpy.random.shuffle(perm)
            self._images = self._images[perm]
            self._labels = self._labels[perm]
            # Start next epoch
            start = 0
            self._index_in_epoch = batch_size
            assert batch_size <= self._num_examples
        end = self._index_in_epoch
        return self._images[start:end], self._labels[start:end]


# %%
def read_data_sets(train_dir,
                   fake_data=False,
                   one_hot=False,
                   dtype=dtypes.float32,
                   reshape=False,
                   validation_size=100):
    if fake_data:
        def fake():
            return DataSet([], [], fake_data=True, one_hot=one_hot, dtype=dtype)

        train = fake()
        validation = fake()
        test = fake()
        return base.Datasets(train=train, validation=validation, test=test)

    train_file = glob(train_dir)
    # print(train_file)
    train_images, train_labels = extract_images(train_file)
    train_labels = extract_labels(train_labels, one_hot=one_hot)
    print(train_images.shape)

    # TEST_IMAGES = (['test_data/rb.HOT.15m(1).csv'])
    test_file = glob(cfg.test_dataset)
    test_images, test_labels = extract_images(test_file)
    test_labels = extract_labels(test_labels, one_hot=one_hot)

    if not 0 <= validation_size <= len(train_images):
        raise ValueError(
            'Validation size should be between 0 and {}. Received: {}.'
                .format(len(train_images), validation_size))

    validation_images = train_images[:validation_size]
    validation_labels = train_labels[:validation_size]
    train_images = train_images[validation_size:]
    train_labels = train_labels[validation_size:]

    train = DataSet(train_images, train_labels, dtype=dtype, reshape=reshape)
    validation = DataSet(validation_images,
                         validation_labels,
                         dtype=dtype,
                         reshape=reshape)
    test = DataSet(test_images, test_labels, dtype=dtype, reshape=reshape)

    return base.Datasets(train=train, validation=validation, test=test)


# %%
def read_test_sets(filename,
                   one_hot=False,
                   dtype=dtypes.float32,
                   reshape=False,
                   validation_size=100):
    local_file = [filename]  # TRAIN_IMAGES
    train_images, train_labels = extract_images(local_file)
    train_labels = extract_labels(train_labels, one_hot=one_hot)

    train = DataSet(train_images, train_labels, dtype=dtype, reshape=reshape)

    return train


# %%

def load_data(train_dir='train_data'):
    return read_data_sets(train_dir)


# %%
def get_csv_data2(filename,
                  features_dtype="float32",
                  datalen=DATALEN):
    """  get data from given file(.txt or .csv)
        Args:
            file: the given file (the full name)
            features_dtype:  datatype of the features
            datalen:  the length of data
        Returns:
            train_images:  feature 2D array [Index,900]
            train_labels:  label 1D array  [Index]
    """
    idx = -1
    with gfile.Open(filename) as csv_file:
        data_file = csv.reader(csv_file)  # DictReader
        header = next(data_file)
        header = np.asarray(header)
        n_samples = int(header[0])
        n_features = header.shape[0] - 2
        data = np.zeros((n_samples, n_features), dtype=features_dtype)
        train_images = np.zeros((n_samples - datalen, datalen * n_features), dtype=features_dtype)
        for i, row in enumerate(data_file):
            data[i] = np.asarray((row[2], row[3], row[4], row[5], row[6]), features_dtype)
            # print(" i = %d row = %s " % (i,row))
            #
        # get the end record [date & time ]
        dt = [row[0], row[1]]

        for i, d in enumerate(data):
            if i >= datalen:
                fArr1 = diverse_standard_comp(data[i - datalen:i, :], 0)  # open
                fArr2 = diverse_standard_comp(data[i - datalen:i, :], 1)  # high
                fArr3 = diverse_standard_comp(data[i - datalen:i, :], 2)  # low
                fArr4 = diverse_standard_comp(data[i - datalen:i, :], 3)  # close
                fArr5 = diverse_standard(data[i - datalen:i, 4])  # volume
                f1 = np.transpose(fArr1)
                f2 = np.transpose(fArr2)
                f3 = np.transpose(fArr3)
                f4 = np.transpose(fArr4)
                f5 = np.transpose(fArr5)
                idx += 1
                train_images[idx] = np.concatenate((f1, f2, f3, f4, f5), axis=1)
    return train_images, dt


# %%   output the original data and indi data to a csv file
def out_indi_data(infile,
                  indi,
                  outpath="mc_test/",
                  datalen=DATALEN):
    """  get data from given file(.txt or .csv)
        Args:
            infile: the given file (the full name)
            indi: calculated indicator values(array)[None]
            datalen:  the length of data
        Returns:
    """
    path = outpath  # os.path.split(os.path.realpath(__file__))[0] + "/"+outpath+"/"
    filename = infile
    tempfile = filename.split("/")
    outfile = tempfile[len(tempfile) - 1]
    outfile = outfile.replace("csv", "indi.csv")
    outfile = path + outfile

    fp = open(outfile, 'w')
    writer = csv.writer(fp)

    with gfile.Open(filename) as csv_file:
        data_file = csv.reader(csv_file)  # DictReader
        header = next(data_file)
        header1 = header + [' <Indi>']
        header1[0] = ' <Date>'
        writer.writerow(header1)  # write the header
        header = np.asarray(header)
        n_samples = int(header[0])
        n_features = header.shape[0] + 1
        for i, row in enumerate(data_file):
            if i < datalen:
                idx = '0'
            else:
                idx = str(indi[i - datalen])
            writer.writerow(row + [idx])
            # print(" i = %d row = %s " % (i,row),type(row))

    # close the file pointer
    fp.close()