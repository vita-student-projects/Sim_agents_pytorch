import os
import tensorflow as tf

DATASET_FOLDER = 'waymo_open_dataset_'

TRAIN_FILES = os.path.join(DATASET_FOLDER, 'training.tfrecord*')
VALIDATION_FILES = os.path.join(DATASET_FOLDER, 'validation.tfrecord*')
TEST_FILES = os.path.join(DATASET_FOLDER, 'testing.tfrecord*')

def Prepare_train_dataset():
    filenames = tf.io.matching_files(TRAIN_FILES)
    dataset = tf.data.TFRecordDataset(filenames)
    return dataset

def Prepare_validation_dataset():
    filenames = tf.io.matching_files(VALIDATION_FILES)
    dataset = tf.data.TFRecordDataset(filenames)
    return dataset

def Prepare_test_dataset():
    filenames = tf.io.matching_files(TEST_FILES)
    dataset = tf.data.TFRecordDataset(filenames)
    return dataset