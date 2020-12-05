from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

import os
import numpy as np
import cv2
from functools import partial

from tensorflow.python.lib.io import file_io


##########################################################################################################################################
# Create data paths and info for training
##########################################################################################################################################

# Data directories
MODEL_DIR = None    # Set in task.py
TRAIN_DATA = None
TEST_DATA = None

TFRECORDS_TRAIN = file_io.FileIO(TRAIN_DATA, mode='rb')
TFRECORDS_TEST = file_io.FileIO(TEST_DATA, mode='rb')

# Training inputs
IMG_SIZE = 150
NUM_CHANNELS = 1
IMG_SHAPE = [IMG_SIZE,IMG_SIZE,NUM_CHANNELS]

# Hyperparameters 
BATCH_SIZE = None
NUM_CLASSES = None
NUM_EPOCHS = None
LEARNING_RATE = None
NUM_STEPS = None

##########################################################################################################################################
# Build custom CNN in Tensorflow 
##########################################################################################################################################

def model_fn(features, labels, mode, params):
    # Args:
    #
    # features: This is the x-arg from the input_fn.
    # labels:   This is the y-arg from the input_fn.
    # mode:     Either TRAIN, EVAL, or PREDICT
    # params:   User-defined hyper-parameters, e.g. learning-rate.
    
    # Reference to the tensor named "image" in the input-function.
    x = features["image"]

    # The convolutional layers expect 4-rank tensors
    # but x is a 2-rank tensor, so reshape it.
    net = tf.reshape(x, [-1, IMG_SIZE, IMG_SIZE, NUM_CHANNELS], name='input')    

    # First convolutional layer.
    net = tf.layers.conv2d(inputs=net, name='layer_conv1',
                           filters=128, kernel_size=3,
                           padding='same', activation=tf.nn.relu)
    net = tf.layers.max_pooling2d(inputs=net, pool_size=2, strides=2)

    # Second convolutional layer.
    net = tf.layers.conv2d(inputs=net, name='layer_conv2',
                           filters=128, kernel_size=3,
                           padding='same', activation=tf.nn.relu)
    net = tf.layers.max_pooling2d(inputs=net, pool_size=2, strides=2)    

    # Second convolutional layer.
    net = tf.layers.conv2d(inputs=net, name='layer_conv3',
                           filters=128, kernel_size=3,
                           padding='same', activation=tf.nn.relu)
    net = tf.layers.max_pooling2d(inputs=net, pool_size=2, strides=2) 

    # Flatten to a 2-rank tensor.
    net = tf.contrib.layers.flatten(net)
    # Eventually this should be replaced with:
    # net = tf.layers.flatten(net)

    # First fully-connected / dense layer.
    # This uses the ReLU activation function.
    net = tf.layers.dense(inputs=net, name='layer_fc1',
                          units=512, activation=tf.nn.relu)    
    net = tf.layers.dropout(inputs=net, rate=0.2)
    # Second fully-connected / dense layer.
    # This is the last layer so it does not use an activation function.
    net = tf.layers.dense(inputs=net, name='layer_fc_2',
                          units=512)
    net = tf.layers.dropout(inputs=net, rate=0.2)
    # Second fully-connected / dense layer.
    # This is the last layer so it does not use an activation function.
    net = tf.layers.dense(inputs=net, name='layer_fc_3',
                          units=params["num_classes"])
    
    logits = net 
    
    predictions = {
        "classes": tf.argmax(input=logits, axis=1),
        "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
      }


    if mode == tf.estimator.ModeKeys.PREDICT:
        # If the estimator is supposed to be in prediction-mode
        # then use the predicted class-number that is output by
        # the neural network. Optimization etc. is not needed.
        spec = tf.estimator.EstimatorSpec(mode=mode,
                                          predictions=predictions)
    else:
        # Otherwise the estimator is supposed to be in either
        # training or evaluation-mode. Note that the loss-function
        # is also required in Evaluation mode.
        
        # Define the loss-function to be optimized, by first
        # calculating the cross-entropy between the output of
        # the neural network and the true labels for the input data.
        # This gives the cross-entropy for each image in the batch.
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels,
                                                                       logits=logits)

        # Reduce the cross-entropy batch-tensor to a single number
        # which can be used in optimization of the neural network.
        loss = tf.reduce_mean(cross_entropy)

        # Define the optimizer for improving the neural network.
        optimizer = tf.train.AdamOptimizer(learning_rate=params["learning_rate"])

        # Get the TensorFlow op for doing a single optimization step.
        train_op = optimizer.minimize(
            loss=loss, global_step=tf.train.get_global_step())

        # Define the evaluation metrics,
        # in this case the classification accuracy.
        metrics = \
        {
            "accuracy": tf.metrics.accuracy(labels, predictions["classes"])
        }

        # Wrap all of this in an EstimatorSpec.
        spec = tf.estimator.EstimatorSpec(
            mode=mode,
            loss=loss,
            train_op=train_op,
            eval_metric_ops=metrics)
        
    return spec

##########################################################################################################################################
# Data input pipline, image preprocessing, train and evalulate fns
##########################################################################################################################################

def resize_image(image, lo_dim=IMG_SIZE):
    # Take width/height
    initial_width = tf.shape(image)[0]
    initial_height = tf.shape(image)[1]

    # Take the greater value, and use it for the ratio
    min_ = tf.minimum(initial_width, initial_height)
    ratio = tf.to_float(min_) / tf.constant(lo_dim, dtype=tf.float32)

    new_width = tf.to_int32(tf.to_float(initial_width) / ratio)
    new_height = tf.to_int32(tf.to_float(initial_height) / ratio)

    return tf.image.resize_images(image, [new_width, new_height])

def parse(serialized):
    with tf.Session() as sess:
        features = \
            {
                'image': tf.FixedLenFeature([], tf.string),
                'label': tf.FixedLenFeature([], tf.int64)
            }

        # Parse the serialized data so we get a dict with our data.
        parsed_example = tf.parse_single_example(serialized=serialized,
                                                 features=features)
        # Get the image as raw bytes.
        image_raw = parsed_example['image']
        # Decode the raw bytes so it becomes a tensor with type.
        image = tf.image.decode_jpeg(image_raw, channels=NUM_CHANNELS)
        resized_image = resize_image(image)
        image = tf.cast(resized_image, tf.float32) / 255.
        label = tf.cast(parsed_example['label'], tf.int64)
    return image, label

def input_fn(filenames, batch_size, train=False):

    # Convert the inputs to a Dataset. (E)
    dataset = tf.data.TFRecordDataset(filenames=filenames)
    dataset = dataset.map(parse, num_parallel_calls=4)

    if train:
        # If training then read a buffer of the given size and
        # randomly shuffle it.
        dataset = dataset.shuffle(buffer_size=batch_size * 10)
        # Allow infinite reading of the data.
        num_repeat = None
    else:
        # If testing then don't shuffle the data.
        # Only go through the data once.
        num_repeat = 1

    # Repeat the dataset the given number of times.
    dataset = dataset.repeat(num_repeat)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(1)
    iterator = dataset.make_one_shot_iterator()

    # Get the next batch of images and labels.
    images_batch, labels_batch = iterator.get_next()

    # The input-function must return a dict wrapping the images.
    x = {'image': images_batch}
    y = labels_batch
    
    return x, y

def train_input_fn():
    # Function to feed training data 
    return input_fn(filenames=TRAIN_DATA, batch_size=BATCH_SIZE, train=True)

def test_input_fn():
    # Function to feed testing data
    return input_fn(filenames=TEST_DATA, batch_size=BATCH_SIZE, train=False)

##########################################################################################################################################
# Build tf.estimator and run training and evaluation 
##########################################################################################################################################

def create_estimator(model_dir):
    params = {
        "learning_rate": LEARNING_RATE, 
        "num_classes": NUM_CLASSES,
        "input_shape": IMG_SHAPE
        }
    model = tf.estimator.Estimator(model_fn=model_fn,
                                params=params,
                                model_dir=model_dir)
    return model

def run_experiment(model_dir):
    tf.logging.set_verbosity(tf.logging.INFO)
    with tf.device('/device:GPU:0'):
        estimator = create_estimator(model_dir)
        for e in range(0, NUM_EPOCHS): 
            estimator.train(input_fn=train_input_fn, steps=NUM_STEPS)
            estimator.evaluate(input_fn=test_input_fn)

    return estimator
 
##########################################################################################################################################
# Serving input function and image preprocessing for predictions
##########################################################################################################################################

"""def image_preprocessing(image):
    Decodes jpeg string, resizes it and returns a uint8 tensor.
    image = tf.expand_dims(image, 0)
    image = tf.image.resize_bilinear(
        image, [IMG_SIZE, IMG_SIZE], align_corners=False)
    image = tf.squeeze(image, squeeze_dims=[0])
    image = tf.cast(image, dtype=tf.uint8)
    return image"""

def serving_input_receiver_fn():
    # URL serving input reciever function
    def prepare_image(image_str_tensor):
        image_contents = tf.read_file(image_str_tensor)
        image = tf.image.decode_jpeg(image_contents, channels=NUM_CHANNELS)
        return resize_image(image)

    input_ph = tf.placeholder(tf.string, shape=[None], name='serving_input_image')
    images_tensor = tf.map_fn(
        prepare_image, input_ph, back_prop=False, dtype=tf.uint8)
    images_tensor = tf.image.convert_image_dtype(images_tensor, dtype=tf.float32)
    return tf.estimator.export.ServingInputReceiver(
        {'image': images_tensor},
        {'serving_input_image': input_ph})



