import tensorflow as tf

# this is the workaround for Pycharm/Tensorflow problem with name and attributes resolving
keras = tf.keras
from keras import Input
from keras.applications import ResNet50
from keras.layers import (
    Activation, Add, AveragePooling2D, BatchNormalization,
    Conv2D, MaxPooling2D, ZeroPadding2D
)
from tensorflow.python.keras.utils import data_utils
import numpy as np
import pandas as pd
from typing import List, Tuple


def create_resnet_block(
        x: tf.Tensor,
        filters: int,
        kernel_size: int = 3,
        stride: int = 1,
        conv_shortcut: bool = True,
        name: str = None) -> tf.Tensor:
    """
    Create a single residual block
    :param x: input tensor
    :param filters: number of filters of the bottlneck layer
                    A bottleneck layer is a layer that contains few nodes compared to the previous layers.
                    It can be used to obtain a representation of the input with reduced dimensionality.
                    An example of this is the use of autoencoders with bottleneck layers for nonlinear dimensionality reduction.
    :param kernel_size: kernel size of the bottlneck layer
    :param stride: stride of the first layer
    :param conv_shortcut: use convolution shortcut if True.
                          Convolution shortcut skips some of the layers in the convolution block
                          and feeds the output of one layer as the input to the next layers
    :param name: name of the block, added as part of each layer
    :return: output tf.Tensor
    """

    # channels_last format
    bn_axis = 3

    # create shortcut in addition to main branch
    if conv_shortcut:
        shortcut = Conv2D(filters=4 * filters, kernel_size=1, strides=stride,
                          name="_".join([name, "0", 'CONV_shortcut']))(x)
        shortcut = BatchNormalization(axis=bn_axis, epsilon=1.001e-5,
                                      name="_".join([name, "0", 'BN_shortcut']))(shortcut)
    else:
        shortcut = x

    x = Conv2D(filters=filters, kernel_size=1, strides=stride, name="_".join([name, "1", 'conv']))(
        x)
    x = BatchNormalization(axis=bn_axis, epsilon=1.001e-5, name="_".join([name, "1", 'BN']))(x)
    x = Activation("relu", name="_".join([name, "1", 'ReLu']))(x)

    x = Conv2D(filters=filters, kernel_size=kernel_size, padding="SAME",
               name="_".join([name, "2", 'conv']))(x)
    x = BatchNormalization(axis=bn_axis, epsilon=1.001e-5, name="_".join([name, "2", 'BN']))(x)
    x = Activation("relu", name="_".join([name, "2", 'ReLu']))(x)

    x = Conv2D(filters=4 * filters, kernel_size=1, name="_".join([name, "3", 'CONV']))(x)
    x = BatchNormalization(axis=bn_axis, epsilon=1.001e-5, name="_".join([name, "3", 'BN']))(x)

    # adding shortcut to main branch part
    x = Add(name="_".join([name, 'Add']))([shortcut, x])
    x = Activation(activation="relu", name="_".join([name, 'OUT']))(x)
    return x


def create_resnet_stack(
        x: tf.Tensor,
        filters: int,
        blocks: int,
        stride1: int = 2,
        name=None) -> tf.Tensor:
    """
    Create set of stacked residual blocks

    :param x: input tensor,
    :param filters: number of filters in the bottleneck layer in the block
    :param blocks: number of blocks in the stack,
    :param stride_1: stride of the first layer in the first block,
    :param name: stack name
    :return: tensor after the stacked block
    """
    x = create_resnet_block(x=x,filters= filters, kernel_size=3, stride=stride1, name="_".join([name + "block1"]))
    for i in range(2, blocks + 1):
        x = create_resnet_block(x, filters, conv_shortcut=False, name="_".join([name, "block" + str(i)]))
    return x


def set_conv_weights(
        model : tf.keras.models.Model,
        feature_extractor: tf.keras.applications.ResNet50):
    """
    Load pretrained ResNet50 weights for final dense layer into final FCN CNN layer

    :param model: the model to load weights in
    :param feature_extractor:
    :return:
    """
    # get pre-trained ResNet50 FC weights
    dense_layer_weights = feature_extractor.layers[-1].get_weights()
    weights_list = [
        tf.reshape(
            dense_layer_weights[0], (1, 1, *dense_layer_weights[0].shape),
        ).numpy(),
        dense_layer_weights[1],
    ]
    model.get_layer(name="last_conv").set_weights(weights_list)


def build_fully_convolutional_resnet50(
        input_shape: Tuple,
        base_weights_path: str,
        weights_hashes: str,
        num_classes:int= 1000,
        pretrained_resnet:bool= True,
        use_bias:bool= True,
) -> tf.keras.models.Model:
    """
    Build ResNet50 with weights loaded and adapted to FCN architecture.
    :param input_shape: shape of the image. It must be provided to create model,
                        but it does not imapct the ability to classify images of different dimensions
    :param base_weights_path: path used to load pretrained weights from
    :param weights_hashes: hash of weights file
    :param num_classes: number of output classes
    :param pretrained_resnet: flag indicating if pretrained weights are loaded
    :param use_bias: flag indicating if bias has to be added
    :return: model with loaded weights
    """
    # init input layer
    img_input = Input(shape=input_shape)

    # define basic model pipeline
    # add zeros on each side of image
    x = ZeroPadding2D(padding=((3, 3), (3, 3)), name="conv1_pad")(img_input)
    x = Conv2D(64, 7, strides=2, use_bias=use_bias, name="conv1_conv")(x)
    x = BatchNormalization(axis=3, epsilon=1.001e-5, name="conv1_bn")(x)
    x = Activation("relu", name="conv1_relu")(x)

    x = ZeroPadding2D(padding=((1, 1), (1, 1)), name="pool1_pad")(x)
    x = MaxPooling2D(3, strides=2, name="pool1_pool")(x)

    # the sequence of stacked residual blocks
    x = create_resnet_stack(x, 64, 3, stride1=1, name="conv2")
    x = create_resnet_stack(x, 128, 4, name="conv3")
    x = create_resnet_stack(x, 256, 6, name="conv4")
    x = create_resnet_stack(x, 512, 3, name="conv5")

    # add avg pooling layer after feature extraction layers
    x = AveragePooling2D(pool_size=7)(x)

    # add final convolutional layer
    # the last layer is also convolution to solve not-fixed  dimensions consequence
    conv_layer_final = Conv2D(filters=num_classes, kernel_size=1,
                              use_bias=use_bias, name="last_conv")(x)

    # configure fully convolutional ResNet50 model
    model = tf.keras.models.Model(img_input, x)

    # load model weights
    if pretrained_resnet:
        model_name = "resnet50"
        # configure full file name
        file_name = model_name + "_weights_tf_dim_ordering_tf_kernels_notop.h5"
        # get the file hash from TF WEIGHTS_HASHES
        file_hash = weights_hashes[model_name][1]
        weights_path = data_utils.get_file(
            file_name,
            base_weights_path + file_name,
            cache_subdir="models",
            file_hash=file_hash,
        )

        model.load_weights(weights_path)

    # form final model
    model = tf.keras.models.Model(inputs=model.input, outputs=[conv_layer_final])

    if pretrained_resnet:
        # get model with the dense layer for further FC weights extraction
        resnet50_extractor = ResNet50(
            include_top=True, weights="imagenet", classes=num_classes,
        )
        # set ResNet50 FC-layer weights to final convolutional layer
        set_conv_weights(model=model, feature_extractor=resnet50_extractor)
    return model

def get_prediction(prediction:tf.Tensor, labels:List) -> (str, pd.DataFrame):
    """
    Extract predicted class from ResNet50-FCN result tensor

    :param prediction: result tensor from ResNet50-FCN
    :param labels: list of labels for ResNet50-FCN
    :return: tuple composed of the label for predicted class and Dataframe with
             other classes found by ResNet50 in image
    """
    preds = tf.nn.softmax(prediction, axis=3)
    pred_max = np.max(preds, 3)
    class_idx = np.argmax(preds,3)
    max_mask = pred_max == np.max(pred_max)
    most_likely_class = np.max(class_idx * max_mask)
    recognized_objects = np.unique(class_idx)
    recognised_objects = pd.DataFrame([
        (ro, np.sum(class_idx == ro), np.max(pred_max * (class_idx == ro)),
         labels[ro]) for ro in recognized_objects],
        columns=['label index', 'occurences','max probability', 'object']).sort_values(
        by = 'max probability', ascending = False)

    return labels[most_likely_class], recognised_objects