"""Parser for VGG caffemodel."""
# Authors: Michael Eickenberg
#          Kyle Kastner
# License: BSD 3 Clause

from sklearn.externals import joblib
from ...datasets import get_dataset_dir, download
from .caffemodel import _parse_caffe_model, parse_caffe_model
from ...utils import check_tensor, get_minibatch_indices

#if the class label are the same than the googlenet one
from .googlenet_class_labels import get_googlenet_class_label
from .googlenet_layer_names import get_googlenet_layer_names

from sklearn.base import BaseEstimator, TransformerMixin

import os
import theano
import numpy as np

VGG_PATH = get_dataset_dir("caffe/vgg")


def fetch_vgg_protobuffer_file(caffemodel_file=None):
    """Checks for existence of caffemodel protobuffer.
    Downloads it if it cannot be found."""

    default_filename = os.path.join(VGG_PATH,
                                    "VGG_ILSVRC_19_layers.caffemodel")

    if caffemodel_file is not None:
        if os.path.exists(caffemodel_file):
            return caffemodel_file
        else:
            if os.path.exists(default_filename):
                import warnings
                warnings.warn('Did not find %s, but found and returned %s.' %
                              (caffemodel_file, default_filename))
                return default_filename
    else:
        if os.path.exists(default_filename):
            return default_filename

    # We didn't find the file, let's download it. To the specified location
    # if specified, otherwise to the default place
    if caffemodel_file is None:
        caffemodel_file = default_filename
        if not os.path.exists(VGG_PATH):
            os.makedirs(VGG_PATH)

    url = "http://www.robots.ox.ac.uk/%7Evgg/software/very_deep/caffe/"
    url += "VGG_ILSVRC_19_layers.caffemodel"
    download(url, caffemodel_file, progress_update_percentage=1)
    return caffemodel_file


def fetch_vgg_architecture(caffemodel_parsed=None, caffemodel_protobuffer=None):
    """Fetch a pickled version of the caffe model, represented as list of
    dictionaries."""

    default_filename = os.path.join(VGG_PATH, 'vgg.pickle')
    if caffemodel_parsed is not None:
        if os.path.exists(caffemodel_parsed):
            return joblib.load(caffemodel_parsed)
        else:
            if os.path.exists(default_filename):
                import warnings
                warnings.warn('Did not find %s, but found %s. Loading it.' %
                              (caffemodel_parsed, default_filename))
                return joblib.load(default_filename)
    else:
        if os.path.exists(default_filename):
            return joblib.load(default_filename)

    # We didn't find the file: let's create it by parsing the protobuffer
    protobuf_file = fetch_vgg_protobuffer_file(caffemodel_protobuffer)
    model = _parse_caffe_model(protobuf_file)

    if caffemodel_parsed is not None:
        joblib.dump(model, caffemodel_parsed)
    else:
        joblib.dump(model, default_filename)

    return model


def create_theano_expressions(model=None, verbose=0):

    if model is None:
        model = fetch_vgg_architecture()

    layers, blobs, inputs, params = parse_caffe_model(
        model, convert_fc_to_conv=False, verbose=verbose)
    data_input = inputs['data']
    return blobs, data_input


def _get_fprop(output_layers=('prob',), model=None, verbose=0):

    if model is None:
        model = fetch_vgg_architecture(model)

    expressions, input_data = create_theano_expressions(model,
                                                        verbose=verbose)
    to_compile = [expressions[expr] for expr in output_layers]

    return theano.function([input_data], to_compile)




class VGGClassifier(BaseEstimator):
    """
    A classifier for cropped images using the VGG neural network.

    Parameters
    ----------
    top_n : integer, optional (default=5)
        How many classes to return, based on sorted class probabilities.

    output_strings : boolean, optional (default=True)
        Whether to return class strings or integer classes. Returns class
        strings by default.

    Attributes
    ----------
    crop_bounds_ : tuple, (x_left, x_right, y_lower, y_upper)
        The coordinate boundaries of the cropping box used.

    """

    min_size = (224, 224)
    layer_names = get_googlenet_layer_names()

    def __init__(self, top_n=5, large_network=False, output_strings=True,
                 transpose_order=(0, 3, 1, 2)):

        self.top_n = top_n
        self.large_network = large_network
        self.output_strings = output_strings
        self.transpose_order = transpose_order
        self.transform_function = _get_fprop()

    def fit(self, X, y=None):
        """Passthrough for scikit-learn pipeline compatibility."""
        return self

    def _predict_proba(self, X):
        x_midpoint = X.shape[2] // 2
        y_midpoint = X.shape[1] // 2

        x_lower_bound = x_midpoint - self.min_size[0] // 2
        if x_lower_bound <= 0:
            x_lower_bound = 0
        x_upper_bound = x_lower_bound + self.min_size[0]
        y_lower_bound = y_midpoint - self.min_size[1] // 2
        if y_lower_bound <= 0:
            y_lower_bound = 0
        y_upper_bound = y_lower_bound + self.min_size[1]
        self.crop_bounds_ = (x_lower_bound, x_upper_bound, y_lower_bound,
                             y_upper_bound)

        res = self.transform_function(
            X[:, y_lower_bound:y_upper_bound,
                x_lower_bound:x_upper_bound, :].transpose(
                    *self.transpose_order))[0]

        return res

    def predict(self, X):
        """
        Classify a set of cropped input images.

        Returns the top_n classes.

        Parameters
        ----------
        X : array-like, shape = [n_images, height, width, color]
                        or
                        shape = [height, width, color]

        Returns
        -------
        T : array-like, shape = [n_images, top_n]

            Returns the top_n classes for each of the n_images in X.
            If output_strings is True, then the result will be string
            description of the class label.

            Otherwise, the returned values will be the integer class label.
        """
        X = check_tensor(X, dtype=np.float32, n_dim=4)
        res = self._predict_proba(X)[0]
        indices=np.argsort(res)
        indices=indices[-self.top_n:]
        if self.output_strings:
            class_strings = np.empty_like(indices,
                                          dtype=object)
            for index, value in enumerate(indices.flat):
                class_strings.flat[index] = get_googlenet_class_label(value)
            return class_strings
        else:
            return indices

    def predict_proba(self, X):
        """
        Prediction probability for a set of cropped input images.

        Returns the top_n probabilities.

        Parameters
        ----------
        X : array-like, shape = [n_images, height, width, color]
                        or
                        shape = [height, width, color]

        Returns
        -------
        T : array-like, shape = [n_images, top_n]

            Returns the top_n probabilities for each of the n_images in X.
        """
        X = check_tensor(X, dtype=np.float32, n_dim=4)
        res = self._predict_proba(X)[:, :, 0, 0]
        return np.sort(res, axis=1)[:, -self.top_n:]
