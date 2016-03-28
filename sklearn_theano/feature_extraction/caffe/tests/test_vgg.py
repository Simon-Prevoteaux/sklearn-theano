from skimage.data import coffee, camera

from sklearn_theano.feature_extraction.caffe.vgg import VGGClassifier
from sklearn_theano.feature_extraction.caffe.balanced_vgg import balancedVGGClassifier


import numpy as np
from nose import SkipTest
import os

co = coffee().astype(np.float32)
ca = camera().astype(np.float32)[:, :, np.newaxis] * np.ones((1, 1, 3),
                                                             dtype='float32')


def test_vgg_classifier():
    """smoke test for vgg classifier"""
    print 'testing the vgg classifier'
    if os.environ.get('CI', None) is not None:
        raise SkipTest("Skipping heavy data loading on CI")

    VGG = VGGClassifier()

    print VGG.predict(co)
    print VGG.predict(ca)

def test_balanced_vgg_classifier():
    """smoke test for vgg classifier"""
    print 'testing the balanced vgg classifier'
    if os.environ.get('CI', None) is not None:
        raise SkipTest("Skipping heavy data loading on CI")

    VGG = balancedVGGClassifier()

    print VGG.predict(co)
    print VGG.predict(ca)


if __name__=="__main__":
    test_vgg_classifier()
    #test_balanced_vgg_classifier()  #output doesn't seem to match the good predictions
