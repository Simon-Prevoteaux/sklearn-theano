from skimage.data import coffee, camera
from sklearn_theano.feature_extraction import (
    GoogLeNetTransformer, GoogLeNetClassifier)

from sklearn_theano.feature_extraction.caffe.vgg import VGGClassifier

import numpy as np
from nose import SkipTest
import os

co = coffee().astype(np.float32)
ca = camera().astype(np.float32)[:, :, np.newaxis] * np.ones((1, 1, 3),dtype='float32')


if __name__=="__main__":
    print 'This test aims to compare the outputs of a vgg and a googlenet classifier'

    print 'Loading the models...'
    VGG = VGGClassifier()
    c = GoogLeNetClassifier()

    print '-----First example'

    print '\t Using the vgg :'
    print VGG.predict(co)

    print '\t Using the googlenet :'
    print c.predict(co)

    print '-----Second example'

    print '\t Using the vgg :'
    print VGG.predict(ca)

    print '\t Using the googlenet :'
    print c.predict(ca)
