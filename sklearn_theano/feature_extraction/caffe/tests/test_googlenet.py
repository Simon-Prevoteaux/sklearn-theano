from skimage.data import coffee, camera
from sklearn_theano.feature_extraction import (
    GoogLeNetTransformer, GoogLeNetClassifier)
import numpy as np
from nose import SkipTest
import os

co = coffee().astype(np.float32)
ca = camera().astype(np.float32)[:, :, np.newaxis] * np.ones((1, 1, 3),
                                                             dtype='float32')


def test_googlenet_transformer():
    """smoke test for googlenet transformer"""
    print 'testing the googlenet transformer'
    if os.environ.get('CI', None) is not None:
        raise SkipTest("Skipping heavy data loading on CI")
    t = GoogLeNetTransformer()


    temp=t.transform(co)
    print temp
    print temp.shape
    tempb=t.transform(ca)
    print tempb
    print tempb.shape

def test_googlenet_classifier():
    """smoke test for googlenet classifier"""
    print 'testing the googlenet classifier'
    if os.environ.get('CI', None) is not None:
        raise SkipTest("Skipping heavy data loading on CI")
    c = GoogLeNetClassifier()

    print c.predict(co)

    print c.predict(ca)


if __name__=="__main__":
    test_googlenet_transformer()
    test_googlenet_classifier()
