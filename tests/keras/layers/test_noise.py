import pytest
from bidu.utils.test_utils import layer_test, bidu_test
from bidu.layers import noise


@bidu_test
def test_GaussianNoise():
    layer_test(noise.GaussianNoise,
               kwargs={'sigma': 1.},
               input_shape=(3, 2, 3))


@bidu_test
def test_GaussianDropout():
    layer_test(noise.GaussianDropout,
               kwargs={'p': 0.5},
               input_shape=(3, 2, 3))


if __name__ == '__main__':
    pytest.main([__file__])
