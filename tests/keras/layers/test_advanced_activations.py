import pytest
from bidu.utils.test_utils import layer_test, bidu_test


@bidu_test
def test_leaky_relu():
    from bidu.layers.advanced_activations import LeakyReLU
    for alpha in [0., .5, -1.]:
        layer_test(LeakyReLU, kwargs={'alpha': alpha},
                   input_shape=(2, 3, 4))


@bidu_test
def test_prelu():
    from bidu.layers.advanced_activations import PReLU
    layer_test(PReLU, kwargs={},
               input_shape=(2, 3, 4))


@bidu_test
def test_elu():
    from bidu.layers.advanced_activations import ELU
    for alpha in [0., .5, -1.]:
        layer_test(ELU, kwargs={'alpha': alpha},
                   input_shape=(2, 3, 4))


@bidu_test
def test_parametric_softplus():
    from bidu.layers.advanced_activations import ParametricSoftplus
    for alpha in [0., .5, -1.]:
        layer_test(ParametricSoftplus,
                   kwargs={'alpha_init': 1.,
                           'beta_init': -1},
                   input_shape=(2, 3, 4))


@bidu_test
def test_thresholded_relu():
    from bidu.layers.advanced_activations import ThresholdedReLU
    layer_test(ThresholdedReLU, kwargs={'theta': 0.5},
               input_shape=(2, 3, 4))


@bidu_test
def test_srelu():
    from bidu.layers.advanced_activations import SReLU
    layer_test(SReLU, kwargs={},
               input_shape=(2, 3, 4))


if __name__ == '__main__':
    pytest.main([__file__])
