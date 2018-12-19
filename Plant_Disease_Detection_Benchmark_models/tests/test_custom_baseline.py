import argparse
import unittest
from unittest import mock

import tensorflow as tf
import numpy as np

from Inception_V3 import utils
from Inception_V3 import custom_baseline
from Inception_V3.custom_baseline import build_baseline_model


class TestCustomInceptionV3(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        args = argparse.Namespace()
        args.nb_classes = 3
        cls.args = args

    @mock.patch('Inception_V3.custom_baseline.Inceptionv3')
    def test_InceptionV3_is_created(self, mock_IV3):
        build_baseline_model(self.args)

        # check input_tensor shape is set up right
        mock_IV3.assert_called_with(self.args.nb_classes, input_shape=utils.INPUT_SHAPE)

    @mock.patch('Inception_V3.custom_baseline.Inceptionv3')
    def test_InceptionV3_is_compiled(self, mock_IV3):
        build_baseline_model(self.args)

        self.assertTrue(mock_IV3.return_value.compile.called)

    @mock.patch('Inception_V3.custom_baseline.Activation')
    @mock.patch('Inception_V3.custom_baseline.BatchNormalization')
    @mock.patch('Inception_V3.custom_baseline.Conv2D')
    def test_conv2d_bn_uses_necessary_layers_with_proper_arguments(self, m_conv2d, m_batch_norm, m_activation):
        x = np.random.rand(100, 200, 3)
        filters = 12
        kernel_size = (3, 3)
        padding = 'valid'
        strides = (2, 1)
        name = '01'

        output = custom_baseline.conv2d_bn(x, filters, kernel_size, padding, strides, name)

        # check conv2d is working as intended
        m_conv2d.assert_called_once_with(
            filters, kernel_size, strides=strides, padding=padding, use_bias=False, name='01_conv'
        )
        m_conv2d.return_value.assert_called_once_with(x)
        m_conv2d_output = m_conv2d.return_value.return_value

        # check batchnorm is working as intended
        m_batch_norm.assert_called_once_with(scale=False, name='01_bn')
        m_batch_norm.return_value.assert_called_once_with(m_conv2d_output)
        m_batch_norm_output = m_batch_norm.return_value.return_value

        # check relu activation
        m_activation.assert_called_once_with('relu', name='01')
        m_activation.return_value.assert_called_once_with(m_batch_norm_output)

        # check the output tensor is the relu activation output
        self.assertEqual(m_activation.return_value.return_value, output)

    @mock.patch('Inception_V3.custom_baseline.Dense')
    def test_custom_inceptionv3_has_dense_layer_at_last(self, m_dense):
        nb_classes = 3
        input_shape = (200, 200, 3)

        model = custom_baseline.Inceptionv3(nb_classes, input_shape=input_shape)
        m_dense.assert_called_once_with(nb_classes, activation='softmax')

        self.assertEqual(model.layers[-1], m_dense.return_value)

    @mock.patch('Inception_V3.custom_baseline.Dense')
    @mock.patch('Inception_V3.custom_baseline.Input')
    @mock.patch('Inception_V3.custom_baseline.Model')
    def test_model_is_build_with_appropriate_inputs_and_outputs(self, m_model, m_input, m_dense, ):
        nb_classes = 3
        input_shape = (200, 200, 3)

        model = custom_baseline.Inceptionv3(nb_classes, input_shape=input_shape)

        # input tensor is setup correctly
        m_input.assert_called_once_with(shape=input_shape)
        input = m_input.return_value

        # output tensor is from dense layer
        output = m_dense.return_value.return_value

        # check model is built correctly
        m_model.assert_called_once_with(inputs=input, outputs=output)
        self.assertEqual(m_model.return_value, model)


if __name__ == '__main__':
    unittest.main()
