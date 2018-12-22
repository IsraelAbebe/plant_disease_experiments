import argparse
import unittest
from unittest import mock

import tensorflow as tf
import numpy as np

import shared.utils as utils

from Inception_V3 import custom_baseline
from Inception_V3.custom_baseline import build_custom_model


class TestCustomInceptionV3Model(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        args = argparse.Namespace()
        args.nb_classes = 3
        cls.args = args

    @mock.patch('Inception_V3.custom_baseline.Inceptionv3')
    def test_InceptionV3_is_created(self, mock_IV3):
        build_custom_model(self.args, utils.INPUT_SHAPE)

        # check input_tensor shape is set up right
        mock_IV3.assert_called_with(self.args.nb_classes, input_shape=utils.INPUT_SHAPE)

    @mock.patch('Inception_V3.custom_baseline.Inceptionv3')
    def test_InceptionV3_is_compiled(self, mock_IV3):
        build_custom_model(self.args, utils.INPUT_SHAPE)

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

    def test_custom_inceptionv3_raises_valueError_when_either_input_tensor_or_shape_are_not_provided(self):
        with self.assertRaises(ValueError) as ve:
            custom_baseline.Inceptionv3(self.args.nb_classes)

    @mock.patch('Inception_V3.custom_baseline.Model')
    @mock.patch('Inception_V3.custom_baseline.Dense')
    def test_custom_inceptionv3_has_dense_layer(self, m_dense, m_model):
        custom_baseline.Inceptionv3(self.args.nb_classes, input_shape=utils.INPUT_SHAPE)
        m_dense.assert_called_once_with(self.args.nb_classes, activation='softmax')

    @mock.patch('Inception_V3.custom_baseline.Flatten')
    @mock.patch('Inception_V3.custom_baseline.GlobalAveragePooling2D')
    @mock.patch('Inception_V3.custom_baseline.concatenate')
    @mock.patch('Inception_V3.custom_baseline.MaxPooling2D')
    @mock.patch('Inception_V3.custom_baseline.AveragePooling2D')
    @mock.patch('Inception_V3.custom_baseline.conv2d_bn')
    @mock.patch('Inception_V3.custom_baseline.Dense')
    @mock.patch('Inception_V3.custom_baseline.Input')
    @mock.patch('Inception_V3.custom_baseline.Model')
    def test_model_is_created_with_appropriate_inputs_and_outputs(
            self, m_model, m_input, m_dense, _conv2d_bn, _avg, _max, _concat, _glob_avg, _flat
    ):
        model = custom_baseline.Inceptionv3(self.args.nb_classes, input_shape=utils.INPUT_SHAPE)

        # input tensor is setup correctly
        m_input.assert_called_once_with(shape=utils.INPUT_SHAPE)
        input = m_input.return_value

        # output tensor is from dense layer
        output = m_dense.return_value.return_value

        # check model is created correctly
        m_model.assert_called_once_with(inputs=input, outputs=output)
        self.assertEqual(m_model.return_value, model)


if __name__ == '__main__':
    unittest.main()
