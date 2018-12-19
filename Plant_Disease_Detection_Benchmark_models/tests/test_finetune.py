import argparse
import unittest
from unittest import mock

import tensorflow as tf
import numpy as np

from Inception_V3 import utils
from Inception_V3 import finetune
from Inception_V3.finetune import build_baseline_model


class TestCustomInceptionV3WithLayers(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        args = argparse.Namespace()
        args.nb_classes = 3
        cls.args = args

    @mock.patch('Inception_V3.finetune.InceptionV3WithCustomLayers')
    def test_InceptionV3_is_created(self, mock_IV3):
        build_baseline_model(self.args)

        # check input_tensor shape is set up right
        mock_IV3.assert_called_with(self.args.nb_classes, input_shape=utils.INPUT_SHAPE)

    @mock.patch('Inception_V3.finetune.setup_trainable_layers')
    @mock.patch('Inception_V3.finetune.InceptionV3WithCustomLayers')
    def test_layers_are_setup_to_be_freezed(self, mock_IV3, mock_setup):
        build_baseline_model(self.args)

        mock_setup.assert_called_once_with(mock_IV3.return_value)

    @mock.patch('Inception_V3.finetune.InceptionV3WithCustomLayers')
    def test_InceptionV3_is_compiled(self, mock_IV3):
        build_baseline_model(self.args)

        self.assertTrue(mock_IV3.return_value.compile.called)

    @mock.patch('Inception_V3.finetune.Model')
    @mock.patch('Inception_V3.finetune.Dense')
    @mock.patch('Inception_V3.finetune.GlobalAveragePooling2D')
    @mock.patch('Inception_V3.finetune.Dropout')
    @mock.patch('Inception_V3.finetune.Input')
    @mock.patch('Inception_V3.finetune.InceptionV3')
    def test_finetune_inceptionv3_uses_library_inceptionv3_as_base(
            self, m_iv3, m_input, _drop, _glob_avg, _dense, _model
    ):
        finetune.InceptionV3WithCustomLayers(self.args.nb_classes, input_shape=utils.INPUT_SHAPE)

        # check input is setup with proper shape
        m_input.assert_called_once_with(shape=utils.INPUT_SHAPE)
        m_iv3.assert_called_once_with(input_tensor=m_input.return_value, weights='imagenet', include_top=False)

    @mock.patch('Inception_V3.finetune.Dropout')
    @mock.patch('Inception_V3.finetune.Model')
    @mock.patch('Inception_V3.finetune.Dense')
    def test_finetune_inceptionv3_has_dense_layer(self, m_dense, m_model, _drop):
        finetune.InceptionV3WithCustomLayers(self.args.nb_classes, input_shape=utils.INPUT_SHAPE)
        m_dense.assert_called_with(self.args.nb_classes, activation='softmax')

    @mock.patch('Inception_V3.finetune.Dropout')
    @mock.patch('Inception_V3.finetune.GlobalAveragePooling2D')
    @mock.patch('Inception_V3.finetune.Model')
    @mock.patch('Inception_V3.finetune.Dense')
    @mock.patch('Inception_V3.finetune.InceptionV3')
    def test_finetune_inceptionv3_is_created_with_appropriate_inputs_and_outputs(self, m_iv3, m_dense, m_model,
                                                                                 _glob_avg, _drop):
        iv3 = finetune.InceptionV3WithCustomLayers(self.args.nb_classes, input_shape=utils.INPUT_SHAPE)

        m_model.assert_called_once_with(inputs=m_iv3.return_value.input, outputs=m_dense.return_value.return_value)
        self.assertEqual(m_model.return_value, iv3)

    def test_trainable_layers(self):
        pass

if __name__ == '__main__':
    unittest.main()
