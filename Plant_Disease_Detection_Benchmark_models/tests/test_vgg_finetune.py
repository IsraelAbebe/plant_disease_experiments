import argparse
import unittest
from unittest import mock

import tensorflow as tf
import numpy as np
from tensorflow.python.keras import Model

import shared.utils as utils

from VGG import finetune
from VGG.finetune import build_finetuned_model

mock_model = mock.create_autospec(Model)
mock_base_model = mock.create_autospec(Model)
mock_models = [mock_model, mock_base_model]


class TestBuildFinetunedModel(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        args = argparse.Namespace()
        args.nb_classes = 3
        args.layers_to_freeze = 3
        cls.args = args

    @mock.patch('VGG.finetune.setup_trainable_layers')
    @mock.patch('VGG.finetune.VGGWithCustomLayers', return_value=mock_models)
    def test_vgg_is_created(self, mock_VGG, _setup):
        build_finetuned_model(self.args, utils.INPUT_SHAPE, utils.FC_SIZE)

        mock_VGG.assert_called_with(self.args.nb_classes, utils.INPUT_SHAPE, utils.FC_SIZE)

    @mock.patch('VGG.finetune.setup_trainable_layers')
    @mock.patch('VGG.finetune.VGGWithCustomLayers', return_value=mock_models)
    def test_layers_are_setup_to_be_freezed(self, mock_VGG, mock_setup):
        build_finetuned_model(self.args, utils.INPUT_SHAPE, utils.FC_SIZE)

        mock_setup.assert_called_once_with(mock_VGG.return_value[0], self.args.layers_to_freeze)

    @mock.patch('VGG.finetune.setup_trainable_layers')
    @mock.patch('VGG.finetune.VGGWithCustomLayers', return_value=mock_models)
    def test_vgg_is_compiled(self, mock_VGG, _setup):
        build_finetuned_model(self.args, utils.INPUT_SHAPE, utils.FC_SIZE)

        self.assertTrue(mock_VGG.return_value[0].compile.called)


class TestVGGWithCustomLayers(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        args = argparse.Namespace()
        args.nb_classes = 3
        args.layers_to_freeze = 3
        cls.args = args

    @mock.patch('VGG.finetune.Model')
    @mock.patch('VGG.finetune.Dense')
    @mock.patch('VGG.finetune.GlobalAveragePooling2D')
    @mock.patch('VGG.finetune.Dropout')
    @mock.patch('VGG.finetune.Input')
    @mock.patch('VGG.finetune.VGGFace')
    def test_vgg_finetune_uses_library_vggface_as_base(
            self, m_vggface, m_input, _drop, _glob_avg, _dense, _model
    ):
        finetune.VGGWithCustomLayers(self.args.nb_classes, input_shape=utils.INPUT_SHAPE, fc_size=utils.FC_SIZE)

        # check input is setup with proper shape
        m_input.assert_called_once_with(shape=utils.INPUT_SHAPE)
        m_vggface.assert_called_once_with(input_tensor=m_input.return_value, include_top=False)

    @mock.patch('VGG.finetune.Dropout')
    @mock.patch('VGG.finetune.Model')
    @mock.patch('VGG.finetune.Dense')
    def test_finetune_VGG_has_last_dense_layer(self, m_dense, m_model, _drop):
        finetune.VGGWithCustomLayers(self.args.nb_classes, input_shape=utils.INPUT_SHAPE, fc_size=utils.FC_SIZE)
        m_dense.assert_called_with(self.args.nb_classes, activation='softmax')

    @mock.patch('VGG.finetune.Dropout')
    @mock.patch('VGG.finetune.GlobalAveragePooling2D')
    @mock.patch('VGG.finetune.Model')
    @mock.patch('VGG.finetune.Dense')
    @mock.patch('VGG.finetune.VGGFace')
    def test_finetune_vgg_is_created_with_appropriate_inputs_and_outputs(self, m_vgg, m_dense, m_model,
                                                                         _glob_avg, _drop):
        vgg, base_vgg = finetune.VGGWithCustomLayers(self.args.nb_classes, input_shape=utils.INPUT_SHAPE,
                                                     fc_size=utils.FC_SIZE)

        m_model.assert_called_once_with(inputs=m_vgg.return_value.input, outputs=m_dense.return_value.return_value)
        self.assertEqual(m_model.return_value, vgg)


if __name__ == '__main__':
    unittest.main()
