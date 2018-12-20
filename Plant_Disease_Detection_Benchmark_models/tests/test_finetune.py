import argparse
import unittest
from unittest import mock

import tensorflow as tf
import numpy as np
from tensorflow.python.keras import Input
from tensorflow.python.keras.applications import InceptionV3

from Inception_V3 import utils
from Inception_V3 import finetune
from Inception_V3.finetune import build_finetuned_model


class TestBuildFinetunedModel(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        args = argparse.Namespace()
        args.nb_classes = 3
        args.layers_to_freeze = 3
        cls.args = args

    @mock.patch('Inception_V3.finetune.setup_trainable_layers')
    @mock.patch('Inception_V3.finetune.InceptionV3WithCustomLayers')
    def test_InceptionV3_is_created(self, mock_IV3, _setup):
        build_finetuned_model(self.args)

        # check input_tensor shape is set up right
        mock_IV3.assert_called_with(self.args.nb_classes, input_shape=utils.INPUT_SHAPE)

    @mock.patch('Inception_V3.finetune.setup_trainable_layers')
    @mock.patch('Inception_V3.finetune.InceptionV3WithCustomLayers')
    def test_layers_are_setup_to_be_freezed(self, mock_IV3, mock_setup):
        build_finetuned_model(self.args)

        mock_setup.assert_called_once_with(mock_IV3.return_value, self.args.layers_to_freeze)

    @mock.patch('Inception_V3.finetune.setup_trainable_layers')
    @mock.patch('Inception_V3.finetune.InceptionV3WithCustomLayers')
    def test_InceptionV3_is_compiled(self, mock_IV3, _setup):
        build_finetuned_model(self.args)

        self.assertTrue(mock_IV3.return_value.compile.called)


class TestCustomInceptionV3WithLayers(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        args = argparse.Namespace()
        args.nb_classes = 3
        args.layers_to_freeze = 3
        cls.args = args

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


class TestTrainableLayersSetup(unittest.TestCase):
    def setUp(self):
        self.model = InceptionV3(input_tensor=Input(shape=utils.INPUT_SHAPE), weights=None,
                                 include_top=True, classes=3)
        self.layers_bool = []
        bool = True
        for layer in self.model.layers:
            layer.trainable = bool
            self.layers_bool.append(layer.trainable)
            # switch true and false every iteration
            bool = not bool

    def test_no_effect_on_layers(self):
        finetune.setup_trainable_layers(self.model)

        # extract layers trainable boooleans in to separate list
        affected_layers_bool = []
        for layer in self.model.layers:
            affected_layers_bool.append(layer.trainable)

        self.assertListEqual(self.layers_bool, affected_layers_bool)

    def test_first_specified_number_of_layers_are_freezed(self):
        nb_layers = len(self.model.layers) // 2
        finetune.setup_trainable_layers(self.model, nb_layers)

        # extract layers trainable boooleans in to separate list
        affected_layers_bool = []
        for layer in self.model.layers:
            affected_layers_bool.append(layer.trainable)

        # check layer trainable booleans prior to specified index are changed to False
        self.assertListEqual(affected_layers_bool[:nb_layers], [False for i in range(nb_layers)])

    def test_layers_after_specified_number_of_layers_are_not_freezed(self):
        nb_layers = len(self.model.layers) // 2
        finetune.setup_trainable_layers(self.model, nb_layers)

        # extract layers trainable boooleans in to separate list
        affected_layers_bool = []
        for layer in self.model.layers:
            affected_layers_bool.append(layer.trainable)

        # check layer trainable booleans prior to specified index are changed to False
        self.assertTrue(all(affected_layers_bool[nb_layers:]))

    def test_no_layers_are_freezing_when_number_of_specified_layers_is_0(self):
        nb_layers = 0
        finetune.setup_trainable_layers(self.model, nb_layers)

        # extract layers trainable boooleans in to separate list
        affected_layers_bool = []
        for layer in self.model.layers:
            affected_layers_bool.append(layer.trainable)

        # check layer trainable booleans are all set to True
        self.assertTrue(all(affected_layers_bool))

    def test_all_layers_are_freezing_when_number_of_layers_is_same_as_model_length(self):
        nb_layers = len(self.model.layers)
        finetune.setup_trainable_layers(self.model, nb_layers)

        # extract layers trainable boooleans in to separate list
        affected_layers_bool = []
        for layer in self.model.layers:
            affected_layers_bool.append(layer.trainable)

        # check layer trainable booleans prior to specified index are changed to False
        self.assertListEqual(affected_layers_bool, [False for i in range(nb_layers)])

    def test_raises_valueError_when_number_of_layers_specified_is_greater_than_available_layers(self):
        nb_layers = len(self.model.layers) + 1
        with self.assertRaises(ValueError) as ve:
            finetune.setup_trainable_layers(self.model, nb_layers)

    def test_raises_valueError_when_number_of_layers_specified_is_negative(self):
        nb_layers = -1
        with self.assertRaises(ValueError) as ve:
            finetune.setup_trainable_layers(self.model, nb_layers)


if __name__ == '__main__':
    unittest.main()
