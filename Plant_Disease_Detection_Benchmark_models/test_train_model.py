import argparse
import unittest
from unittest import mock

import train_model

from shared import utils


class TestMain(unittest.TestCase):
    @mock.patch('train_model.setup_args')
    @mock.patch('train_model.get_model')
    @mock.patch('train_model.train_model')
    def test_main_code_is_behaving_as_expected(self, m_train_model, m_get_model, m_setup_args):
        train_model.main()

        m_setup_args.assert_called_once()
        m_get_model.assert_called_once_with(m_setup_args.return_value, utils.INPUT_SHAPE)
        m_train_model.assert_called_once_with(m_get_model.return_value, m_setup_args.return_value)


class TestGetModel(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        args = argparse.Namespace()
        args.model_name = 'mock_model'
        args.model_type = utils.INCEPTIONV3_ARCHITECTURE
        args.model_mode = utils.CUSTOM
        args.nb_classes = 3
        args.batch_size = 12
        args.epochs = 10
        args.augment = False
        args.layers_to_freeze = 2
        cls.args = args

        cls.unimplemented_exceptions = {
            utils.VGG_ARCHITECTURE: {utils.BASELINE},
            utils.RESNET_ARCHITECTURE: {utils.BASELINE, utils.FINETUNE},
        }

    def setUp(self):
        self.args.model_type = utils.INCEPTIONV3_ARCHITECTURE
        self.args.model_mode = utils.CUSTOM

    def get_unimplemented_modes_for_model_type(self, model_type):
        """Helper method to get unimplemented mode exceptions for a model type"""
        model_type_unimplemented_exceptions = self.unimplemented_exceptions.get(model_type)
        return model_type_unimplemented_exceptions if model_type_unimplemented_exceptions is not None else {}

    def test_raises_valueError_when_unsupported_model_type_is_given(self):
        unsupported_model_type = 'xx'
        self.args.model_type = unsupported_model_type
        if unsupported_model_type in utils.SUPPORTED_MODEL_TYPES:
            self.fail('Test condition not right: Write the test again with proper unsupported_model_type string')
        with self.assertRaises(ValueError) as ve:
            train_model.get_model(self.args, utils.INPUT_SHAPE)

    def test_valueError_not_raised_when_supported_model_type_is_given(self):
        try:
            for model_type in utils.SUPPORTED_MODEL_TYPES:
                self.args.model_type = model_type
                if self.args.model_mode not in self.get_unimplemented_modes_for_model_type(self.args.model_type):
                    train_model.get_model(self.args, utils.INPUT_SHAPE)
        except ValueError:
            self.fail(
                '{} model architecture should be supported and should not raise error'.format(self.args.model_type))

    def test_raises_valueError_when_unsupported_model_mode_is_given(self):
        unsupported_model_mode = 'xx'
        self.args.model_mode = unsupported_model_mode
        for supported_mode_type in utils.SUPPORTED_MODEL_TYPES:
            self.args.model_type = supported_mode_type
            if unsupported_model_mode in utils.SUPPORTED_MODEL_MODES:
                self.fail('Test condition not right: Write the test again with proper unsupported_model_mode string')
            with self.assertRaises(ValueError) as ve:
                train_model.get_model(self.args, utils.INPUT_SHAPE)

    def test_valueError_not_raised_when_supported_model_mode_is_given(self):
        try:
            for supported_mode_type in utils.SUPPORTED_MODEL_TYPES:
                self.args.model_type = supported_mode_type
                for model_mode in utils.SUPPORTED_MODEL_MODES:
                    self.args.model_mode = model_mode
                    model_type_unimplemented_exceptions = self.unimplemented_exceptions.get(self.args.model_type)
                    if model_type_unimplemented_exceptions is not None \
                            and self.args.model_mode not in model_type_unimplemented_exceptions:
                        train_model.get_model(self.args, utils.INPUT_SHAPE)
        except ValueError:
            self.fail(
                '{} model architecture should be supported and should not raise error'.format(self.args.model_type))

    @mock.patch('train_model.Inception_V3.build_baseline_model')
    @mock.patch('train_model.Inception_V3.build_finetuned_model')
    @mock.patch('train_model.Inception_V3.build_custom_model')
    def test_inceptionv3_architecture_module_is_used_for_inceptionv3(self, m_custom, m_finetune, m_baseline):
        self.args.model_type = utils.INCEPTIONV3_ARCHITECTURE
        for supported_mode_type in utils.SUPPORTED_MODEL_MODES:
            self.args.model_mode = supported_mode_type
            if self.args.model_mode not in self.get_unimplemented_modes_for_model_type(self.args.model_type):
                train_model.get_model(self.args, utils.INPUT_SHAPE)
                self.assertTrue(any([m_custom.called, m_baseline.called, m_finetune.called]))

    @mock.patch('train_model.VGG.build_finetuned_model')
    @mock.patch('train_model.VGG.build_custom_model')
    def test_vgg_architecture_module_is_used_for_vgg(self, m_custom, m_finetune):
        self.args.model_type = utils.VGG_ARCHITECTURE
        for supported_mode_type in utils.SUPPORTED_MODEL_MODES:
            self.args.model_mode = supported_mode_type
            if self.args.model_mode not in self.get_unimplemented_modes_for_model_type(self.args.model_type):
                train_model.get_model(self.args, utils.INPUT_SHAPE)
                self.assertTrue(any([m_custom.called, m_finetune.called]))

    @mock.patch('train_model.ResNet.build_custom_model')
    def test_resnet_architecture_module_is_used_for_resnet(self, m_custom):
        self.args.model_type = utils.RESNET_ARCHITECTURE
        for supported_mode_type in utils.SUPPORTED_MODEL_MODES:
            self.args.model_mode = supported_mode_type
            if self.args.model_mode not in self.get_unimplemented_modes_for_model_type(self.args.model_type):
                train_model.get_model(self.args, utils.INPUT_SHAPE)
                self.assertTrue(any([m_custom.called]))


if __name__ == '__main__':
    unittest.main()
