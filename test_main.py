import unittest
from unittest import mock

import numpy as np

import main
from main import VGG_ARCHITECTURE


class TestMain(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.unsupported_model_type = 'xx'
        cls.unsupported_species = 'xx'
        cls.supported_species = main.APPLE

        if cls.supported_species not in main.SUPPORTED_SPECIES:
            raise ValueError("supported species is not setup right in unit test\n"
                             "Please, Write unit test condition again with appropriate supported species")

        if cls.unsupported_species in main.SUPPORTED_SPECIES:
            raise ValueError("unsupported species is not setup right in unit test\n"
                             "Please, Write unit test condition again with appropriate unsupported species")

        if cls.unsupported_model_type in main.SUPPORTED_MODEL_TYPES:
            raise ValueError("unsupported model type is not setup right in unit test\n"
                             "Please, Write unit test condition again with appropriate unsupported model type")

    @mock.patch('main.argparse.ArgumentParser', autospec=True)
    def test_get_cmd_args(self, mock_parser_class):
        args = main.get_cmd_args()
        mock_parser = mock_parser_class.return_value

        # check mandatory image cmd arg
        mock_parser.add_argument.assert_any_call('image', type=str, help=mock.ANY)

        # check for optional cmd args
        mock_parser.add_argument.assert_any_call('--model_type', default=VGG_ARCHITECTURE,
                                                 choices=[VGG_ARCHITECTURE, main.INCEPTIONV3_ARCHITECTURE],
                                                 help=mock.ANY)
        mock_parser.add_argument.assert_any_call('--segment', type=bool, default=False, help=mock.ANY)
        mock_parser.add_argument.assert_any_call('--species', type=str, default='', help=mock.ANY)

        # check cmd args are parsed and returned from the function
        self.assertEqual(args, mock_parser.parse_args.return_value)

    def test_get_species_model_raises_error_if_unsupported_model_type_is_given(self):
        with self.assertRaises(ValueError) as ve:
            main.get_species_model(self.unsupported_model_type)

    def test_get_species_model_doesnot_raise_error_if_supported_model_type_is_given(self):
        for supported_model_type in main.SUPPORTED_MODEL_TYPES:
            try:
                main.get_species_model(supported_model_type)
            except ValueError:
                self.fail("`{}` model type should be supported but it is not".format(supported_model_type))

    def test_get_disease_model_raises_error_if_unsupported_model_type_is_given(self):
        with self.assertRaises(ValueError) as ve:
            main.get_disease_model(self.unsupported_species, self.unsupported_model_type)

    def test_get_disease_model_doesnot_raise_error_if_supported_model_type_is_given(self):
        for supported_model_type in main.SUPPORTED_MODEL_TYPES:
            try:
                main.get_disease_model(self.supported_species, supported_model_type)
            except ValueError:
                self.fail("`{}` model type should be supported but it is not".format(supported_model_type))

    def test_get_disease_model_raises_if_unsupported_species_is_given(self):
        for supported_model_type in main.SUPPORTED_MODEL_TYPES:
            with self.assertRaises(ValueError) as ve:
                main.get_disease_model(self.unsupported_species, supported_model_type)

    def test_get_disease_model_doesnot_raise_if_supported_species_is_given(self):
        for supported_model_type in main.SUPPORTED_MODEL_TYPES:
            for supported_species in main.SUPPORTED_SPECIES:
                try:
                    main.get_disease_model(supported_species, supported_model_type)
                except ValueError:
                    self.fail("`{}` species should be supported but it is not".format(supported_species))



class TestGetPredictions(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model_path = 'model_path'
        cls.img_path = 'img_path'
        cls.target_size = (5, 5)

    @mock.patch('main.np')
    @mock.patch('main.image')
    @mock.patch('main.Image')
    @mock.patch('main.load_model')
    @mock.patch('main.os.path')
    def test_get_predictions_raises_error_if_model_file_doesnot_exist_only(self, mock_path, _load, _Image, _image, _np):
        # if model path exist, valueerror should not be raised
        mock_path.exists.return_value = True
        try:
            main.get_predictions(self.model_path, self.img_path, self.target_size)
        except ValueError:
            self.fail("No value error should have been raised while model path existing")

        # if model path does not exist, valueerror should be raised
        mock_path.exists.return_value = False
        with self.assertRaises(ValueError) as ve:
            main.get_predictions('dummy_path', 'dummy_path', self.target_size)

    @mock.patch('main.np')
    @mock.patch('main.image')
    @mock.patch('main.Image')
    @mock.patch('main.load_model')
    @mock.patch('main.os.path')
    def test_get_predictions_uses_model_appropriately(self, mock_path, _load, _Image, _image, _np):
        mock_path.exists.return_value = True
        main.get_predictions(self.model_path, self.img_path, self.target_size)

        _load.assert_called_once_with(self.model_path)

        _load.return_value.predict.assert_called_once()

    @mock.patch('main.np')
    @mock.patch('main.image')
    @mock.patch('main.Image')
    @mock.patch('main.load_model')
    @mock.patch('main.os.path')
    def test_get_predictions_loads_image_appropriately(self, mock_path, _load, _Image, _image, _np):
        mock_path.exists.return_value = True
        img_target_size = (5, 5)
        _Image.open.return_value = np.zeros((6, 6))

        main.get_predictions(self.model_path, self.img_path, self.target_size)

        _Image.open.assert_called_once_with(self.img_path)
        _np.resize.assert_called_once_with(_Image.open.return_value, self.target_size)

        _image.img_to_array.assert_called_once_with(_np.resize.return_value)

    @mock.patch('main.preprocess_input')
    @mock.patch('main.np')
    @mock.patch('main.image')
    @mock.patch('main.Image')
    @mock.patch('main.load_model')
    @mock.patch('main.os.path')
    def test_get_predictions_input_is_preprocessed(self, mock_path, _load, _Image, _image, _np, _preprocess):
        mock_path.exists.return_value = True
        main.get_predictions(self.model_path, self.img_path, self.target_size)

        _preprocess.assert_called_once_with(_np.expand_dims.return_value)
        _load.return_value.predict.assert_called_once_with(_preprocess.return_value)

    @mock.patch('main.preprocess_input')
    @mock.patch('main.np')
    @mock.patch('main.image')
    @mock.patch('main.Image')
    @mock.patch('main.load_model')
    @mock.patch('main.os.path')
    def test_get_predictions_returns_right_preds_and_its_sorting_index(self, mock_path, _load, _Image, _image, _np, _preprocess):
        mock_path.exists.return_value = True
        expected_preds = np.array([2,3,1])
        _load.return_value.predict.return_value.flatten.return_value = expected_preds
        preds, sorrting_index = main.get_predictions(self.model_path, self.img_path, self.target_size)

        expected_sorting_index = np.array([1, 0, 2])
        np.testing.assert_array_equal(preds, expected_preds)
        np.testing.assert_array_equal(sorrting_index, expected_sorting_index)


if __name__ == '__main__':
    unittest.main()
