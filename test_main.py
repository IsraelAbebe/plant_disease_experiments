import io
import os
import random
import sys
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
        cls.supported_species = next(iter(main.DISEASE_SUPPORTED_SPECIES))

        if cls.supported_species not in main.DISEASE_SUPPORTED_SPECIES:
            raise ValueError("supported species is not setup right in unit test\n"
                             "Please, Write unit test condition again with appropriate supported species")

        if cls.unsupported_species in main.DISEASE_SUPPORTED_SPECIES:
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
        mock_parser.add_argument.assert_any_call('--model', type=str.lower, default=VGG_ARCHITECTURE,
                                                 choices=[VGG_ARCHITECTURE, main.INCEPTIONV3_ARCHITECTURE],
                                                 help=mock.ANY)
        mock_parser.add_argument.assert_any_call('--segment', action='store_true', help=mock.ANY)
        mock_parser.add_argument.assert_any_call('--species', type=str.lower, default='', help=mock.ANY)

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
            for supported_species in main.DISEASE_SUPPORTED_SPECIES:
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

    @mock.patch('main.image')
    @mock.patch('main.load_model')
    @mock.patch('main.np')
    @mock.patch('main.Image')
    @mock.patch('main.os.path')
    def test_get_predictions_raises_error_if_model_file_doesnot_exist_only(self, mock_path, _Image, _np, _load, _image):
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

    @mock.patch('main.image')
    @mock.patch('main.load_model')
    @mock.patch('main.np')
    @mock.patch('main.Image')
    @mock.patch('main.os.path')
    def test_get_predictions_uses_model_appropriately(self, mock_path, _Image, _np, _load, _image):
        mock_path.exists.return_value = True
        main.get_predictions(self.model_path, self.img_path, self.target_size)

        _load.assert_called_once_with(self.model_path)

        _load.return_value.predict.assert_called_once()

    @mock.patch('main.image')
    @mock.patch('main.load_model')
    @mock.patch('main.np')
    @mock.patch('main.Image')
    @mock.patch('main.os.path')
    def test_get_predictions_loads_image_appropriately(self, mock_path, _Image, _np, _load, _image):
        mock_path.exists.return_value = True
        _Image.open.return_value.size = (6, 6)

        main.get_predictions(self.model_path, self.img_path, self.target_size)

        _Image.open.assert_called_once_with(self.img_path)
        _Image.open.return_value.resize.assert_called_once_with(self.target_size)

        _image.img_to_array.assert_called_once_with(_Image.open.return_value.resize.return_value)

    @mock.patch('main.preprocess_input')
    @mock.patch('main.image')
    @mock.patch('main.load_model')
    @mock.patch('main.np')
    @mock.patch('main.Image')
    @mock.patch('main.os.path')
    def test_get_predictions_input_is_preprocessed(self, mock_path, _Image, _np, _load, _image, _preprocess):
        mock_path.exists.return_value = True
        main.get_predictions(self.model_path, self.img_path, self.target_size)

        _preprocess.assert_called_once_with(_np.expand_dims.return_value)
        _load.return_value.predict.assert_called_once_with(_preprocess.return_value)

    @mock.patch('main.preprocess_input')
    @mock.patch('main.image')
    @mock.patch('main.load_model')
    @mock.patch('main.np')
    @mock.patch('main.Image')
    @mock.patch('main.os.path')
    def test_get_predictions_returns_right_preds_and_its_sorting_index(self, mock_path, _Image, _np, _load, _image,
                                                                       _preprocess):
        mock_path.exists.return_value = True
        expected_preds = np.array([2, 3, 1])
        _load.return_value.predict.return_value.flatten.return_value = expected_preds
        preds, sorrting_index = main.get_predictions(self.model_path, self.img_path, self.target_size)

        expected_sorting_index = np.array([1, 0, 2])
        np.testing.assert_array_equal(preds, expected_preds)
        np.testing.assert_array_equal(sorrting_index, expected_sorting_index)


class TestSegmentImage(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.img_path = 'img_path.jpg'
        cls.segmented_img_path = 'img_path_marked.jpg'

    @mock.patch('main.subprocess')
    def test_segment_image_returns_the_right_file_name(self, _subprocess):
        result_img_path = main.segment_image(self.img_path)

        self.assertEqual(self.segmented_img_path, result_img_path)

    @mock.patch('main.subprocess')
    def test_segment_image_segments_given_image(self, _subprocess):
        result_img_path = main.segment_image(self.img_path)

        _subprocess.check_output(['python', "leaf-image-segmentation/segment.py", "-s", self.img_path])


class TestPipelines(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.supported_species = main.APPLE
        cls.unsupported_species = 'xx'
        cls.species_model = 'apple.h5'
        cls.disease_model = 'healthy.h5'
        cls.default_model_type = VGG_ARCHITECTURE
        cls.model_path = 'model_path'
        cls.img_path = 'img_path.jpg'
        cls.segmented_img_path = 'img_path_marked.jpg'
        cls.target_size = (5, 5)
        cls.preds = np.array([2, 3, 1])
        cls.sorting_index = np.array([1, 0, 2])

        random.seed(0)

        if cls.unsupported_species in main.DISEASE_SUPPORTED_SPECIES:
            raise ValueError("unsupported species is not setup right in unit test\n"
                             "Please, Write unit test condition again with appropriate unsupported species")

        if cls.supported_species not in main.DISEASE_SUPPORTED_SPECIES:
            raise ValueError("supported species is not setup right in unit test\n"
                             "Please, Write unit test condition again with appropriate supported species")

        if len(cls.sorting_index) > len(main.APPLE_CLASSES):
            raise ValueError("used species classes and sorting index length is not compatible\n"
                             "Please, Write unit test condition again with appropriate length")

        if len(cls.sorting_index) != len(cls.preds):
            raise ValueError("preds and sorting index should be equal length since sorting index sorts preds\n"
                             "Please, Write unit test condition again with appropriate length and items")

    def setUp(self):
        self.old_stdout = sys.stdout

    def tearDown(self):
        # restore stdout to console
        sys.stdout = self.old_stdout

    @mock.patch('main.get_predictions')
    @mock.patch('main.get_species_model')
    @mock.patch('main.segment_image')
    def test_segment_and_predict_species_image_is_segmented(self, _segment_image, _get_species_model, _get_predictions):
        _get_species_model.return_value = self.species_model
        _get_predictions.return_value = self.preds, self.sorting_index

        main.segment_and_predict_species(self.img_path, self.default_model_type, False)

        _segment_image.assert_called_once_with(self.img_path)

    @mock.patch('main.get_predictions')
    @mock.patch('main.get_species_model')
    @mock.patch('main.segment_image')
    def test_segment_and_predict_species_loads_correct_model_and_segmented_image_with_right_size(self, _segment_image,
                                                                                                 _get_species_model,
                                                                                                 _get_predictions):
        _get_species_model.return_value = self.species_model
        _get_predictions.return_value = self.preds, self.sorting_index
        _segment_image.return_value = self.segmented_img_path
        model_path = os.path.join(main.MODEL_STORAGE_BASE, self.species_model)

        for model_type in main.SUPPORTED_MODEL_TYPES:
            main.segment_and_predict_species(self.img_path, model_type, False)

            target_img_size = main.TARGET_IMAGE_SIZES[model_type][main.SPECIES_DETECTION]
            _get_predictions.assert_called_with(model_path, self.segmented_img_path, target_img_size)

    @mock.patch('main.get_predictions')
    @mock.patch('main.get_species_model')
    @mock.patch('main.segment_image')
    def test_segment_and_predict_species_returns_what_is_expected(self, _segment_image, _get_species_model,
                                                                  _get_predictions):
        _get_species_model.return_value = self.species_model
        _get_predictions.return_value = self.preds, self.sorting_index
        _segment_image.return_value = self.segmented_img_path

        top_species, segmented_image_name = main.segment_and_predict_species(self.img_path, self.default_model_type,
                                                                             False)

        self.assertEqual(top_species, main.SPECIES[self.sorting_index[0]])
        self.assertEqual(segmented_image_name, self.segmented_img_path)

    @mock.patch('main.get_predictions')
    @mock.patch('main.get_species_model')
    @mock.patch('main.segment_image')
    def test_segment_and_predict_species_prints_the_right_result(self, _segment_image, _get_species_model,
                                                                 _get_predictions):
        _get_species_model.return_value = self.species_model
        _get_predictions.return_value = self.preds, self.sorting_index
        out_string = io.StringIO()
        sys.stdout = out_string

        top_species, segmented_image_name = main.segment_and_predict_species(self.img_path, self.default_model_type,
                                                                             True)

        # check one random item from a list of printed results
        random_i = random.choice(self.sorting_index)
        printed_content = out_string.getvalue()
        self.assertIn(str(main.SPECIES[random_i]), printed_content)
        self.assertIn(str(self.preds[random_i]), printed_content)

    @mock.patch('main.get_predictions')
    @mock.patch('main.get_species_model')
    def test_predict_species_loads_correct_model_and_image_with_right_size(self, _get_species_model,
                                                                           _get_predictions):
        _get_species_model.return_value = self.species_model
        _get_predictions.return_value = self.preds, self.sorting_index
        model_path = os.path.join(main.MODEL_STORAGE_BASE, self.species_model)

        for model_type in main.SUPPORTED_MODEL_TYPES:
            main.predict_species(self.img_path, model_type, False)

            target_img_size = main.TARGET_IMAGE_SIZES[model_type][main.SPECIES_DETECTION]
            _get_predictions.assert_called_with(model_path, self.img_path, target_img_size)

    @mock.patch('main.get_predictions')
    @mock.patch('main.get_species_model')
    def test_predict_species_returns_what_is_expected(self, _get_species_model,
                                                      _get_predictions):
        _get_species_model.return_value = self.species_model
        _get_predictions.return_value = self.preds, self.sorting_index

        top_species = main.predict_species(self.img_path, self.default_model_type, False)

        self.assertEqual(top_species, main.SPECIES[self.sorting_index[0]])

    @mock.patch('main.get_predictions')
    @mock.patch('main.get_species_model')
    def test_predict_species_prints_the_right_result(self, _get_species_model,
                                                     _get_predictions):
        _get_species_model.return_value = self.species_model
        _get_predictions.return_value = self.preds, self.sorting_index
        out_string = io.StringIO()
        sys.stdout = out_string

        main.predict_species(self.img_path, self.default_model_type, True)

        # check one random item from a list of printed results
        random_i = random.choice(self.sorting_index)
        printed_content = out_string.getvalue()
        self.assertIn(str(main.SPECIES[random_i]), printed_content)
        self.assertIn(str(self.preds[random_i]), printed_content)

    @mock.patch('main.get_predictions')
    @mock.patch('main.get_disease_model')
    def test_predict_disease_loads_correct_model_and_image_with_right_size(self, _get_disease_model, _get_predictions):
        _get_disease_model.return_value = self.disease_model
        _get_predictions.return_value = self.preds, self.sorting_index
        model_path = os.path.join(main.MODEL_STORAGE_BASE, self.disease_model)

        for model_type in main.SUPPORTED_MODEL_TYPES:
            main.predict_disease(self.img_path, self.supported_species, model_type, False)

            target_img_size = main.TARGET_IMAGE_SIZES[model_type][main.DISEASE_DETECTION]
            _get_predictions.assert_called_with(model_path, self.img_path, target_img_size)

    @mock.patch('main.get_classes')
    @mock.patch('main.get_disease_model')
    @mock.patch('main.get_predictions')
    def test_predict_disease_uses_appropriate_species_class_and_returns_proper_element_from_it(self, _get_predictions,
                                                                                               _get_disease_model,
                                                                                               _get_classes):
        _get_disease_model.return_value = self.disease_model
        _get_predictions.return_value = self.preds, self.sorting_index
        _get_classes.return_value = main.APPLE_CLASSES

        top_disease = main.predict_disease(self.img_path, self.supported_species, self.default_model_type, False)

        _get_classes.assert_called_with(self.supported_species)
        self.assertEqual(top_disease, main.APPLE_CLASSES[self.sorting_index[0]])

    @mock.patch('main.get_classes')
    @mock.patch('main.get_disease_model')
    @mock.patch('main.get_predictions')
    def test_predict_disease_prints_the_right_thing(self, _get_predictions, _get_disease_model, _get_classes):
        _get_disease_model.return_value = self.disease_model
        _get_predictions.return_value = self.preds, self.sorting_index
        _get_classes.return_value = main.APPLE_CLASSES
        out_string = io.StringIO()
        sys.stdout = out_string

        main.predict_disease(self.img_path, self.supported_species, self.default_model_type, True)

        # check one random item from a list of printed results
        random_i = random.choice(self.sorting_index)
        printed_content = out_string.getvalue()
        self.assertIn(str(main.APPLE_CLASSES[random_i]), printed_content)
        self.assertIn(str(self.preds[random_i]), printed_content)

    @mock.patch('main.get_classes')
    @mock.patch('main.get_disease_model')
    @mock.patch('main.get_predictions')
    def test_predict_disease_raises_error_if_unsupported_species_is_given(self, _get_predictions,
                                                                          _get_disease_model,
                                                                          _get_classes):
        _get_disease_model.return_value = self.disease_model
        _get_predictions.return_value = self.preds, self.sorting_index
        _get_classes.return_value = main.APPLE_CLASSES

        with self.assertRaises(ValueError) as ve:
            main.predict_disease(self.img_path, self.unsupported_species, False)


if __name__ == '__main__':
    unittest.main()
