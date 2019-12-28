import argparse
import unittest
import os
import shutil
import tempfile
from unittest import mock

from tensorflow.python.keras.applications.inception_v3 import preprocess_input, InceptionV3
from tensorflow.python.keras import Model, Input

import shared.utils as utils
from shared.utils import IM_WIDTH, IM_HEIGHT


class TestSimpleFunctions(unittest.TestCase):

    def test_get_model_storage_name(self):
        id = 'xyz'
        model_type = 'InceptionV3'
        found = utils.get_model_storage_name(model_type, id)
        expected = os.path.join(utils.MODEL_STORE_FOLDER, 'InceptionV3-xyz.h5')
        self.assertEqual(found, expected)

    @mock.patch('shared.utils.os.path.isdir', return_value=False)
    @mock.patch('shared.utils.os.mkdir')
    def test_get_model_storage_name_will_create_folder_when_needed(self, m_mkdir,m_isdir):
        id = 'z'
        model_type = 'xx'
        found = utils.get_model_storage_name(model_type, id)
        m_mkdir.assert_called_with(utils.MODEL_STORE_FOLDER)
        expected = os.path.join(utils.MODEL_STORE_FOLDER, 'xx-z.h5')
        self.assertEqual(found, expected)

    def test_get_model_log_name(self):
        id = 'xyz'
        model_type = 'InceptionV3'
        found = utils.get_model_log_name(model_type, id)
        expected = os.path.join(utils.MODEL_LOG_FOLDER, 'InceptionV3_xyz_log.csv')
        self.assertEqual(found, expected)

    @mock.patch('shared.utils.os.path.isdir', return_value=False)
    @mock.patch('shared.utils.os.mkdir')
    def test_get_model_log_name_will_create_folder_when_needed(self, m_mkdir, m_isdir):
        id = 'z'
        model_type = 'xx'
        found = utils.get_model_log_name(model_type, id)
        m_mkdir.assert_called_with(utils.MODEL_LOG_FOLDER)
        expected = os.path.join(utils.MODEL_LOG_FOLDER, 'xx_z_log.csv')
        self.assertEqual(found, expected)



    @mock.patch('shared.utils.argparse.ArgumentParser', autospec=True)
    def test_get_cmd_args(self, mock_parser_class):
        args = utils.get_cmd_args()
        mock_parser = mock_parser_class.return_value
        # check model_name, model_type and model_mode is mandatory in cmd args
        mock_parser.add_argument.assert_any_call('model_identifier', help=mock.ANY)
        mock_parser.add_argument.assert_any_call('model_type', choices=utils.SUPPORTED_MODEL_TYPES, default=mock.ANY,
                                                 help=mock.ANY)
        mock_parser.add_argument.assert_any_call('model_mode', choices=utils.SUPPORTED_MODEL_MODES, default=mock.ANY,
                                                 help=mock.ANY)

        # check for optional cmd args
        mock_parser.add_argument.assert_any_call('--train_dir', default=utils.TRAIN_DIR, help=mock.ANY)
        mock_parser.add_argument.assert_any_call('--val_dir', default=utils.VAL_DIR, help=mock.ANY)
        mock_parser.add_argument.assert_any_call('--batch_size', default=utils.BATCH_SIZE, type=int)
        mock_parser.add_argument.assert_any_call('--epochs', default=utils.NB_EPOCHS, type=int)
        mock_parser.add_argument.assert_any_call('--layers_to_freeze', default=0, type=int,
                                                 help=mock.ANY)
        mock_parser.add_argument.assert_any_call('--augment', type=bool, default=False, help=mock.ANY)

        # check the parsed arguments is what returned
        self.assertEqual(args, mock_parser.parse_args.return_value)


class TestDataGenerator(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.train_dir = tempfile.mkdtemp()
        cls.val_dir = tempfile.mkdtemp()

        args = argparse.Namespace()
        args.train_dir = cls.train_dir
        args.val_dir = cls.val_dir
        args.batch_size = 12
        cls.args = args

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.train_dir)
        shutil.rmtree(cls.val_dir)

    @mock.patch('shared.utils.ImageDataGenerator')
    def test_get_data_generators_calls_with_no_augmentation(self, mock_generator):
        empty_kwargs = {}
        train_data_generator, val_data_generator = utils.get_data_generators(self.args, empty_kwargs)

        mock_generator.assert_has_calls([
            mock.call(preprocessing_function=preprocess_input),
            mock.call(preprocessing_function=preprocess_input),
            mock.call().
                flow_from_directory(self.args.train_dir, batch_size=self.args.batch_size,
                                    target_size=(IM_WIDTH, IM_HEIGHT)),
            mock.call().
                flow_from_directory(self.args.val_dir, batch_size=self.args.batch_size,
                                    target_size=(IM_WIDTH, IM_HEIGHT))
        ])

    @mock.patch('shared.utils.ImageDataGenerator')
    def test_get_data_generators_calls_with_augmentation(self, mock_generator):
        augment_kwargs = {
            'zoom_range': 2,
            'rotation_range': 0.1,
            'horizontal_flip': True
        }
        train_data_generator, val_data_generator = utils.get_data_generators(self.args, augment_kwargs)

        mock_generator.assert_has_calls([
            mock.call(preprocessing_function=preprocess_input, zoom_range=2, rotation_range=0.1, horizontal_flip=True),
            mock.call(preprocessing_function=preprocess_input, zoom_range=2, rotation_range=0.1, horizontal_flip=True),
            mock.call().
                flow_from_directory(self.args.train_dir, batch_size=self.args.batch_size,
                                    target_size=(IM_WIDTH, IM_HEIGHT)),
            mock.call().
                flow_from_directory(self.args.val_dir, batch_size=self.args.batch_size,
                                    target_size=(IM_WIDTH, IM_HEIGHT))
        ])

    def test_get_data_generators_returns_the_right_folder_generators(self):
        empty_kwargs = {}
        train_data_generator, val_data_generator = utils.get_data_generators(self.args, empty_kwargs)
        self.assertEqual(train_data_generator.directory, self.args.train_dir)
        self.assertEqual(val_data_generator.directory, self.args.val_dir)


class TestFileCounting(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # create temporary folder to test counting files
        cls.p_dir = tempfile.mkdtemp()

        os.makedirs(os.path.join(cls.p_dir, 'child1/grand_child1'), exist_ok=True)
        os.makedirs(os.path.join(cls.p_dir, 'child1/grand_child2'), exist_ok=True)
        os.makedirs(os.path.join(cls.p_dir, 'child2'), exist_ok=True)
        os.makedirs(os.path.join(cls.p_dir, 'child3'), exist_ok=True)
        os.makedirs(os.path.join(cls.p_dir, 'child3/grand_child1'), exist_ok=True)
        os.makedirs(os.path.join(cls.p_dir, 'child3/grand_child2'), exist_ok=True)

        open(os.path.join(cls.p_dir, 'child1/grand_child1/abc.txt'), 'w').close()
        open(os.path.join(cls.p_dir, 'child1/grand_child1/xyz'), 'w').close()
        open(os.path.join(cls.p_dir, 'child1/grand_child1/media.mp4'), 'w').close()

        open(os.path.join(cls.p_dir, 'child1/abc.txt'), 'w').close()
        open(os.path.join(cls.p_dir, 'child1/xyz'), 'w').close()
        open(os.path.join(cls.p_dir, 'child1/media.mp4'), 'w').close()

        open(os.path.join(cls.p_dir, 'child2/abc.txt'), 'w').close()
        open(os.path.join(cls.p_dir, 'child2/xyz'), 'w').close()
        open(os.path.join(cls.p_dir, 'child2/media.mp4'), 'w').close()

        open(os.path.join(cls.p_dir, 'abc.txt'), 'w').close()
        open(os.path.join(cls.p_dir, 'xyz'), 'w').close()
        open(os.path.join(cls.p_dir, 'media.mp4'), 'w').close()

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.p_dir)

    def test_counts_files_only_in_subfodlers(self):
        self.assertEqual(3, utils.get_nb_files(os.path.join(self.p_dir, 'child1')))

    def test_zero_count_in_no_subfolders_but_files(self):
        self.assertEqual(0, utils.get_nb_files(os.path.join(self.p_dir, 'child2')))

    def test_zero_count_in_empty_subfolders(self):
        self.assertEqual(0, utils.get_nb_files(os.path.join(self.p_dir, 'child3/')))

    def test_zero_count_in_empty_folder(self):
        self.assertEqual(0, utils.get_nb_files(os.path.join(self.p_dir, 'child3/grand_child1')))

    def test_counts_files_in_many_folders_and_subfolders(self):
        """ N.B.
        The function actually counts files and additionally folders also
        Don't know if this is logic error or not
        """
        self.assertEqual(13, utils.get_nb_files(self.p_dir))

    def test_zero_count_if_folder_not_exist(self):
        self.assertEqual(0, utils.get_nb_files('xyz'))


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
        utils.setup_trainable_layers(self.model)

        # extract layers trainable boooleans in to separate list
        affected_layers_bool = []
        for layer in self.model.layers:
            affected_layers_bool.append(layer.trainable)

        self.assertListEqual(self.layers_bool, affected_layers_bool)

    def test_first_specified_number_of_layers_are_freezed(self):
        nb_layers = len(self.model.layers) // 2
        utils.setup_trainable_layers(self.model, nb_layers)

        # extract layers trainable boooleans in to separate list
        affected_layers_bool = []
        for layer in self.model.layers:
            affected_layers_bool.append(layer.trainable)

        # check layer trainable booleans prior to specified index are changed to False
        self.assertListEqual(affected_layers_bool[:nb_layers], [False for i in range(nb_layers)])

    def test_layers_after_specified_number_of_layers_are_not_freezed(self):
        nb_layers = len(self.model.layers) // 2
        utils.setup_trainable_layers(self.model, nb_layers)

        # extract layers trainable boooleans in to separate list
        affected_layers_bool = []
        for layer in self.model.layers:
            affected_layers_bool.append(layer.trainable)

        # check layer trainable booleans prior to specified index are changed to False
        self.assertTrue(all(affected_layers_bool[nb_layers:]))

    def test_no_layers_are_freezing_when_number_of_specified_layers_is_0(self):
        nb_layers = 0
        utils.setup_trainable_layers(self.model, nb_layers)

        # extract layers trainable boooleans in to separate list
        affected_layers_bool = []
        for layer in self.model.layers:
            affected_layers_bool.append(layer.trainable)

        # check layer trainable booleans are all set to True
        self.assertTrue(all(affected_layers_bool))

    def test_all_layers_are_freezing_when_number_of_layers_is_same_as_model_length(self):
        nb_layers = len(self.model.layers)
        utils.setup_trainable_layers(self.model, nb_layers)

        # extract layers trainable boooleans in to separate list
        affected_layers_bool = []
        for layer in self.model.layers:
            affected_layers_bool.append(layer.trainable)

        # check layer trainable booleans prior to specified index are changed to False
        self.assertListEqual(affected_layers_bool, [False for i in range(nb_layers)])

    def test_raises_valueError_when_number_of_layers_specified_is_greater_than_available_layers(self):
        nb_layers = len(self.model.layers) + 1
        with self.assertRaises(ValueError) as ve:
            utils.setup_trainable_layers(self.model, nb_layers)

    def test_raises_valueError_when_number_of_layers_specified_is_negative(self):
        nb_layers = -1
        with self.assertRaises(ValueError) as ve:
            utils.setup_trainable_layers(self.model, nb_layers)


@mock.patch('shared.utils.get_data_generators', return_value=('train_data_generator', 'val_data_generator'))
class TestTrainModel(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.train_dir = tempfile.mkdtemp()
        cls.val_dir = tempfile.mkdtemp()

        args = argparse.Namespace()
        args.model_identifier = 'mock_model'
        args.model_type = utils.INCEPTIONV3_ARCHITECTURE
        args.model_type = utils.FINETUNE
        args.train_dir = cls.train_dir
        args.val_dir = cls.val_dir
        args.batch_size = 12
        args.epochs = 10
        args.augment = False
        cls.args = args

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.train_dir)
        shutil.rmtree(cls.val_dir)

    def setUp(self):
        self.mock_model = mock.create_autospec(Model)
        self.args.model_type = utils.INCEPTIONV3_ARCHITECTURE
        self.args.model_type = utils.FINETUNE
        self.args.augment = False

    def test_augmentation_is_setup_correctly_when_required(self, mock_data_generators):
        self.args.augment = True
        utils.train_model(self.mock_model, self.args)
        mock_data_generators.assert_called_once_with(self.args, utils.AUGMENTATION_KWARGS)

    def test_augmentation_is_setup_correctly_vgg_model_is_used(self, mock_data_generators):
        self.args.augment = False
        self.args.model_type = utils.VGG_ARCHITECTURE
        utils.train_model(self.mock_model, self.args)
        mock_data_generators.assert_called_once_with(self.args, utils.AUGMENTATION_KWARGS)

    def test_augmentation_is_not_setup_when_not_required_and_inceptionv3_model_is_used(self, mock_data_generators):
        self.args.augment = False
        utils.train_model(self.mock_model, self.args)
        mock_data_generators.assert_called_once_with(self.args, {})

    @mock.patch('shared.utils.CSVLogger')
    @mock.patch('shared.utils.get_model_log_name')
    def test_csv_logger_callback_is_setup(self, mock_get_log_name, mock_csv_logger, mock_get_generators):
        utils.train_model(self.mock_model, self.args)
        # check csv logger is there and  called with the right argument
        mock_csv_logger.assert_called_once_with(mock_get_log_name.return_value)
        self.assertIn(mock_csv_logger.return_value, self.mock_model.fit_generator.call_args[1]['callbacks'])

    @mock.patch('shared.utils.get_model_storage_name')
    def test_model_is_saved(self, mock_storage_name, mock_get_generators):
        utils.train_model(self.mock_model, self.args)
        # check model is saved with the right argument
        self.mock_model.save.assert_called_once_with(mock_storage_name.return_value)

    def test_fit_generator_is_called(self, mock_get_generators):
        utils.train_model(self.mock_model, self.args)

        # check fit generator is called and with right arguments
        self.mock_model.fit_generator.assert_called_once()
        call_args, call_kwargs = self.mock_model.fit_generator.call_args
        self.assertEqual(call_args[0], 'train_data_generator')
        self.assertEqual(call_kwargs['validation_data'], 'val_data_generator')
        self.assertEqual(call_kwargs['epochs'], self.args.epochs)


if __name__ == '__main__':
    unittest.main()
