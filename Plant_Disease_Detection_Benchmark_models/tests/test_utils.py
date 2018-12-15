import argparse
import unittest
import os
import shutil
import tempfile
from unittest import mock

from tensorflow.python.keras.applications.inception_v3 import preprocess_input
from tensorflow.python.keras import Model

from Inception_V3 import utils
from Inception_V3.utils import IM_WIDTH, IM_HEIGHT

def setupModule():
    print('setting up module')


class TestSimpleFunctions(unittest.TestCase):

    def test_get_model_storage_name(self):
        id = 'xyz'
        found = utils.get_model_storage_name(id)
        expected = '../Models/Inception_V3-xyz.h5'
        self.assertEqual(found, expected)

    def test_get_model_log_name(self):
        id = 'xyz'
        found = utils.get_model_log_name(id)
        expected = 'xyz_iv3_log.csv'
        self.assertEqual(found, expected)

    @mock.patch('Inception_V3.utils.argparse.ArgumentParser', autospec=True)
    def test_get_cmd_args(self, mock_parser_class):
        args = utils.get_cmd_args()
        mock_parser = mock_parser_class.return_value
        # check model_name is mandatory in cmd args
        mock_parser.add_argument.assert_any_call('model_name', mock.ANY)
        mock_parser.add_argument.assert_any_call('--batch_size', default=utils.BATCH_SIZE, type=int)
        mock_parser.add_argument.assert_any_call('--epochs', default=utils.NB_EPOCHS, type=int)
        mock_parser.add_argument.assert_any_call('--val_dir', default=utils.VAL_DIR)
        mock_parser.add_argument.assert_any_call('--train_dir', default=utils.TRAIN_DIR)

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

    @mock.patch('Inception_V3.utils.ImageDataGenerator')
    def test_get_data_generators_calls(self, mock_generator):
        train_data_generator, val_data_generator = utils.get_data_generators(self.args)

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

    def test_get_data_generators_returns_the_right_folder_generators(self):
        train_data_generator, val_data_generator = utils.get_data_generators(self.args)
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


@mock.patch('Inception_V3.utils.get_data_generators', return_value=('train_data_generator', 'val_data_generator'))
class TestTrainModel(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.train_dir = tempfile.mkdtemp()
        cls.val_dir = tempfile.mkdtemp()

        args = argparse.Namespace()
        args.train_dir = cls.train_dir
        args.val_dir = cls.val_dir
        args.batch_size = 12
        args.epochs = 10
        args.model_name = 'mock_model'
        cls.args = args

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.train_dir)
        shutil.rmtree(cls.val_dir)

    def setUp(self):
        self.mock_model = mock.create_autospec(Model)

    @mock.patch('Inception_V3.utils.CSVLogger')
    @mock.patch('Inception_V3.utils.get_model_log_name')
    def test_csv_logger_callback_is_setup(self, mock_get_log_name, mock_csv_logger, mock_get_generators):
        utils.train_model(self.mock_model, self.args)
        # check csv logger is there and  called with the right argument
        mock_csv_logger.assert_called_once_with(mock_get_log_name.return_value)
        self.assertIn(mock_csv_logger.return_value, self.mock_model.fit_generator.call_args[1]['callbacks'])

    @mock.patch('Inception_V3.utils.get_model_storage_name')
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
