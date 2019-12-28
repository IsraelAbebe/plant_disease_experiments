import argparse
import unittest
from unittest import mock

from ResNet import build_custom_model
from ResNet.resnet import ResnetBuilder
from shared import utils


class TestCustomResnetModel(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        args = argparse.Namespace()
        args.nb_classes = 3
        cls.args = args

    @mock.patch.object(ResnetBuilder, 'build_resnet_18')
    def test_Resnet18_is_created(self, mock_resnet18):
        build_custom_model(self.args, utils.INPUT_SHAPE)

        expected_input_shape = (3, utils.INPUT_SHAPE[0], utils.INPUT_SHAPE[1])
        # check input_tensor shape is set up right
        mock_resnet18.assert_called_with(expected_input_shape, self.args.nb_classes)

    @mock.patch.object(ResnetBuilder, 'build_resnet_18')
    def test_Resnet_is_compiled(self, mock_resnet18):
        build_custom_model(self.args, utils.INPUT_SHAPE)

        self.assertTrue(mock_resnet18.return_value.compile.called)


if __name__ == '__main__':
    unittest.main()
