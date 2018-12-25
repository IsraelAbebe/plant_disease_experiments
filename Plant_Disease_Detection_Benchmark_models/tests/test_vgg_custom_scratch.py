import argparse
import unittest
from unittest import mock

from VGG import build_custom_model
from shared import utils


class TestCustomVGGModel(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        args = argparse.Namespace()
        args.nb_classes = 3
        cls.args = args

    @mock.patch('VGG.custom_scratch.VGG')
    def test_VGG_is_created(self, mock_VGG):
        build_custom_model(self.args, utils.INPUT_SHAPE)

        # check input_tensor shape is set up right
        mock_VGG.assert_called_with(self.args.nb_classes, input_shape=utils.INPUT_SHAPE)

    @mock.patch('VGG.custom_scratch.VGG')
    def test_VGG_is_compiled(self, mock_VGG):
        build_custom_model(self.args, utils.INPUT_SHAPE)

        self.assertTrue(mock_VGG.return_value.compile.called)


if __name__ == '__main__':
    unittest.main()
