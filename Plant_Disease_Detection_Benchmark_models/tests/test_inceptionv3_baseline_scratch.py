import argparse
import unittest
from unittest import mock

import tensorflow as tf
import numpy as np

import shared.utils as utils

from Inception_V3.baseline_scratch import build_baseline_model


class TestInceptionV3(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        args = argparse.Namespace()
        args.nb_classes = 3
        cls.args = args

    @mock.patch('Inception_V3.baseline_scratch.Input')
    @mock.patch('Inception_V3.baseline_scratch.InceptionV3')
    def test_InceptionV3_is_created(self, mock_IV3, mock_Input):
        build_baseline_model(self.args, utils.INPUT_SHAPE)

        # check input_tensor shape is set up right
        mock_Input.assert_called_with(shape=utils.INPUT_SHAPE)
        mock_IV3.assert_called_with(include_top=True, classes=self.args.nb_classes, weights=None,
                                    input_tensor=mock_Input.return_value)

    @mock.patch('Inception_V3.baseline_scratch.InceptionV3')
    def test_InceptionV3_is_compiled(self, mock_IV3):
        build_baseline_model(self.args, utils.INPUT_SHAPE)

        self.assertTrue(mock_IV3.return_value.compile.called)


if __name__ == '__main__':
    unittest.main()
