from __future__ import absolute_import
from __future__ import print_function

__all__ = ['ensure_dir', 'checkfrozen', 'CorrectnessTest']

import os
import unittest
import numpy as np

def _compute_SNR(x,y):
    noise = x - y
    noise_var = np.sum(noise ** 2) / len(noise) + 1e-7
    signal_energy = np.sum(y ** 2) / len(y)
    max_signal_energy = np.amax(y ** 2)
    SNR = 10 * np.log10(signal_energy / noise_var)
    PSNR = 10 * np.log10(max_signal_energy / noise_var)
    return SNR, PSNR


def _compute_max_relative_error(x, y):
    from six.moves import xrange
    rerror = 0
    index = 0
    for i in xrange(len(x)):
        den = max(1.0, np.abs(x[i]), np.abs(y[i]))
        if np.abs(x[i]/den - y[i] / den) > rerror:
            rerror = np.abs(x[i] / den - y[i] / den)
            index = i
    return rerror, index


def _compute_L1_error(x, y):
    return np.linalg.norm(x - y, ord=1)


def ensure_dir(f):
    d = os.path.dirname(f)
    if not os.path.exists(d):
        os.makedirs(d)

def checkfrozen(f):
    if f == 'tensorflow_frozen':
        return 'tensorflow'
    else:
        return f


class CorrectnessTest(unittest.TestCase):

    err_thresh = 0.15
    snr_thresh = 12
    psnr_thresh = 30

    @classmethod
    def setUpClass(cls):
        """ Set up the unit test by loading common utilities.
        """
        pass


    def _compare_outputs(self, original_framework, target_framework, network_name, original_predict, converted_predict, need_assert=True):
        # Function self.assertEquals has deprecated, change to assertEqual
        if (converted_predict is None or original_predict is None) and not need_assert:
            return

        # self.assertEqual(original_predict.shape, converted_predict.shape)
        original_predict = original_predict.flatten()
        converted_predict = converted_predict.flatten()
        len1 = original_predict.shape[0]
        len2 = converted_predict.shape[0]
        length = min(len1, len2)
        original_predict = np.sort(original_predict)[::-1]
        converted_predict = np.sort(converted_predict)[::-1]
        original_predict = original_predict[0:length]
        converted_predict = converted_predict[0:length]
        error, ind = _compute_max_relative_error(converted_predict, original_predict)
        L1_error = _compute_L1_error(converted_predict, original_predict)
        SNR, PSNR = _compute_SNR(converted_predict, original_predict)
        print("error:", error)
        print("L1 error:", L1_error)
        print("SNR:", SNR)
        print("PSNR:", PSNR)

        if need_assert:
            self.assertGreater(SNR, self.snr_thresh, "Error in converting {} from {} to {}".format(network_name, original_framework, target_framework))
            self.assertGreater(PSNR, self.psnr_thresh, "Error in converting {} from {} to {}".format(network_name, original_framework, target_framework))
            self.assertLess(error, self.err_thresh, "Error in converting {} from {} to {}".format(network_name, original_framework, target_framework))
