import numpy as np
import tensorflow as tf

import sys
sys.path.append('../../action_recognition/architectures/')
from Metrics import RankMetrics

class RankTest(tf.test.TestCase):

    def testRankLossOutputCorrectness(self):
        y_true = np.array([[1, 0, 0], [0, 0, 1]])
        y_pred = np.array([[0.75, 0.5, 1], [1, 0.2, 0.1]])
        expected_output = 5.

        output = RankMetrics().rank_loss(y_true, y_pred)

        self.assertAllEqual(expected_output, output)
    
    def testCoverageErrorOutputCorrectness(self):
        y_true = np.array([[1, 0, 1], [0, 0, 1]])
        y_pred = np.array([[0.75, 0.5, 1], [1, 0.2, 0.1]])
        expected_output = 2.5

        output = RankMetrics().coverage_error(y_true, y_pred)

        self.assertAllEqual(expected_output, output)
    
    # def testCoverageErrorOutputCorrectness(self):
    #     y_true = np.array([[1, 0, 0], [0, 0, 1]])
    #     y_pred = np.array([[0.75, 0.5, 1], [1, 0.2, 0.1]])
    #     expected_output = 0.

    #     output = RankMetrics().one_error(y_true, y_pred)

    #     self.assertAllEqual(expected_output, output)


tf.test.main()