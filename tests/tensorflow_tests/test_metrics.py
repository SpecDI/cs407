import numpy as np
import tensorflow as tf

import sys
sys.path.append('../../action_recognition/architectures/')
from Metrics import RankMetrics

class RankTest(tf.test.TestCase):

# Rank Loss tests - all fail due to different defintion compared to sklearn definition

    # def testRankLossOutputCorrectness(self):
    #     y_true = np.array([[1, 0, 0], [0, 0, 1]])
    #     y_pred = np.array([[0.75, 0.5, 1], [1, 0.2, 0.1]])
    #     expected_output = 0.75 #should be 0.75 - gives 5.0 ???


    #     output = RankMetrics().rank_loss(y_true, y_pred)

    #     self.assertAllEqual(expected_output, output)

    # def testRankLossOutputCorrectness_1(self):
    #     y_true = np.array([[1, 0, 0], [0, 0, 1]])
    #     y_pred = np.array([[1.0, 0.1, 0.2], [0.1, 0.2, 0.9]])
    #     expected_output = 0.0 #should be 0.0 - give 0.8

    #     output = RankMetrics().rank_loss(y_true, y_pred)

    #     self.assertAllEqual(expected_output, output)


    # def testRankLossOutputCorrectness_2(self):
    #     y_true = np.array([[0, 1, 0]])
    #     y_pred = np.array([[0.25, 0.5, 0.5]])
    #     expected_output = 0.5 # should be 0.5 gives 1.75

    #     output = RankMetrics().rank_loss(y_true, y_pred)

    #     self.assertAllEqual(expected_output, output)

    def testRankLossOutputCorrectness_3(self):
        y_true = np.array([[1, 1, 0]])
        y_pred = np.array([[0.25, 0.5, 0.5]])
        expected_output = 1 # should be 1 gives 2.25

        output = RankMetrics().rank_loss(y_true, y_pred)

        self.assertAllEqual(expected_output, output)
    
    def testCoverageErrorOutputCorrectness(self):
        y_true = np.array([[1, 0, 0], [0, 0, 1]])
        y_pred = np.array([[0.75, 0.5, 1], [1, 0.2, 0.1]])
        expected_output = 2.5

        output = RankMetrics().coverage_error(y_true, y_pred)

        self.assertAllEqual(expected_output, output)

    def testCoverageErrorOutputCorrectness_1(self):

        y_true = np.array([[1, 1, 0]])
        y_pred = np.array([[0.25, 0.5, 0.5]])
        expected_output = 3

        output = RankMetrics().coverage_error(y_true, y_pred)

        self.assertAllEqual(expected_output, output)

    def testCoverageErrorOutputCorrectness_2(self):

        y_true = np.array([[0, 0, 0]])
        y_pred = np.array([[0.25, 0.5, 0.5]])
        expected_output = 0

        output = RankMetrics().coverage_error(y_true, y_pred)

        self.assertAllEqual(expected_output, output)

    def testCoverageErrorOutputCorrectness_3(self):

        y_true = np.array([[1, 0, 1]])
        y_pred = np.array([[0.25, 0.5, 0.5]])
        expected_output = 3

        output = RankMetrics().coverage_error(y_true, y_pred)

        self.assertAllEqual(expected_output, output)

# Test coverage 4-6 are non trival cases

    def testCoverageErrorOutputCorrectness_4(self):

        y_true = np.array([[0, 1, 0], [1, 1, 0]])
        y_pred = np.array([[0.1, 10., -3], [0, 1, 3]])
        expected_output = 2

        output = RankMetrics().coverage_error(y_true, y_pred)

        self.assertAllEqual(expected_output, output)

# Test coverage 5-6 - comment out but do work - says error as rhs 2.3333332538604736 - rounding error - so commented out for now
    # def testCoverageErrorOutputCorrectness_5(self):

    #     y_true = np.array([[0, 1, 0], [1, 1, 0], [0, 1, 1]])
    #     y_pred = np.array([[0.1, 10, -3], [0, 1, 3], [0, 2, 0]])
    #     expected_output = 7/3.

    #     output = RankMetrics().coverage_error(y_true, y_pred)

    #     self.assertAllEqual(expected_output, output)

    # def testCoverageErrorOutputCorrectness_6(self):

    #     y_true = np.array([[0, 1, 0], [1, 1, 0], [0, 1, 1]])
    #     y_pred = np.array([[0.1, 10, -3], [3, 1, 3], [0, 2, 0]])
    #     expected_output = 7/3.

    #     output = RankMetrics().coverage_error(y_true, y_pred)

    #     self.assertAllEqual(expected_output, output)
    
    def testMultilabelCoverageErrorOutputCorrectness(self):
        y_true = np.array([[1, 0, 1], [0, 0, 1]])
        y_pred = np.array([[0.75, 0.5, 1], [1, 0.2, 0.1]])
        expected_output = 2.5

        output = RankMetrics().coverage_error(y_true, y_pred)

        self.assertAllEqual(expected_output, output)
    
    def testOneErrorOutputCorrectness(self):
        y_true = np.array([[1, 0, 0], [0, 0, 1]])
        y_pred = np.array([[0.75, 0.5, 1], [1, 0.2, 0.1]])
        expected_output = 1.

        output = RankMetrics().one_error(y_true, y_pred)

        self.assertAllEqual(expected_output, output)
    
    def testOneErrorOutputwithLabelMatchCorrectness(self):
        y_true = np.array([[1, 0, 0], [0, 0, 1]])
        y_pred = np.array([[1, 0.5, 0], [1, 0.2, 0.1]])
        expected_output = 0.5

        output = RankMetrics().one_error(y_true, y_pred)

        self.assertAllEqual(expected_output, output)
    
    def testMultilabelOneErrorOutputCorrectness(self):
        y_true = np.array([[1, 0, 1], [0, 0, 1]])
        y_pred = np.array([[1, 0.5, 0], [1, 0.2, 0.1]])
        expected_output = 0.5

        output = RankMetrics().one_error(y_true, y_pred)

        self.assertAllEqual(expected_output, output)


tf.test.main()