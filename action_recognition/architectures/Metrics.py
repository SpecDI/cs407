import tensorflow as tf
tf.enable_eager_execution()
K = tf.keras.backend

#from sklearn.metrics import label_ranking_loss
#import numpy as np

class RankMetrics:
    def __init__(self):
        pass

    def rank_loss(self, y_true, y_pred):
        y_true_ = tf.cast(y_true, tf.float32)

        # print("PARTIAL LOSS")
        # print("rows")
        # print(y_pred[:, None, :])
        # print("cols")
        # print(y_pred[:, :, None])
        # print("first cal: 1-y_pred")
        # print(1 - y_pred[:, None, :])
        # t = 1 - y_pred[:, None, :]
        # print("final calc")
        # print(t + y_pred[:, :, None])
        # print("t??")
        # x = t + y_pred[:, :, None]
        # y = t - y_pred[:, :, None]
        # t2 = (x + y)/2
        # print(t2)
        # print("x-t2")
        # print(x - t2)

        # print("max calc")
        # print(tf.maximum(0.0, 1 - y_pred[:, None, :] + y_pred[:, :, None]))
        partial_losses = tf.maximum(0.0, 1 - y_pred[:, None, :] + y_pred[:, :, None])
        # print("Partial loss")
        # with tf.Session() as sess:  print(partial_losses.numpy()) 
        # 
        # print("LOSS")
        # print("rows")
        # print(y_true[:, None, :])
        # print("cols")
        # print(y_true_[:, :, None])
        # print("pl * y_true")
        # print(partial_losses * y_true_[:, None, :])
        # print("1 - y_true")
        # print((1 - y_true_[:, :, None]))
        # print("full")
        # print(partial_losses * y_true_[:, None, :] * (1 - y_true_[:, :, None]))

        loss = partial_losses * y_true_[:, None, :] * (1 - y_true_[:, :, None])
        # print("loss")
        # with tf.Session() as sess:  print(loss.numpy()) 
        # print("result")jii
        # with tf.Session() as sess:  print(tf.reduce_sum(loss).numpy()) 
        # print("TESTING RANK LOSS")
        return tf.reduce_sum(loss)

        # 
    	#### WORKING VERSION USING NUMPY #####
        # rloss = 0.0
        # nrow = y_pred.shape[0]
        # for i in range(nrow):
        #     correct = np.where(y_true[i] == 1)[0]
        #     incorrect = np.where(y_true[i] == 0)[0]
        #     inv_ranks = np.argsort(y_pred[i])
        #     if (len(correct) == 0 or len(incorrect) == 0):
        #         continue  # rank loss = 0

        #     nincorrect = 0.0

        #     print("correct: ", correct)
        #     print("incorrect: ", incorrect)

        #     for l_a in correct:
        #         for l_b in incorrect:
        #             print("vals: ", l_a , " ",  l_b)
        #             rank_l_a = len(inv_ranks) - np.where(inv_ranks == l_a)[0][0]
        #             rank_l_b = len(inv_ranks) - np.where(inv_ranks == l_b)[0][0]
        #             print("rank_l_a: ", rank_l_a)
        #             print("rank_l_b: ", rank_l_b)
        #             if (rank_l_a > rank_l_b): nincorrect += 1
        #     print("t: ", (len(correct) * len(incorrect)))
        #     print("nincorrect:", nincorrect)        
        #     rloss_i = nincorrect / (len(correct) * len(incorrect))
        #     print("rloss_i: ", rloss_i)
        #     rloss += rloss_i
        #     print("rloss: ", rloss)
        #     print("")
        # rloss /= nrow
        # return rloss

        ##### TENSOR FLOW VERSION

        # rloss = 0.0
        # nrow = y_pred.shape[0]
        # for i in range(nrow):
        #     correct = tf.where(y_true[i] == 1)[0]
        #     # print(tf.where(y_true[i] == 0))
        #     incorrect = tf.where(y_true[i] == 0)
        #     inv_ranks = tf.argsort(y_pred[i])
        #     if (tf.size(correct) == 0 or tf.size(incorrect) == 0):
        #         continue  # rank loss = 0

        #     nincorrect = 0.0
        #     # print("correct: ", correct)
        #     # print("incorrect: ", incorrect)
        #     for l_a in correct:
        #         for l_b in incorrect:
        #             # inv_ranks = tf.cast(inv_ranks, tf.int64)
        #             # print("vals: ", l_a , " ",  l_b)
        #             l_a_ = tf.cast(l_a, tf.int32)
        #             l_b_ = tf.cast(l_b, tf.int32)
        #             # print(inv_ranks)
        #             # print(l_a_)
        #             # test = tf.reduce_sum(l_a).numpy
        #             # print(test)
        #             l_a_where =  tf.cast(tf.where(tf.equal(inv_ranks, l_a_))[0][0], tf.int32)
        #             rank_l_a = tf.size(inv_ranks) - l_a_where

        #             l_b_where =  tf.cast(tf.where(tf.equal(inv_ranks, l_b_))[0][0], tf.int32)
        #             rank_l_b = tf.size(inv_ranks) - l_b_where

        #             # print("rank_l_a: ",rank_l_a)
        #             # print("rank_l_b: ",rank_l_b)

        #             # rank_l_b = tf.size(inv_ranks) - tf.where(tf.equal(inv_ranks, l_b_))[0][0]
        #             # rank_l_a = tf.size(inv_ranks) - tf.where(tf.equal(inv_ranks, l_a_))[0][0]
        #             # rank_l_b = tf.size(inv_ranks) - tf.where(tf.equal(inv_ranks, l_b_))[0][0]
        #             # rank_l_a = tf.size(inv_ranks) - tf.where(inv_ranks == l_a)[0][0]
        #             # rank_l_b = tf.size(insv_ranks) - tf.where(inv_ranks == l_b)[0][0]
                    
        #             if (rank_l_a > rank_l_b): nincorrect += 1
        #     denom_calc  = tf.cast((tf.size(correct) * tf.size(incorrect)), tf.float64)
        #     # print("t: ",t)
            
        #     # print("nincorrect: ", nincorrect)
        #     rloss_i = nincorrect / denom_calc
        #     # print("rloss_i: ", rloss_i)
        #     rloss += rloss_i
        #     # print("rloss: ", rloss)
        #     # print("")
        # rloss /= nrow
        # return rloss    
		

    def coverage_error(self, y_true, y_pred):
        y_true = tf.cast(y_true, tf.bool)
       	y_pred_masked = tf.ragged.boolean_mask(y_pred, y_true)
        y_min_relevant = tf.reduce_min(y_pred_masked, axis=1)
        y_min_relevant = tf.reshape(y_min_relevant, [tf.shape(y_pred)[0], 1])
        coverage = tf.reduce_sum(tf.cast((y_pred >= y_min_relevant), tf.float32), axis=1)
        return tf.reduce_mean(coverage)

    def one_error(self, y_true, y_pred):
        max_tensor = tf.reduce_max(y_pred, axis=1, keepdims=True)
        mask = tf.math.equal(y_pred, max_tensor)
        y_true_masked = tf.ragged.boolean_mask(y_true, mask)
        one_error = tf.reduce_min(y_true_masked, axis=1)
        return tf.reduce_mean(1. - tf.cast(one_error, tf.float32))

class MetricsAtTopK:
    def __init__(self, k):
        self.k = k
        self.classes = 13

    def _get_prediction_tensor(self, y_pred):
        """
	Takes y_pred and creates a tensor of same shape with 1 in indices where, the values are in top_k
        """
        topk_values, topk_indices = tf.nn.top_k(y_pred, k=self.k, sorted=False, name="topk")
        # the topk_indices are along last axis (1). Add indices for axis=0
        ii, _ = tf.meshgrid(tf.range(tf.shape(y_pred)[0]), tf.range(self.k), indexing='ij')
        index_tensor = tf.reshape(tf.stack([ii, topk_indices], axis=-1), shape=(-1, 2))
        prediction_tensor = tf.sparse_to_dense(sparse_indices=index_tensor,
                                               output_shape=tf.shape(y_pred),
                                               default_value=0,
                                               sparse_values=1.0,
                                               validate_indices=False
                                               )
        prediction_tensor = tf.cast(prediction_tensor, K.floatx())
        return prediction_tensor

    def true_positives_at_k(self, y_true, y_pred):
        prediction_tensor = self._get_prediction_tensor(y_pred=y_pred)
        true_positive = K.sum(tf.multiply(prediction_tensor, y_true))
        return true_positive

    def false_positives_at_k(self, y_true, y_pred):
        prediction_tensor = self._get_prediction_tensor(y_pred=y_pred)
        true_positive = K.sum(tf.multiply(prediction_tensor, y_true))
        c2 = K.sum(prediction_tensor)  # TP + FP
        false_positive = c2 - true_positive
        return false_positive

    def false_negatives_at_k(self, y_true, y_pred):
        prediction_tensor = self._get_prediction_tensor(y_pred=y_pred)
        true_positive = K.sum(tf.multiply(prediction_tensor, y_true))
        c3 = K.sum(y_true)  # TP + FN
        false_negative = c3 - true_positive
        return false_negative

    def precision_at_k(self, y_true, y_pred):
        prediction_tensor = self._get_prediction_tensor(y_pred=y_pred)
        true_positive = K.sum(tf.multiply(prediction_tensor, y_true))
        c2 = K.sum(prediction_tensor)  # TP + FP
        return true_positive/(c2+K.epsilon())

    def recall_at_k(self, y_true, y_pred):
        prediction_tensor = self._get_prediction_tensor(y_pred=y_pred)
        true_positive = K.sum(tf.multiply(prediction_tensor, y_true))
        c3 = K.sum(y_true)  # TP + FN
        return true_positive/(c3+K.epsilon())

    def f1_at_k(self, y_true, y_pred):
        precision = self.precision_at_k(y_true=y_true, y_pred=y_pred)
        recall = self.recall_at_k(y_true=y_true, y_pred=y_pred)
        f1 = (2*precision*recall)/(precision+recall+K.epsilon())
        return f1

    
