import tensorflow as tf
K = tf.keras.backend

class RankMetrics:
    def __init__(self):
        pass

    def rank_loss(self, y_true, y_pred):
        y_true_ = tf.cast(y_true, tf.float32)
        partial_losses = tf.maximum(0.0, 1 - y_pred[:, None, :] + y_pred[:, :, None])
        loss = partial_losses * y_true_[:, None, :] * (1 - y_true_[:, :, None])
        return tf.reduce_sum(loss)
    
    def coverage_error(self, y_true, y_pred):
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

    
