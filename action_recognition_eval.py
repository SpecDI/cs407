"""
We expect the entries of the ground truth text file to have the form:
<id>_<batch_num>,<action1_action_2_..._action_n>
"""
import tensorflow as tf
K = tf.keras.backend

import os 
import numpy as np
from collections import OrderedDict
from action_recognition.architectures.Metrics_v2 import MetricsAtTopK, RankMetrics
from action_recognition.architectures.Loss import LossFunctions

video = "example"
pred_path = "results/action_recognition/" + video
ground_truth = "ground_truth/" + video + ".txt"
delim = ','

classes = sorted(['Unknown', 'Sitting', 'Lying', 'Drinking', 'Calling', 'Reading', 'Handshaking', 'Running', 'Pushing_Pulling', 'Walking', 'Hugging', 'Carrying', 'Standing'])
classes_dict = OrderedDict(zip(classes, range(len(classes))))

def multi_hot_encoding(class_names):
    class_vector = np.zeros(len(classes))
    for i in class_names:
        class_vector[classes_dict[i]] = 1
    return class_vector

with open(ground_truth) as f:
  d = dict(x.rstrip().split(delim) for x in f)

for k, v in d.items():
    d[k] = v.split('_')

Y_true = []
Y_pred = []
for subdir in sorted(os.listdir(pred_path)):
    id, batch_num, *label = subdir.split('_')
    action_tube_id = id + '_' + batch_num

    Y_true.append(multi_hot_encoding(d[action_tube_id]))
    Y_pred.append(multi_hot_encoding(label))

Y_true = tf.convert_to_tensor(Y_true, dtype=tf.float32)
Y_pred = tf.convert_to_tensor(Y_pred, dtype=tf.float32)

metrics = MetricsAtTopK(3)
rank_metrics = RankMetrics()
losses = LossFunctions()

hamming_loss = losses.hamming_loss(Y_true, Y_pred)
# one_error = rank_metrics.one_error(Y_true, Y_pred)
precision = metrics.precision_at_k(Y_true, Y_pred)
recall = metrics.recall_at_k(Y_true, Y_pred)
f1_at_k = metrics.f1_at_k(Y_true, Y_pred)

print("Hamming Loss: \t %f" % hamming_loss)
# print("One Error: \t %f" % one_error)
print("Precision: \t %f" % precision)
print("Recall: \t %f" % recall)
print("F1 score: \t %f" % f1_at_k)



