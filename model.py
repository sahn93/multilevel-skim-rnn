# Copyright 2017 Google Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Kernel model.

This model allows interactions between the two inputs, such as attention.
"""
from __future__ import division

import sys

import tensorflow as tf
import tensorflow.contrib.learn as learn
from tensorflow.contrib.rnn import BasicLSTMCell

# import random
import tf_utils
from common_model import word_feature_vector, bilinear, embedding_layer
from nested_rnn import NestedLSTMWrapper
#from vc_rnn import VCGRUWrapper

# from IPython import embed
import json


def model(features, mode, params, scope=None):
    """Kernel models that allow interaction between question and context.

    This is handler for all kernel models in this script. Models are called via
    `params.model_id` (e.g. `params.model_id = m00`).

    Function requirement for each model is in:
    https://www.tensorflow.org/extend/estimators

    This function does not have any dependency on FLAGS. All parameters must be
    passed through `params` argument.

    Args:
      features: A dict of feature tensors.
      mode: https://www.tensorflow.org/api_docs/python/tf/contrib/learn/ModeKeys
      params: `params` passed during initialization of `Estimator` object.
      scope: Variable name scope.
    Returns:
      `(logits_start, logits_end, tensors)` pair. Tensors is a dictionary of
      tensors that can be useful outside of this function, e.g. visualization.
    """
    this_module = sys.modules[__name__]
    model_fn = getattr(this_module, '_model_%s' % params.model_id)
    return model_fn(
        features, mode, params, scope=scope)


def _model_m00(features, mode, params, scope=None):
    """Simplified BiDAF, reaching 74~75% F1.

   ://github.com/google/mipsqa Args:
      features: A dict of feature tensors.
      mode: https://www.tensorflow.org/api_docs/python/tf/contrib/learn/ModeKeys
      params: `params` passed during initialization of `Estimator` object.
      scope: Variable name scope.
    Returns:
      `(logits_start, logits_end, tensors)` pair. Tensors is a dictionary of
      tensors that can be useful outside of this function, e.g. visualization.
    """
    with tf.variable_scope(scope or 'kernel_model') as vs:
        training = mode == learn.ModeKeys.TRAIN
        tensors = {}
        x, q = embedding_layer(features, mode, params)

        def bi_rnn(x, d, params, sequence_length_list, scope, skim, i):
            # print("call bi_rnn %d!!!" % (i))
            cells = None
            if skim:
                cells = _get_skim_cells(params, training)
            h = tf_utils.bi_rnn(
                d,
                x,
                sequence_length_list=sequence_length_list,
                scope=scope,
                training=training,
                cells=cells,
                dropout_rate=params.dropout_rate)
            if (not skim):  # or params.vcgru:
                return h
            h_fw, q_fw, h_bw, q_bw = tf.split(h, [d, params.num_cells, d, params.num_cells], axis=2)
            x_mask = tf.expand_dims(
                tf.cast(tf.sequence_mask(sequence_length_list), 'float32'), 2)
            x_masks = []
            for nnn in range(params.num_cells):
                x_masks.append(x_mask)

            x_mask = tf.concat(x_masks, 2)
            tensors['q_fw_%d' % (i)] = q_fw * x_mask
            tensors['q_bw_%d' % (i)] = q_bw * x_mask
            # print("tensor add q_**_%d!!!" % (i))
            return tf.concat([h_fw, h_bw], 2)

        x0 = bi_rnn(x, params.embed_hidden_size, params, features['context_num_words'],
                    'x_bi_rnn_0', params.skim_embed, 0)
        q0 = bi_rnn(q, params.embed_hidden_size, params, features['question_num_words'],
                    'q_bi_rnn_0', False, 0)
        xq = tf_utils.att2d(
            q0,
            x0,
            mask=features['question_num_words'],
            tensors=tensors,
            scope='xq')

        tensors['attention'] = xq

        word_feat = word_feature_vector(features)

        def get_concat_vector(vectors):
            vectors_ = vectors + [word_feat] if params.word_feat else vectors
            if len(vectors_) == 1: return vectors_[0]
            return tf.concat(vectors_, 2)

        xq = get_concat_vector([x0, xq, x0 * xq])
        x1 = bi_rnn(xq, params.hidden_size, params, features['context_num_words'],
                    'x1_bi_rnn', params.skim_1, 1)
        x2 = bi_rnn(get_concat_vector([x1]), params.hidden_size, params,
                    features['context_num_words'], 'x2_bi_rnn', params.skim_2, 2)

        input_start = tf.layers.dense(
            tf.concat([x2, word_feat], 2) if params.out_word_feat else x2, 1, name='logits0')
        input_end = tf.layers.dense(
            tf.concat([x2, word_feat], 2) if params.out_word_feat else x2, 1, name='logits1')
        logits_start = tf_utils.exp_mask(
            tf.squeeze(input_start, 2), features['context_num_words'])
        logits_end = tf_utils.exp_mask(
            tf.squeeze(input_end, 2), features['context_num_words'])

        if (params.skim_embed or params.skim_1 or params.skim_2):
            keys = list(tensors.keys())
            for key in keys:
                if key.startswith('q_'):
                    id_ = key[2:]
                    # print(id_)
                    skim_rates, tensors['avg_logq_%s' % (id_)], tensors['choice_%s' % (id_)] = \
                        _calc_logq(tensors['q_%s' % (id_)], features['context_num_words'], params, training)
                    for i in range(len(skim_rates)):
                        tensors['skim_rate_layer_%s_cell_%d' % (id_, i+1)] = skim_rates[i]

                    # tensors['skim_rate_context_%s' % (id_)], tensors['logq_%s' % id_], tensors['avg_logq_%s' % (id_)], tensors[
                    #     'choice_%s' % (id_)] = \
                    #     _calc_logq(tensors['x_%s' % (id_)], features['context_num_words'], params, training)

            avg_logqs = [tensors[key] for key in tensors.keys() if key.startswith('avg_logq_')]
            tensors['avg_logq'] = sum(avg_logqs) / len(avg_logqs)

        return logits_start, logits_end, tensors


def _get_skim_cells(params, training):
    d = params.hidden_size
    small_sizes = params.small_hidden_sizes

    reuse = False
    global_step = tf.train.get_global_step()

    temp = tf.maximum(0.5, tf.exp(
        -params.temp_decay * tf.cast((global_step / params.temp_period) * params.temp_period, 'float')))

    # threshold = get_threshold(global_step,
    #                           params.threshold_period) if training and params.threshold_period > 0 else params.threshold
    '''
    if params.vcgru:
        big_cell_fw = BasicLSTMCell(d, reuse=reuse)
        big_cell_bw = BasicLSTMCell(d, reuse=reuse)
        sharpness = tf.minimum(1.0, 0.1 * tf.cast(global_step, 'float'))
        cell_fw = VCGRUWrapper(big_cell_fw, sharpness, training, random.random() * 0.3 + 0.2)
        cell_bw = VCGRUWrapper(big_cell_bw, sharpness, training, random.random() * 0.3 + 0.2)
        return cell_fw, cell_bw
    '''

    ref_cell = BasicLSTMCell(d, reuse=reuse)

    big_cell_fw = BasicLSTMCell(d, reuse=reuse)
    big_cell_bw = BasicLSTMCell(d, reuse=reuse)
    nested_cells_fw = [big_cell_fw]
    nested_cells_bw = [big_cell_bw]
    for dd in small_sizes:
        nested_cells_fw.append(BasicLSTMCell(dd, reuse=reuse))
        nested_cells_bw.append(BasicLSTMCell(dd, reuse=reuse))

    # pos = 0

    cell_fw = NestedLSTMWrapper(ref_cell, nested_cells_fw, temp, training)
    cell_bw = NestedLSTMWrapper(ref_cell, nested_cells_bw, temp, training)
    return cell_fw, cell_bw


def _calc_logq(q_logits, x_len, params, training):
    x_mask = tf.sequence_mask(x_len)
    # global_step = tf.train.get_global_step()

    def _get_skim_rate():

        choice = tf.cast(tf.argmax(tf.slice(tf.nn.softmax(q_logits), [0, 0, 1], [-1, -1, -1]), axis=2), 'int32')

        # choice = tf.squeeze(
        #     tf.cast(tf.greater(tf.slice(tf.nn.softmax(q_logits), [0, 0, 1], [-1, -1, -1]), threshold), 'int32'), 2)


        # TODO: adhoc -> scalable
        counts = []
        small_sizes = params.small_hidden_sizes

        for i in range(len(small_sizes)):
            counts.append(tf.reduce_sum(tf.cast(tf.equal(choice, i+1), 'float32') * tf.cast(x_mask, 'float32'), 1))

        # q_count = tf.reduce_sum(tf.cast(tf.greater(choice, 0), 'float32') * tf.cast(x_mask, 'float32'), 1)  # [N]

        skim_rate = []
        for count in counts:
            skim_rate.append(count / tf.cast(x_len, 'float32'))

        # skim_rate = q_count / tf.cast(x_len, 'float32')
        return choice, skim_rate

    # if params.infer or params.threshold_period == 0:
    choice, skim_rate = _get_skim_rate()
    # else:
    #     choices_and_skim_rates = []
    #     for threshold in [0.1, 0.3, 0.5, 0.7, 0.9]:
    #         choices_and_skim_rates.append(_get_skim_rate(threshold))
    #     choice = choices_and_skim_rates[2][0]
    #     skim_rate_list = [tf.expand_dims(i[1], 1) for i in choices_and_skim_rates]
    #     skim_rate = tf.reduce_mean(tf.concat(skim_rate_list, 1), 1)

    logp = log(tf.nn.softmax(q_logits))

    small_sizes = params.small_hidden_sizes
    logq_ = []
    for i in range(len(small_sizes)):
        logq_.append(tf.slice(logp, [0, 0, i+1], [-1, -1, 1]) + log(params.num_cells - 1))  # [N, J, C-1]

    logq = []
    for l in logq_:
        logq.append(tf.reduce_sum(l, 2) * tf.cast(x_mask, 'float')) # [N, J]

    weight_sum_logq = 0.0
    divisor = 0
    for i in range(len(logq)):
        weight_sum_logq += logq[i] * (i+1)

    weight_sum_logq /= (len(logq) * (len(logq) + 1) / 2)

    avg_logq = tf.reduce_sum(weight_sum_logq, 1) / tf.cast(x_len, 'float')  # [N]

    return skim_rate, avg_logq, choice

    # logq_ = tf.slice(log(tf.nn.softmax(q_logits)), [0, 0, 1], [-1, -1, -1])  # [N, J, C-1]
    # logq = tf.reduce_sum(logq_, 2) * tf.cast(x_mask, 'float')  # [N, J]
    # avg_logq = tf.reduce_sum(logq, 1) / tf.cast(x_len, 'float')  # [N]
    # return skim_rate, logq, avg_logq, choice


def log(x):
    return tf.log(x + 1e-12)


# def remainer(a, b):
#     return a - tf.multiply(tf.floordiv(a, b), b)


# def get_threshold(global_step, threshold_period=10):
#     global_step = tf.cast(global_step, 'int32')
#     thresholds = tf.Variable([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0], trainable=False)
#     return tf.slice(thresholds, [remainer(tf.floordiv(global_step, tf.cast(threshold_period / 10, 'int32')), 11)], [-1])

