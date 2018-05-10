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
"""Experiments for kernel vs feature map in SQuAD.

`feature` model does not allow any interaction between question and context
except at the end, where the dot product (or L1/L2 distance) is used to get the
answer.
`kernel` model allows any interaction between question and context
(e.g. cross attention).
This script is for establishing baseline for both feature and kernel models.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from collections import defaultdict
import json
import os
import numpy as np
from IPython import embed

import tensorflow as tf
from tqdm import tqdm
import tensorflow.contrib.learn as learn

# This is required for importing google specific flags:
# `output_dir`, `schedule`
# (`learn` above is not sufficient). Will need to add these flags when
# removing this import for open-sourcing.
from tensorflow.contrib.learn import learn_runner

import squad_data
from common_model import get_loss
from common_model import get_pred_ops
from common_model import get_train_op
from model import model as kernel_model

from color import Color

tf.flags.DEFINE_string('data', 'squad', 'data')
tf.flags.DEFINE_integer('emb_size', 200, 'embedding size')
tf.flags.DEFINE_integer('glove_size', 200, 'GloVe size')
tf.flags.DEFINE_integer('hidden_size', 100, 'hidden state size')
tf.flags.DEFINE_integer('embed_hidden_size', 0, 'hidden state size for embedding. same as hidden_size if 0')
tf.flags.DEFINE_integer('num_train_steps', 15000, 'num train steps')
tf.flags.DEFINE_integer('num_eval_steps', 50, 'num eval steps')
tf.flags.DEFINE_boolean('draft', False, 'draft?')
tf.flags.DEFINE_integer('batch_size', 64, 'batch size')
tf.flags.DEFINE_float('dropout_rate', 0.2,
                      'dropout rate, applied to the input of LSTMs.')
tf.flags.DEFINE_string('root_data_dir', 'prepro', 'root data dir')
tf.flags.DEFINE_integer('save_checkpoints_steps', 500, '')
tf.flags.DEFINE_integer('num_eval_delay_secs', 1, 'eval delay secs')
tf.flags.DEFINE_boolean('shuffle_examples', False, 'Use shuffle example queue?')
tf.flags.DEFINE_boolean('shuffle_files', True, 'Use shuffle file queue?')
tf.flags.DEFINE_string('model', 'kernel', '`feature` or `kernel`.')
tf.flags.DEFINE_boolean('oom_test', False, 'Performs out-of-memory test')
tf.flags.DEFINE_string(
    'dist', 'dot', 'Distance function for feature model. `dot`, `l1` or `l2`.')
tf.flags.DEFINE_string('opt', 'Adam', 'optimizer')
tf.flags.DEFINE_float('learning_rate', 0.001,
                      '(Initial) learning rate for optimizer')
tf.flags.DEFINE_boolean(
    'infer', False,
    'If `True`, obtains and saves predictions for the test dataset '
    'at `answers_path`.')
tf.flags.DEFINE_string('answers_path', '',
                       'The path for saving predictions on test dataset. '
                       'If not specified, saves in `restore_dir` directory.')
tf.flags.DEFINE_float('clip_norm', 0, 'Clip norm threshold, 0 for no clip.')
tf.flags.DEFINE_integer(
    'restore_step', 0,
    'The global step for which the model is restored in the beginning. '
    '`0` for the most recent save file.')
tf.flags.DEFINE_float(
    'restore_decay', 1.0,
    'The decay rate for exponential moving average of variables that '
    'will be restored upon eval or infer. '
    '`1.0` for restoring variables without decay.')
tf.flags.DEFINE_string(
    'ema_decays', '',
    'List of exponential moving average (EMA) decay rates (float) '
    'to track for variables during training. Values are separated by commas.')
tf.flags.DEFINE_string(
    'restore_dir', '',
    'Directory from which variables are restored. If not specfied, `output_dir`'
    'will be used instead. For inference mode, this needs to be specified.')
tf.flags.DEFINE_string('model_id', 'm00', 'Model id.')
tf.flags.DEFINE_string('glove_dir', '/home/kyungissac/data/glove',
                       'GloVe dir.')
tf.flags.DEFINE_boolean('merge', False, 'If `True`, merges answers from same '
                                        'paragraph that were split in preprocessing step.')
tf.flags.DEFINE_integer('queue_capacity', 5000, 'Input queue capacity.')
tf.flags.DEFINE_integer('min_after_dequeue', 1000, 'Minimum number of examples '
                                                   'after queue dequeue.')
tf.flags.DEFINE_integer('max_answer_size', 15, 'Max number of answer words.')
tf.flags.DEFINE_string('restore_scopes', '', 'Restore scopes, separated by ,.')
tf.flags.DEFINE_boolean('reg_gen', True, 'Whether to regularize training '
                                         'with question generation (reconstruction) loss.')
tf.flags.DEFINE_float('reg_cf', 3.0, 'Regularization initial coefficient.')
tf.flags.DEFINE_float('reg_half_life', 6000, 'Regularization decay half life. '
                                             'Set it to very high value to effectively disable decay.')
tf.flags.DEFINE_integer('max_gen_length', 32, 'During inference, maximum '
                                              'length of generated question.')

# Below are added for third party.
tf.flags.DEFINE_string('schedule', 'train_and_evaluate',
                       'schedule for learn_runner.')
tf.flags.DEFINE_string('output_dir', '/tmp/squad_ckpts',
                       'Output directory for saving model.')

# Below are added for Skim-RNN.

tf.flags.DEFINE_boolean('skim_embed', False, 'If `True`, use Skim-RNN instead of plain RNN')
tf.flags.DEFINE_boolean('skim_1', False, 'If `True`, use Skim-RNN instead of plain RNN')
tf.flags.DEFINE_boolean('skim_2', False, 'If `True`, use Skim-RNN instead of plain RNN')

# For multi-level skim cell
tf.flags.DEFINE_string('small_hidden_sizes', '[20, 10, 5, 0]', 'small hidden sizes of Skim-RNN')
tf.flags.DEFINE_integer('num_cells', 5, 'Number of lstm cells in a skim cell')


tf.flags.DEFINE_float('temp_period', 100, '')
tf.flags.DEFINE_float('temp_decay', 1e-3, 'temperature in gumbel-softmax')
tf.flags.DEFINE_float('p_decay', 0.01, 'decay rate for preferred lstm choice loss.')
tf.flags.DEFINE_float('embed_p_decay', 0.01, 'decay rate for preferred lstm choice loss. (for embed)')
tf.flags.DEFINE_integer('p_pref', 1, 'preferred lstm cell, to be used for choice loss.')

# tf.flags.DEFINE_float('threshold', 0.5, 'threshold for skimming')
# tf.flags.DEFINE_integer('threshold_period', 0, 'threshold period')

tf.flags.DEFINE_boolean('emb_word_feat', True, 'use word feature vector (one-hot)')
tf.flags.DEFINE_boolean('word_feat', True, 'use word feature vector (one-hot)')
tf.flags.DEFINE_boolean('out_word_feat', True, 'use word feature vector (one-hot)')
tf.flags.DEFINE_boolean('big2nested', False, '')
tf.flags.DEFINE_boolean('small2nested', False, '')
tf.flags.DEFINE_boolean('only_train_small', False, '')
tf.flags.DEFINE_boolean('only_train_big', False, '')
tf.flags.DEFINE_boolean('vcgru', False, '')

tf.flags.DEFINE_float('sparsity_decay', 0, 'hyperparem for sparsity')
tf.flags.DEFINE_float('sparsity_th', 0.01, 'threshold for sparsity')

FLAGS = tf.flags.FLAGS
tf.logging.set_verbosity(tf.logging.INFO)


def model_fn(features, targets, mode, params):
    """Model function to be used for `Experiment` object.

    Should not access `flags.FLAGS`.

    Args:
      features: a dictionary of feature tensors.
      targets: a dictionary of target tensors.
      mode: `learn.ModeKeys.TRAIN` or `learn.ModeKeys.EVAL`.
      params: `HParams` object.
    Returns:
      `ModelFnOps` object.
    Raises:
      ValueError: rasied if `params.model` is not an appropriate value.
    """
    with tf.variable_scope('model'):
        data = _get_data(params.data)
        if params.model == 'feature':
            logits_start, logits_end, tensors = feature_model(
                features, mode, params)
        elif params.model == 'kernel':
            logits_start, logits_end, tensors = kernel_model(
                features, mode, params)
        else:
            raise ValueError(
                '`%s` is an invalid argument for `model` parameter.' % params.model)
        no_answer_bias = tf.get_variable('no_answer_bias', shape=[], dtype='float')
        no_answer_bias = tf.tile(
            tf.reshape(no_answer_bias, [1, 1]),
            [tf.shape(features['context_words'])[0], 1])

        predictions = get_pred_ops(features, params, logits_start, logits_end,
                                   no_answer_bias)
        predictions.update(tensors)
        predictions.update(features)

    if mode == learn.ModeKeys.INFER:
        eval_metric_ops, loss = None, None
    else:
        eval_metric_ops = data.get_eval_metric_ops(targets, predictions, tensors)
        loss = get_loss(targets['word_answer_starts'], targets['word_answer_ends'],
                        logits_start, logits_end, no_answer_bias, tensors, params)

    emas = {
        decay: tf.train.ExponentialMovingAverage(
            decay=decay, name='EMA_%f' % decay)
        for decay in params.ema_decays
    }

    ema_ops = [ema.apply() for ema in emas.values()]

    restore_vars = []
    for restore_scope in params.restore_scopes:
        restore_vars.extend(
            tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, restore_scope))
    # FIXME: indentation
    if params.restore_dir and not tf.gfile.Exists(params.output_dir):
        assert params.restore_scopes
        checkpoint_dir = params.restore_dir
        print("*" * 20)
        print(params.restore_dir, params.restore_step)
        if params.restore_step:
            checkpoint_dir = os.path.join(params.restore_dir,
                                          'model.ckpt-%d' % params.restore_step)

        def _rename(name):
            if not (params.big2nested or params.small2nested):
                return name

            for rnn in ['x_bi_rnn_0', 'x1_bi_rnn', 'x2_bi_rnn', 'x3_bi_rnn']:
                plain_name = 'model/kernel_model/%s/bidirectional_rnn/' % rnn
                if name.startswith(plain_name):
                    if name[len(plain_name) + 2:].startswith('/nested_rnn_cell/') and not params.small2nested:
                        di = name[len(plain_name)]
                        assert di in ['f', 'b'], (name, di)
                        ty = name[len(plain_name) + 2 + len('/nested_rnn_cell/'):]
                        # For multi skim cell.
                        if ty.startswith('dense') or ty.startswith('cell_1') or ty.startswith('cell_2')\
                                or ty.startswith('cell_3') or ty.startswith('cell_4'):
                            return None
                        assert ty.startswith('cell_0/basic_lstm_cell/'), (name, ty)
                        ty_ = ty[len('cell_0/basic_lstm_cell/'):]
                        assert ty_ in ['kernel', 'bias'], (name, ty_)
                        print(name)
                        return plain_name + '%sw/basic_lstm_cell/%s' % (di, ty_)
                    elif params.small2nested and name.startswith('model/kernel_model/%s/dense' % rnn):
                        print(name)
                        return None
            if params.small2nested and name.startswith('model/kernel_model/logits'):
                return None
            return name

        assignment_map = {_rename(var.op.name): var for var in restore_vars if _rename(var.op.name) is not None}
        tf.contrib.framework.init_from_checkpoint(checkpoint_dir, assignment_map)

    if mode == learn.ModeKeys.TRAIN:
        var_list = restore_vars
        if params.only_train_small or params.only_train_big:
            no_train_list = []

            def get_var_in_lstm(i):
                for di in ['fw', 'bw']:
                    # model layer - number of layers
                    for no in ['1', '2']:
                        for ty in ['kernel', 'bias']:
                            no_train_list.append(
                                'model/kernel_model/x%s_bi_rnn/bidirectional_rnn/%s/nested_rnn_cell/cell_%d/basic_lstm_cell/%s' % (
                                no, di, i, ty))

            def _filter(var, i):
                get_var_in_lstm(i)
                for v in no_train_list:
                    if var.op.name.startswith(v):
                        return True
                return False

            if params.only_train_small:
                print("only train small RNN")
                var_list = [v for v in var_list if _filter(v, 1)]
            else:
                print("only train big RNN")
                var_list = [v for v in var_list if _filter(v, 0)]
            print([var.op.name[len('model/kernel_model/'):] for var in var_list])
        else:
            print("Train all variables")
        train_op = get_train_op(
            loss,
            var_list=var_list,
            opt=params.opt,
            learning_rate=params.learning_rate,
            clip_norm=params.clip_norm,
            post_ops=ema_ops)
    else:
        if params.restore_decay < 1.0:
            ema = emas[params.restore_decay]
            assign_ops = []
            for var in tf.trainable_variables():
                assign_op = tf.assign(var, ema.average(var))
                assign_ops.append(assign_op)
            with tf.control_dependencies(assign_ops):
                for key, val in predictions.items():
                    predictions[key] = tf.identity(val)
        train_op = None

    return learn.ModelFnOps(
        mode=mode,
        predictions=predictions,
        loss=loss,
        train_op=train_op,
        eval_metric_ops=eval_metric_ops)


def _experiment_fn(run_config, hparams):
    """Outputs `Experiment` object given `output_dir`.

    Args:
      run_config: `EstimatorConfig` object fo run configuration.
      hparams: `HParams` object that contains hyperparameters.

    Returns:
      `Experiment` object
    """
    estimator = learn.Estimator(
        model_fn=model_fn, config=run_config, params=hparams)

    num_train_steps = 1 if FLAGS.oom_test else FLAGS.num_train_steps
    num_eval_steps = 1 if FLAGS.oom_test else FLAGS.num_eval_steps

    data = _get_data(hparams.data)

    return learn.Experiment(
        estimator=estimator,
        train_input_fn=_get_train_input_fn(data),
        eval_input_fn=_get_eval_input_fn(data),
        train_steps=num_train_steps,
        eval_steps=num_eval_steps,
        eval_delay_secs=FLAGS.num_eval_delay_secs)


def _get_data(data_name):
    return squad_data


def _get_train_input_fn(data):
    """Get train input function."""
    train_input_fn = data.get_input_fn(
        FLAGS.root_data_dir,
        FLAGS.glove_dir,
        'train',
        FLAGS.batch_size,
        FLAGS.glove_size,
        shuffle_files=FLAGS.shuffle_files,
        shuffle_examples=FLAGS.shuffle_examples,
        queue_capacity=FLAGS.queue_capacity,
        min_after_dequeue=FLAGS.min_after_dequeue,
        oom_test=FLAGS.oom_test)
    return train_input_fn


def _get_eval_input_fn(data):
    """Get eval input function."""
    eval_input_fn = data.get_input_fn(
        FLAGS.root_data_dir,
        FLAGS.glove_dir,
        'dev',
        FLAGS.batch_size,
        FLAGS.glove_size,
        shuffle_files=True,
        shuffle_examples=True,
        queue_capacity=FLAGS.queue_capacity,
        min_after_dequeue=FLAGS.min_after_dequeue,
        num_epochs=1,
        oom_test=FLAGS.oom_test)
    return eval_input_fn


def _get_test_input_fn(data):
    """Get test input function."""
    # TODO(seominjoon) For now, test input is same as eval input (dev).
    test_input_fn = data.get_input_fn(
        FLAGS.root_data_dir,
        FLAGS.glove_dir,
        'dev',
        FLAGS.batch_size,
        FLAGS.glove_size,
        shuffle_files=FLAGS.shuffle_files,
        shuffle_examples=FLAGS.shuffle_examples,
        queue_capacity=FLAGS.queue_capacity,
        min_after_dequeue=FLAGS.min_after_dequeue,
        num_epochs=1,
        oom_test=FLAGS.oom_test)
    return test_input_fn


def _get_config():
    """Get configuration object  for `Estimator` object.

    For open-soucing, `EstimatorConfig` has been replaced with `RunConfig`.
    Depends on `flags.FLAGS`, and should not be used outside of this main script.

    Returns:
      `EstimatorConfig` object.
    """
    config = learn.RunConfig(
        model_dir=FLAGS.restore_dir if FLAGS.infer else FLAGS.output_dir,
        keep_checkpoint_max=0,  # Keep all checkpoints.
        save_checkpoints_steps=FLAGS.save_checkpoints_steps)
    return config


def _get_hparams():
    """Model-specific hyperparameters go here.

    All model parameters go here, since `model_fn()` should not access
    `flags.FLAGS`.
    Depends on `flags.FLAGS`, and should not be used outside of this main script.

    Returns:
      `HParams` object.
    """
    hparams = tf.contrib.training.HParams()
    hparams.data = FLAGS.data
    data = _get_data(hparams.data)
    data_hparams = data.get_params(FLAGS.root_data_dir)
    hparams.infer = FLAGS.infer
    hparams.vocab_size = data_hparams['vocab_size']
    hparams.char_vocab_size = data_hparams['char_vocab_size']
    hparams.batch_size = FLAGS.batch_size
    hparams.hidden_size = FLAGS.hidden_size
    hparams.embed_hidden_size = FLAGS.hidden_size if FLAGS.embed_hidden_size == 0 else FLAGS.embed_hidden_size
    hparams.emb_size = FLAGS.emb_size
    hparams.dropout_rate = FLAGS.dropout_rate
    hparams.dist = FLAGS.dist
    hparams.learning_rate = FLAGS.learning_rate
    hparams.model = FLAGS.model
    hparams.restore_dir = FLAGS.restore_dir
    hparams.output_dir = FLAGS.output_dir
    hparams.clip_norm = FLAGS.clip_norm
    hparams.opt = FLAGS.opt
    hparams.restore_decay = FLAGS.restore_decay
    if FLAGS.ema_decays:
        hparams.ema_decays = list(map(float, FLAGS.ema_decays.split(',')))
    else:
        hparams.ema_decays = []
    hparams.restore_step = FLAGS.restore_step
    hparams.model_id = FLAGS.model_id
    hparams.max_answer_size = FLAGS.max_answer_size
    hparams.restore_scopes = FLAGS.restore_scopes.split(',')
    hparams.glove_size = FLAGS.glove_size

    # Regularization by Query Generation (reconstruction) parameters.
    hparams.reg_gen = FLAGS.reg_gen
    hparams.reg_cf = FLAGS.reg_cf
    hparams.reg_half_life = FLAGS.reg_half_life

    # For Skim-RNN
    hparams.skim_embed = FLAGS.skim_embed
    hparams.skim_1 = FLAGS.skim_1
    hparams.skim_2 = FLAGS.skim_2

    # For multi-level skim cells
    hparams.small_hidden_sizes = json.loads(FLAGS.small_hidden_sizes)
    hparams.num_cells = FLAGS.num_cells

    # hparams.threshold = FLAGS.threshold
    # hparams.threshold_period = FLAGS.threshold_period

    hparams.temp_period = FLAGS.temp_period
    hparams.temp_decay = FLAGS.temp_decay
    hparams.p_decay = FLAGS.p_decay
    hparams.embed_p_decay = FLAGS.embed_p_decay
    hparams.p_pref = FLAGS.p_pref

    hparams.emb_word_feat = FLAGS.emb_word_feat
    hparams.word_feat = FLAGS.word_feat
    hparams.out_word_feat = FLAGS.out_word_feat
    hparams.big2nested = FLAGS.big2nested
    hparams.small2nested = FLAGS.small2nested
    hparams.only_train_small = FLAGS.only_train_small
    hparams.only_train_big = FLAGS.only_train_big
    hparams.vcgru = FLAGS.vcgru
    hparams.sparsity_decay = FLAGS.sparsity_decay
    hparams.sparsity_th = FLAGS.sparsity_th

    return hparams


def train_and_eval():
    """Train and eval routine."""
    learn_runner.run(
        experiment_fn=_experiment_fn,
        schedule=FLAGS.schedule,
        run_config=_get_config(),
        hparams=_get_hparams())


def _set_ckpt():
    # TODO(seominjoon): This is adhoc. Need better ckpt loading during inf.
    if FLAGS.restore_step:
        path = os.path.join(FLAGS.restore_dir, 'checkpoint')
        with tf.gfile.GFile(path, 'w') as fp:
            fp.write('model_checkpoint_path: "model.ckpt-%d"\n' % FLAGS.restore_step)


def infer():
    """Inference routine, outputting answers to `FLAGS.answers_path`."""
    _set_ckpt()
    params = _get_hparams()
    estimator = learn.Estimator(
        model_fn=model_fn, config=_get_config(), params=params)
    # estimator.evaluate(
    #    input_fn=_get_test_input_fn(_get_data(params.data)))
    # return
    predictions = estimator.predict(
        input_fn=_get_test_input_fn(_get_data(params.data)))
    global_step = estimator.get_variable_value('global_step')
    answer_path = FLAGS.answers_path or os.path.join(FLAGS.restore_dir,
                                                     'answers-%d-%.2f.json' % (global_step, params.threshold))
    choice_path = os.path.join(FLAGS.restore_dir,
                               'choices-%d-%.2f.json' % (global_step, params.threshold))
    answer_dict = {'no_answer_prob': {}, 'answer_prob': {}, 'choice': {}}
    skim = params.skim_embed or params.skim_1 or params.skim_2
    choice_dict = {}
    for prediction in tqdm(predictions):
        id_ = prediction['id'].decode('utf-8')
        context_words = [str(word, encoding="utf-8")
                         for word in prediction['context_words'].tolist()]
        question = prediction['question'].decode('utf-8')
        gt_answers = [a.decode('utf-8') for a in prediction['answers'] if len(a) > 0]
        answer = prediction['a'].decode('utf-8')
        answer_dict[id_] = {
            'context_words': context_words,
            'question': question,
            'gt_answers': gt_answers,
            'answer': answer,
            'answer_prob': prediction['answer_prob'].tolist(),
            'no_answer_prob': prediction['no_answer_prob'].tolist()
        }
        if skim:
            choices = {}
            for key in prediction:
                if key.startswith('choice'):
                    choices[key] = prediction[key].tolist()
            choice_dict[id_] = {
                'context_words': context_words,
                'question': question,
                'answer': answer,
                'gt_answers': gt_answers,
                'choice': choices}
        if FLAGS.oom_test:
            break
    if FLAGS.merge:
        new_answer_dict = defaultdict(list)
        context_dict, question_dict, gt_dict = {}, {}, {}
        for id_, dic in answer_dict.items():
            answer = dic['answer']
            answer_prob = dic['answer_prob']
            id_ = id_.split(' ')[0]  # retrieve true id
            new_answer_dict[id_].append([answer_prob, answer])
            context_dict[id_] = dic['context_words']
            question_dict[id_] = dic['question']
            gt_dict[id_] = dic['gt_answers']
        answer_dict = {
            id_: {
                'context': context_dict[id_],
                'question': question_dict[id_],
                'gt_answrs': gt_dict[id_],
                'answer': max(each, key=lambda pair: pair[0])[1]
            }
            for id_, each in new_answer_dict.items()
        }
    with tf.gfile.GFile(answer_path, 'w') as fp:
        print(answer_path)
        json.dump(answer_dict, fp)
    if skim:
        with tf.gfile.GFile(choice_path, 'w') as fp:
            print(choice_path)
            json.dump(choice_dict, fp)


def main(_):
    if FLAGS.infer:
        infer()
    else:
        train_and_eval()


if __name__ == '__main__':
    tf.app.run()
