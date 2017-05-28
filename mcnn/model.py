import math
from abc import abstractmethod
from pathlib import Path
from typing import List, Dict

import numpy as np
import tensorflow as tf
from tensorflow.contrib.layers import xavier_initializer


class Model:

    def __init__(self, sample_length: int, learning_rate: float, num_classes: int, batch_size: int):
        self.global_step = tf.Variable(0, name='global_step', trainable=False)

        self.batch_size = batch_size
        self.sample_length = sample_length
        self.num_classes = num_classes
        self.input = tf.placeholder(tf.float32, shape=(None, sample_length), name='input')
        self.dynamic_batch_size = tf.shape(self.input)[0]
        self.labels = tf.placeholder(tf.int64, shape=(None,), name='labels')

        input_2d = tf.reshape(self.input, shape=(self.dynamic_batch_size, 1, sample_length, 1))

        self.logits = self._create_network(input_2d)

        with tf.variable_scope('training'):
            cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.labels, logits=self.logits,
                                                                           name='crossentropy')
            self.loss = tf.reduce_mean(cross_entropy, name='crossentropy_mean')
            tf.summary.scalar('loss', self.loss)
            optimizer = tf.train.RMSPropOptimizer(learning_rate)
            self.train_op = optimizer.minimize(self.loss, global_step=self.global_step, name='train_op')

        with tf.variable_scope('evaluation'):
            correct = tf.nn.in_top_k(self.logits, self.labels, 1, name='correct')
            self.correct_count = tf.reduce_sum(tf.cast(correct, tf.int32), name='correct_count')

        self.init = tf.global_variables_initializer()
        self.summary = tf.summary.merge_all()
        self.summary_writer = None
        self.saver = tf.train.Saver()

    @abstractmethod
    def _create_network(self, input_2d: tf.Tensor) -> tf.Tensor:
        raise NotImplementedError()

    def create(self, session: tf.Session):
        print('Created model with fresh parameters.')
        session.run(self.init)

    def setup_summary_writer(self, session: tf.Session, log_dir: Path):
        self.summary_writer = tf.summary.FileWriter(str(log_dir), graph=session.graph)

    def restore(self, session: tf.Session, checkpoint_dir: Path):
        ckpt = tf.train.get_checkpoint_state(str(checkpoint_dir))
        if ckpt and ckpt.model_checkpoint_path:
            print('Reading model parameters from {}'.format(ckpt.model_checkpoint_path))
            self.saver.restore(session, ckpt.model_checkpoint_path)
        else:
            raise FileNotFoundError('No checkpoint for evaluation found')

    def restore_or_create(self, session: tf.Session, checkpoint_dir: Path):
        try:
            self.restore(session, checkpoint_dir)
        except FileNotFoundError:
            self.create(session)

    def save(self, session: tf.Session, checkpoint_dir: Path):
        self.saver.save(session, str(checkpoint_dir / 'mcnn'), global_step=self.global_step.eval(session))

    def step(self, session: tf.Session, feed_dict: Dict, loss=False, train=False, logits=False, correct_count=False,
             update_summary=False):
        output_feed = [self.global_step]

        if loss:
            output_feed.append(self.loss)

        if train:
            output_feed.append(self.train_op)

        if logits:
            output_feed.append(self.logits)

        if correct_count:
            output_feed.append(self.correct_count)

        if update_summary:
            output_feed.append(self.summary)

        results = session.run(output_feed, feed_dict=feed_dict)

        if update_summary:
            summary_result = results[-1]
            step = results[0]
            self.summary_writer.add_summary(summary_result, global_step=step)
            return results[:-1]

        return results

    @classmethod
    def _full_fullyconnected(cls, input: tf.Tensor, input_size: int, layer_size: int, activation=None) -> tf.Tensor:
        with tf.variable_scope(None, 'fullyconnected'):
            weight = tf.get_variable('weight', shape=(input_size, layer_size), dtype=tf.float32,
                                     initializer=xavier_initializer())
            bias = tf.get_variable('bias', shape=(layer_size,), dtype=tf.float32,
                                   initializer=tf.constant_initializer(0.0))
            output = tf.matmul(input, weight)
            output = tf.nn.bias_add(output, bias)
            if activation:
                output = activation(output, name='activation')

        return output


class McnnConfiguration():

    def __init__(self, downsample_strides: List[int], smoothing_window_sizes: List[int],
                 pooling_factor: int, channel_count: int, local_filter_width: int, full_filter_width: int,
                 layer_size: int, full_pool_size: int):
        self.local_filter_width = local_filter_width
        self.channel_count = channel_count
        self.pooling_factor = pooling_factor
        self.local_filter_width = local_filter_width
        self.smoothing_window_sizes = smoothing_window_sizes
        self.downsample_strides = downsample_strides
        self.full_pool_size = full_pool_size
        self.full_filter_width = full_filter_width
        self.layer_size = layer_size


class AutoCnnModel(Model):
    def __init__(self, sample_length: int, learning_rate: float, num_classes: int, filter_width: int, batch_size: int):
        self.filter_width = filter_width
        super().__init__(sample_length, learning_rate, num_classes, batch_size)

    def _create_network(self, input_2d: tf.Tensor) -> tf.Tensor:
        output = input_2d

        for layer_index in range(int(math.log2(self.sample_length))):
            with tf.variable_scope('conv_{}'.format(layer_index)):
                channels_in = 1 if layer_index == 0 else 256
                channels_out = 256
                filter = tf.get_variable('filter', shape=(1, self.filter_width, channels_in, channels_out),
                                         dtype=tf.float32, initializer=xavier_initializer())
                bias = tf.get_variable('bias', shape=(channels_out,), dtype=tf.float32,
                                       initializer=tf.constant_initializer(0.0))
                output = tf.nn.conv2d(output, filter, strides=[1, 1, 2, 1], padding='SAME')
                output = tf.nn.bias_add(output, bias)
                output = tf.nn.relu(output, name='relu')

        output = tf.reshape(output, shape=(self.dynamic_batch_size, -1))
        output = self._full_fullyconnected(output, 512, 256, activation=tf.nn.relu)
        logits = self._full_fullyconnected(output, 256, self.num_classes)
        logits = tf.identity(logits, name='logits')

        return logits


class McnnModel(Model):

    def __init__(self, batch_size: int, learning_rate: float, sample_length: int,
                 mcnn_configuration: McnnConfiguration, num_classes: int):

        self.cfg = mcnn_configuration

        if self.cfg.channel_count % len(self.cfg.smoothing_window_sizes) != 0:
            raise ValueError('channel_count must be a multiple of number of smoothing window sizes')

        if self.cfg.channel_count % len(self.cfg.downsample_strides) != 0:
            raise ValueError('channel_count must be a multiple of number of downsamples')

        super().__init__(sample_length, learning_rate, num_classes, batch_size)

    def _create_network(self, input_2d: tf.Tensor) -> tf.Tensor:

        with tf.variable_scope('local'):
            local_outputs = [
                self._local_convolve(input_2d, self.cfg.local_filter_width, self.cfg.channel_count, self.cfg.pooling_factor),
                self._local_smoothed_convolve(input_2d, self.cfg.local_filter_width, self.cfg.channel_count,
                                              self.cfg.smoothing_window_sizes, self.cfg.pooling_factor),
                self._local_downsampled_convolve(input_2d, self.cfg.local_filter_width, self.cfg.channel_count,
                                                 self.cfg.downsample_strides, self.cfg.pooling_factor)
            ]
            local_output = tf.concat(local_outputs, axis=-1, name='local_concat')

        with tf.variable_scope('full'):
            full_convolved = self._full_convolve(local_output, 3 * self.cfg.channel_count,
                                                 self.cfg.channel_count, self.cfg.full_filter_width, self.cfg.full_pool_size)
            sequence_length = self.cfg.pooling_factor // self.cfg.full_pool_size
            full_convolved_reshaped = tf.reshape(full_convolved, shape=(self.dynamic_batch_size, -1))
            connected = self._full_fullyconnected(full_convolved_reshaped, self.cfg.channel_count * sequence_length,
                                                  self.cfg.layer_size, tf.nn.relu)
            logits = self._full_fullyconnected(connected, self.cfg.layer_size, self.num_classes)
            logits = tf.identity(logits, name='logits')

        return logits

    @classmethod
    def _full_convolve(cls, input: tf.Tensor, channels_in: int, channels_out: int,
                       filter_width: int, pooling: int) -> tf.Tensor:
        with tf.variable_scope('conv'):
            filter = tf.get_variable('filter', shape=(1, filter_width, channels_in, channels_out), dtype=tf.float32,
                                     initializer=xavier_initializer())
            bias = tf.get_variable('bias', shape=(channels_out,), dtype=tf.float32,
                                   initializer=tf.constant_initializer(0.0))
            output = tf.nn.conv2d(input, filter, strides=[1, 1, 1, 1], padding='SAME')
            output = tf.nn.bias_add(output, bias)
            output = tf.nn.relu(output, name='relu')
            size = [1, 1, pooling, 1]
            output = tf.nn.max_pool(output, ksize=size, strides=size, padding='SAME', name='maxpool')

        return output

    @classmethod
    def _local_convolve(cls, input: tf.Tensor, filter_width: int, channels_out: int, pooling_factor: int) -> tf.Tensor:
        with tf.variable_scope('conv'):
            filter = tf.get_variable('filter', shape=(1, filter_width, 1, channels_out), dtype=tf.float32,
                                     initializer=xavier_initializer())
            bias = tf.get_variable('bias', shape=(channels_out,), dtype=tf.float32,
                                   initializer=tf.constant_initializer(0.0))
            output = tf.nn.conv2d(input, filter, strides=[1, 1, 1, 1], padding='SAME')
            output = tf.nn.bias_add(output, bias)
            output = tf.nn.relu(output, name='relu')
            output = cls._local_pool(output, pooling_factor)

        return output

    @classmethod
    def _local_smoothed_convolve(cls, input: tf.Tensor, filter_width: int, channels_out: int,
                                 smoothing_window_sizes: List[int], pooling_factor: int) -> tf.Tensor:
        with tf.variable_scope('smoothing'):
            max_window_size = max(*smoothing_window_sizes)
            filter = np.zeros((1, max_window_size, 1, len(smoothing_window_sizes)), dtype=np.float32)
            for idx, window in enumerate(smoothing_window_sizes):
                filter[0, :window, 0, idx] = 1.0 / window
            filter = tf.constant(filter, dtype=tf.float32, name='filter')
            # TODO VALID discards a little at the end, SAME would make weird endings on both sides
            smoothed = tf.nn.conv2d(input, filter, strides=[1, 1, 1, 1], padding='VALID')

        with tf.variable_scope('smoothed_conv'):
            filter_shape = (1, filter_width, len(smoothing_window_sizes), channels_out // len(smoothing_window_sizes))
            filter = tf.get_variable('filter', shape=filter_shape, dtype=tf.float32, initializer=xavier_initializer())
            bias = tf.get_variable('bias', shape=(channels_out,), dtype=tf.float32,
                                   initializer=tf.constant_initializer(0.0))
            # We want to convolve independently per channel, therefore use depthwise conv2d
            output = tf.nn.depthwise_conv2d_native(smoothed, filter, strides=[1, 1, 1, 1], padding='SAME')
            output = tf.nn.bias_add(output, bias)
            output = tf.nn.relu(output, name='relu')
            output = cls._local_pool(output, pooling_factor)

        return output

    @classmethod
    def _local_downsampled_convolve(cls, input: tf.Tensor, filter_width: int, channels_out: int,
                                    downsample_strides: List[int], pooling_factor: int) -> tf.Tensor:
        with tf.variable_scope('downsampling'):
            downsampled_list = [(stride, input[:, :, ::stride, :]) for stride in downsample_strides]

        def conv(stride: int, downsampled_tensor: tf.Tensor) -> tf.Tensor:
            with tf.variable_scope('stride{}'.format(stride)):
                return cls._local_convolve(downsampled_tensor, filter_width, channels_out // len(downsample_strides),
                                           pooling_factor)

        with tf.variable_scope('downsampled_conv'):
            outputs = [conv(stride, downsampled_tensor) for stride, downsampled_tensor in downsampled_list]
            output = tf.concat(outputs, axis=-1)

        return output

    @classmethod
    def _local_pool(cls, local_convolved: tf.Tensor, pooling_factor: int) -> tf.Tensor:
        size = [1, 1, local_convolved.get_shape()[2] // pooling_factor, 1]
        return tf.nn.max_pool(local_convolved, ksize=size, strides=size, padding='VALID', name='maxpool')
