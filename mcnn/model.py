import math
import random
import uuid
from abc import abstractmethod, abstractproperty
from pathlib import Path
from typing import List, Dict, Set

import numpy as np
import tensorflow as tf
import pickle
import logging
from tensorflow.contrib.layers import xavier_initializer


class Model:

    def __init__(self, sample_length: int, learning_rate: float, num_classes: int, batch_size: int):
        self.batch_size = batch_size
        self.sample_length = sample_length
        self.num_classes = num_classes
        self.learning_rate = learning_rate
        self.build()

    # noinspection PyAttributeOutsideInit
    def build(self):
        tf.reset_default_graph()

        self.global_step = tf.Variable(0, name='global_step', trainable=False)

        self.input = tf.placeholder(tf.float32, shape=(None, self.sample_length), name='input')
        self.dynamic_batch_size = tf.shape(self.input)[0]
        self.labels = tf.placeholder(tf.int64, shape=(None,), name='labels')

        input_2d = tf.reshape(self.input, shape=(self.dynamic_batch_size, 1, self.sample_length, 1))

        self.logits = self._create_network(input_2d)

        with tf.variable_scope('training'):
            cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.labels, logits=self.logits,
                                                                           name='crossentropy')
            self.loss = self._define_loss(cross_entropy)
            tf.summary.scalar('loss', self.loss)
            optimizer = tf.train.RMSPropOptimizer(self.learning_rate)
            self.train_op = optimizer.minimize(self.loss, global_step=self.global_step, name='train_op')

        with tf.variable_scope('evaluation'):
            correct = tf.nn.in_top_k(self.logits, self.labels, 1, name='correct')
            self.correct_count = tf.reduce_sum(tf.cast(correct, tf.int32), name='correct_count')
            self.accuracy = self.correct_count / self.dynamic_batch_size
            tf.summary.scalar('accuracy', self.accuracy)

        self.init = tf.global_variables_initializer()
        self.summary = tf.summary.merge_all()
        self.summary_writer = None
        self._create_saver()

    def _define_loss(self, cross_entropy: tf.Tensor) -> tf.Tensor:
        return tf.reduce_mean(cross_entropy, name='crossentropy_mean')

    def _create_saver(self):
        self.saver = tf.train.Saver()

    @abstractmethod
    def _create_network(self, input_2d: tf.Tensor) -> tf.Tensor:
        raise NotImplementedError()

    def create(self, session: tf.Session):
        logging.info('Created model with fresh parameters.')
        session.run(self.init)

    def setup_summary_writer(self, session: tf.Session, log_dir: Path):
        self.summary_writer = tf.summary.FileWriter(str(log_dir))

    def restore(self, session: tf.Session, checkpoint_dir: Path):
        ckpt = tf.train.get_checkpoint_state(str(checkpoint_dir))
        if ckpt and ckpt.model_checkpoint_path:
            logging.info('Reading model parameters from {}'.format(ckpt.model_checkpoint_path))
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


class Node:

    def __init__(self, parents: List):
        self.children = []
        self.parents = list(parents)
        for node in parents:
            node.children.append(self)
        self.output_tensor = None
        self.max_depth = None
        self._update_uuid()

    def __getstate__(self):
        return {
            'children': self.children,
            'parents': self.parents,
            'uuid': self.uuid
        }

    def __setstate__(self, state):
        self.output_tensor = None
        self.children = state['children']
        self.parents = state['parents']
        self.uuid = state['uuid']

    def _update_uuid(self):
        self.uuid = str(uuid.uuid4())

    def add_parent(self, parent):
        self.parents.append(parent)
        parent.children.append(self)

    def _concat(self, tensors: List[tf.Tensor]) -> tf.Tensor:
        if len(tensors) == 0:
            return tensors[0]
        time_lengths = [tensor.get_shape().as_list()[2] for tensor in tensors]
        min_time_length = min(time_lengths)

        resized_tensors = []
        for tensor, time_length in zip(tensors, time_lengths):
            # TODO pad instead of max pool if stride = 2
            if time_length > min_time_length:
                ksize_time = int(round(time_length / min_time_length, ndigits=0))
                kernel_size = [1, 1, ksize_time, 1]
                tensor = tf.nn.max_pool(tensor, ksize=kernel_size, strides=kernel_size, padding='SAME')
                # in case of bad roundings
                new_time_length = tensor.get_shape().as_list()[2]
                if new_time_length > min_time_length:
                    tensor = tensor[:, :, :min_time_length, ...]
            resized_tensors.append(tensor)

        return tf.concat(resized_tensors, axis=-1)

    def save_variables(self, session: tf.Session):
        pass

    def restore_variables(self, session: tf.Session):
        pass

    def reset(self):
        self.output_tensor = None

    def all_descendants(self, visited: Set = None) -> Set:
        if visited is None:
            visited = set()
        visited.add(self)
        for child in self.children:
            if child not in visited:
                child.all_descendants(visited)
        return visited

    def is_buildable(self) -> bool:
        return all(parent.output_tensor is not None for parent in self.parents)

    def is_built(self) -> bool:
        return self.output_tensor is not None

    def reset_recursively(self):
        for node in self.all_descendants():
            node.reset()

    def build_recursively(self):
        if not self.is_buildable() or self.is_built():
            return
        self._build()
        for child in self.children:
            child.build_recursively()

    def _build(self):
        if len(self.parents) == 0:
            self.max_depth = 0
        else:
            self.max_depth = max(p.max_depth for p in self.parents) + 1


class InputNode(Node):

    def __init__(self):
        super().__init__([])
        # TODO change this if multivariate
        self.channels_out = 1
        self._tmp_input = None

    def __setstate__(self, state):
        super().__setstate__(state)
        self.channels_out = 1
        self._tmp_input = None

    def build_dag(self, input_tensor: tf.Tensor):
        self._tmp_input = input_tensor
        self.reset_recursively()
        self.build_recursively()

    def _build(self):
        super()._build()
        with tf.variable_scope('input_node'):
            self.output_tensor = tf.identity(self._tmp_input, name='reshaped_input')
        self._tmp_input = None


class TerminusNode(Node):

    def __init__(self, parents: List):
        super().__init__(parents)

    def add_parent(self, parent):
        # Disallow adding parents to this node to be compatible with fully connected layer
        pass

    def _build(self):
        super()._build()
        with tf.variable_scope('terminus_node'):
            self.output_tensor = self._concat([node.output_tensor for node in self.parents])


class ConvNode(Node):

    WEIGHT_COLLECTION = 'CONV_WEIGHT_COLLECTION'
    # Parameters from 2016 Miconi
    NEURONS_BELOW_DEL_THRESHOLD = 1
    DELETION_THRESHOLD = 0.5        # Originally 0.05
    L1_NORM_PENALTY_STRENGTH = 1e-4

    def __init__(self, parents: List):
        super().__init__(parents)
        self.stride = random.randint(1, 2)
        # TODO use 1 here?
        self.channels_out = 16
        # TODO how to set this?
        self.filter_width = 16

        self._saved_filter = None
        self._saved_bias = None
        self.reset()

    # noinspection PyAttributeOutsideInit
    def reset(self):
        super().reset()
        self._scope = None
        self._filter_var = None
        self._bias_var = None
        self._filter_query = None
        self._bias_query = None

    def __getstate__(self):
        s = super().__getstate__()
        s.update({
            'stride': self.stride,
            'saved_filter': self._saved_filter,
            'saved_bias': self._saved_bias
        })
        return s

    def __setstate__(self, state):
        super().__setstate__(state)
        self.stride = state['stride']
        self._saved_filter = state['saved_filter']
        self._saved_bias = state['saved_bias']
        self.channels_out = 16
        self.filter_width = 16
        self.reset()

    def grow(self):
        if self.channels_out < 256 and not any(isinstance(child, TerminusNode) for child in self.children):
            self._add_out_channels(16)
        else:
            self._add_parallel_node()

    def _add_parallel_node(self):
        new_conv = ConvNode(self.parents)
        self.add_parent(new_conv)
        for child in self.children:
            child.add_parent(new_conv)

    def _add_out_channels(self, add_channels: int):
        if self._scope is None:
            return
        with tf.variable_scope(self._scope):
            channels_in = self._filter_query.get_shape().as_list()[2]
            new_filters = tf.random_normal([1, self.filter_width, channels_in, add_channels], stddev=0.35)
            # Concat at channels_out
            self._filter_query = tf.concat([self._filter_query, new_filters], axis=-1)
            self.channels_out += add_channels
        pass

    def mutate(self, session: tf.Session):
        count = self._below_del_threshold_count.eval(session=session)
        logging.info('{} neurons below del threshold'.format(count))
        if count < self.NEURONS_BELOW_DEL_THRESHOLD:
            logging.info('Growing')
            self.grow()
        elif count > self.NEURONS_BELOW_DEL_THRESHOLD:
            logging.info('Would shrink if implemented')
            # TODO shrink
            pass

    def _change_var_queries(self, new_parent):
        if self._scope is None:
            return
        with tf.variable_scope(self._scope):
            channels_in = new_parent.channels_out
            new_filters = tf.random_normal([1, self.filter_width, channels_in, self.channels_out], stddev=0.35)
            # Concat at channels_in
            self._filter_query = tf.concat([self._filter_query, new_filters], axis=-2)

    def save_variables(self, session: tf.Session):
        if self._scope is None:
            return
        self._saved_filter = self._filter_query.eval(session=session)
        self._saved_bias = self._bias_query.eval(session=session)

    def restore_variables(self, session: tf.Session):
        if self._scope is None or self._saved_bias is None or self._saved_filter is None:
            return
        with tf.variable_scope(self._scope):
            self._filter_var.assign(self._saved_filter).op.run(session=session)
            self._bias_var.assign(self._saved_bias).op.run(session=session)

    def add_parent(self, parent):
        super().add_parent(parent)
        self._change_var_queries(parent)

    def _build(self):
        super()._build()
        with tf.variable_scope('conv_node_' + self.uuid) as scope:
            self._scope = scope
            input_tensor = self._concat([node.output_tensor for node in self.parents])

            channels_in = input_tensor.get_shape()[-1]

            filter = tf.get_variable('filter', shape=(1, self.filter_width, channels_in, self.channels_out),
                                     dtype=tf.float32, initializer=xavier_initializer())
            tf.add_to_collection(self.WEIGHT_COLLECTION, filter)
            bias = tf.get_variable('bias', shape=(self.channels_out,), dtype=tf.float32,
                                   initializer=tf.constant_initializer(0.0))
            output = tf.nn.conv2d(input_tensor, filter, strides=[1, 1, self.stride, 1], padding='SAME')
            output = tf.nn.bias_add(output, bias)
            output = tf.nn.relu(output, name='relu')

            weight_sums = tf.reduce_sum(tf.abs(filter), axis=[0, 1, 2])
            tf.summary.histogram('outgoing_weight_sums', weight_sums)
            below_del_threshold = weight_sums < self.DELETION_THRESHOLD
            self._below_del_threshold_count = tf.reduce_sum(tf.cast(below_del_threshold, tf.int16))
            tf.summary.scalar('below_del_threshold_count', self._below_del_threshold_count)

        self._filter_var = filter
        self._bias_var = bias
        self._filter_query = filter
        self._bias_query = bias
        self.output_tensor = output
        return output


class MutatingCnnModel(Model):

    def __init__(self, sample_length: int, learning_rate: float, num_classes: int, batch_size: int,
                 checkpoint_dir: Path):
        nodes_file = (checkpoint_dir / 'nodes.pickle')
        if nodes_file.exists():
            with nodes_file.open('rb') as infile:
                self.input_node, self.terminus_node = pickle.load(infile)
        else:
            self.input_node = InputNode()
            first_conv_node = ConvNode([self.input_node])
            self.terminus_node = TerminusNode([first_conv_node])
        super().__init__(sample_length, learning_rate, num_classes, batch_size)

    def mutate_random_uniform(self):
        nodes = tuple(node for node in self.input_node.all_descendants() if isinstance(node, ConvNode))
        sample = random.choice(nodes)
        sample.grow()

    def mutate(self, session: tf.Session):
        conv_nodes = [node for node in self.input_node.all_descendants() if isinstance(node, ConvNode)]
        for node in conv_nodes:
            assert node.is_built()
            node.mutate(session)

    def build(self):
        if self.input_node is not None:
            self.input_node.reset_recursively()
        super().build()

    def _define_loss(self, cross_entropy: tf.Tensor) -> tf.Tensor:
        loss = super()._define_loss(cross_entropy)
        tf.summary.scalar('cross_entropy', loss)
        # Calculate L1-norm on outgoing weights
        # TODO maybe try different regularization as defined by Sentiono 1997
        # TODO Idea: take the maximum over axis [0, 1, 2]
        weights = tf.get_collection(ConvNode.WEIGHT_COLLECTION)
        l1 = tf.add_n([tf.reduce_sum(tf.abs(weight)) for weight in weights])
        l1_penalty = ConvNode.L1_NORM_PENALTY_STRENGTH * l1
        tf.summary.scalar('l1_penalty', l1_penalty)
        # noinspection PyTypeChecker
        return loss + l1_penalty

    def restore(self, session: tf.Session, checkpoint_dir: Path):
        # Init all values first, because not all are saved
        session.run(self.init)
        super().restore(session, checkpoint_dir)
        if self.input_node is not None:
            self.input_node.restore_variables(session)

    def save(self, session: tf.Session, checkpoint_dir: Path):
        super().save(session, checkpoint_dir)
        for node in self.input_node.all_descendants():
            node.save_variables(session)
        with (checkpoint_dir / 'nodes.pickle').open('wb') as outfile:
            pickle.dump((self.input_node, self.terminus_node), outfile)

    def _create_saver(self):
        # Exclude variables of nodes, those are saved separately
        vars = [var for var in tf.global_variables() if 'nodes/' not in var.name]
        self.saver = tf.train.Saver(var_list=vars)

    def _create_network(self, input_2d: tf.Tensor) -> tf.Tensor:
        with tf.variable_scope('nodes'):
            self.input_node.build_dag(input_2d)
            output = self.terminus_node.output_tensor

        tf.summary.scalar('node_count', len(self.input_node.all_descendants()))
        tf.summary.scalar('max_depth', self.terminus_node.max_depth - 1)

        # TODO maxpool instead?
        output = tf.reduce_max(output, axis=2, keep_dims=True)
        num_filters = output.get_shape()[-1]

        output = tf.reshape(output, shape=(self.dynamic_batch_size, -1))
        output = self._full_fullyconnected(output, num_filters, 256, activation=tf.nn.relu)
        logits = self._full_fullyconnected(output, 256, self.num_classes)
        logits = tf.identity(logits, name='logits')

        return logits


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
