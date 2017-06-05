import math
import random
import uuid
from abc import abstractmethod, abstractproperty
from functools import lru_cache
from pathlib import Path
from typing import List, Dict, Set, Tuple

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

    def update_sample_length(self, sample_length: int):
        self.sample_length = sample_length
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
            self.summary_writer.add_graph(session.graph, global_step=step)
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


class McnnConfiguration:

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
        self.reset()

    def __getstate__(self):
        return {
            'children': self.children,
            'parents': self.parents,
            'uuid': self.uuid
        }

    def __setstate__(self, state):
        self.reset()
        self.children = state['children']
        self.parents = state['parents']
        self.uuid = state['uuid']

    def _update_uuid(self):
        self.uuid = str(uuid.uuid4())

    def add_parent(self, parent):
        self.parents.append(parent)
        parent.children.append(self)

    @staticmethod
    @lru_cache(maxsize=None)
    def _concat(tensors: Tuple[tf.Tensor]) -> tf.Tensor:
        if len(tensors) == 1:
            return tensors[0]

        time_lengths = [tensor.get_shape().as_list()[2] for tensor in tensors]
        min_time_length = min(time_lengths)

        with tf.variable_scope('concat_nodes'):
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
        self._concat.cache_clear()
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

    def _get_input_index(self, parent_node) -> int:
        node_idx = self.parents.index(parent_node)
        return sum(p.output_count for p in self.parents[:node_idx])

    @abstractproperty
    @property
    def output_count(self):
        raise NotImplementedError()

    @property
    def input_count(self):
        return sum(p.output_count for p in self.parents)


class InputNode(Node):

    def __init__(self):
        super().__init__([])
        self._tmp_input = None

    @property
    def output_count(self):
        return 1

    def __setstate__(self, state):
        super().__setstate__(state)
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


class VariableNode(Node):

    WEIGHT_COLLECTION = 'NODE_WEIGHT_COLLECTION'

    # Parameters from 2016 Miconi
    NEURONS_BELOW_DEL_THRESHOLD = 1
    DELETION_THRESHOLD = 0.5        # Originally 0.05
    L1_NORM_PENALTY_STRENGTH = 1e-4

    DELETE_NODE_THRESHOLD = 8
    OUTPUT_INCREMENT = 16

    def __init__(self, parents: List):
        super().__init__(parents)
        self._saved_filter = None
        self._saved_bias = None

    # noinspection PyAttributeOutsideInit
    def reset(self):
        super().reset()
        self._scope = None
        self._filter_var = None
        self._bias_var = None
        self._filter_query = None
        self._bias_query = None
        self._below_del_threshold_count = None
        self._below_del_threshold_indices = None

    def __getstate__(self):
        s = super().__getstate__()
        s.update({
            'saved_filter': self._saved_filter,
            'saved_bias': self._saved_bias
        })
        return s

    def __setstate__(self, state):
        super().__setstate__(state)
        self._saved_filter = state['saved_filter']
        self._saved_bias = state['saved_bias']
        self.reset()

    def mutate(self, session: tf.Session):
        count = self._below_del_threshold_count.eval(session=session)
        logging.info('{} neurons below del threshold'.format(count))
        if count < self.NEURONS_BELOW_DEL_THRESHOLD:
            logging.info('Growing')
            self.grow()
        elif count > self.NEURONS_BELOW_DEL_THRESHOLD:
            logging.info('Shrinking')
            deletion_indices = self._below_del_threshold_indices.eval(session=session)
            self.shrink(deletion_indices)

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
        logging.info('Restored variables for {}'.format(self.uuid))

    def grow(self):
        self._add_outputs(self.OUTPUT_INCREMENT)

    def shrink(self, deletion_indices: np.ndarray):
        active_outputs = self.output_count - deletion_indices.shape[0]
        if active_outputs < self.DELETE_NODE_THRESHOLD:
            self.delete()
        else:
            self._remove_outputs(deletion_indices)

    @abstractmethod
    def _add_outputs(self, add_outputs_count: int):
        raise NotImplementedError()

    @abstractmethod
    def _remove_outputs(self, indices: np.ndarray):
        raise NotImplementedError()

    @abstractmethod
    def _add_inputs(self, parent_node: Node, add_inputs_count: int):
        raise NotImplementedError()

    @abstractmethod
    def _remove_inputs(self, parent_node: Node, indices: np.ndarray):
        raise NotImplementedError()

    def _notify_output_removal(self, indices: np.ndarray):
        for child in self.children:
            assert isinstance(child, VariableNode)
            child._remove_inputs(self, indices)

    def _notify_output_addition(self, add_outputs_count: int):
        for child in self.children:
            assert isinstance(child, VariableNode)
            child._add_inputs(self, add_outputs_count)

    def delete(self):
        for parent in self.parents:
            assert isinstance(parent, Node)
            parent.children.remove(self)

        self._notify_output_removal(np.arange(self.output_count))
        for child in self.parents:
            assert isinstance(child, VariableNode)
            child.parents.remove(self)

    def variable_initialization(self, shape: List[int]):
        return tf.random_normal(shape, stddev=0.035)


class FullyConnectedNode(VariableNode):

    def __init__(self, parents: List, fixed_output_count: int = None, non_linearity: bool = True):
        super().__init__(parents)
        self.can_mutate = fixed_output_count is None
        self.non_linearity = non_linearity
        self._output_count = fixed_output_count or 16

    @property
    def output_count(self):
        return self._output_count

    def __setstate__(self, state):
        super().__setstate__(state)
        self.can_mutate = state['can_mutate']
        self._output_count = state['output_count']
        self.non_linearity = state['non_linearity']

    def __getstate__(self):
        s = super().__getstate__()
        s.update({
            'can_mutate': self.can_mutate,
            'output_count': self._output_count,
            'non_linearity': self.non_linearity
        })
        return s

    def mutate(self, session: tf.Session):
        if self.can_mutate:
            super().mutate(session)

    def _add_outputs(self, add_outputs_count: int):
        if self._scope is None:
            return
        with tf.variable_scope(self._scope):
            new_weights = self.variable_initialization([self.input_count, add_outputs_count])
            new_bias = self.variable_initialization([add_outputs_count])
            # Concat at output_count
            self._filter_query = tf.concat([self._filter_query, new_weights], axis=-1)
            self._bias_query = tf.concat([self._bias_query, new_bias], axis=0)
        self._output_count += add_outputs_count
        self._notify_output_addition(add_outputs_count)

    def _add_inputs(self, parent_node: Node, add_inputs_count: int):
        if self._scope is None:
            return
        with tf.variable_scope(self._scope):
            new_weights = self.variable_initialization([add_inputs_count, self._output_count])
            # TODO using output_count here relies on the fact that the maxpooling op reduces to time_length 1
            offset = self._get_input_index(parent_node) + parent_node.output_count - add_inputs_count
            new_tensors = [self._filter_query[:offset], new_weights, self._filter_query[offset:]]
            # Concat at input dimension
            self._filter_query = tf.concat(new_tensors, axis=0)

    def _remove_inputs(self, parent_node: Node, indices: np.ndarray):
        if self._scope is None:
            return
        with tf.variable_scope(self._scope):
            deletion_count = indices.shape[0]
            mask = np.ones([self.input_count + deletion_count], dtype=np.bool_)
            mask[indices] = 0
            self._filter_query = tf.boolean_mask(self._filter_query, mask)

    def _remove_outputs(self, indices: np.ndarray):
        if self._scope is None:
            return
        with tf.variable_scope(self._scope):
            mask = np.ones([self.output_count], dtype=np.bool_)
            mask[indices] = 0
            self._filter_query = tf.transpose(tf.boolean_mask(tf.transpose(self._filter_query), mask))
            self._bias_query = tf.boolean_mask(self._bias_query, mask)
        deletion_count = indices.shape[0]
        self._output_count -= deletion_count
        self._notify_output_removal(indices)

    def _build(self):
        super()._build()
        # noinspection PyTypeChecker
        input_tensor = self._concat(tuple(node.output_tensor for node in self.parents))
        with tf.variable_scope('fullyconnected_node_' + self.uuid) as scope:
            self._scope = scope

            # Pool and reshape if not compatible
            if input_tensor.get_shape().ndims > 2:
                # TODO maxpool instead?
                input_tensor = tf.reduce_max(input_tensor, axis=2, keep_dims=True)
                batch_size = tf.shape(input_tensor)[0]
                filter_count = input_tensor.get_shape().as_list()[-1]
                input_tensor = tf.reshape(input_tensor, shape=(batch_size, filter_count))

            input_size = input_tensor.get_shape()[-1]

            weight = tf.get_variable('weight', shape=(input_size, self._output_count), dtype=tf.float32,
                                     initializer=xavier_initializer())
            bias = tf.get_variable('bias', shape=(self._output_count,), dtype=tf.float32,
                                   initializer=tf.constant_initializer(0.0))
            output = tf.matmul(input_tensor, weight)
            output = tf.nn.bias_add(output, bias)
            if self.non_linearity:
                output = tf.nn.relu(output, name='activation')

            weight_sums = tf.reduce_sum(tf.abs(weight), axis=0)
            tf.summary.histogram('outgoing_weight_sums', weight_sums)
            below_del_threshold = weight_sums < self.DELETION_THRESHOLD
            self._below_del_threshold_indices = tf.where(below_del_threshold)
            self._below_del_threshold_count = tf.reduce_sum(tf.cast(below_del_threshold, tf.int16))
            tf.summary.scalar('below_del_threshold_count', self._below_del_threshold_count)
            tf.summary.scalar('output_count', self.output_count)

        self._filter_var = weight
        self._bias_var = bias
        self._filter_query = weight
        self._bias_query = bias
        self.output_tensor = output
        return output


class ConvNode(VariableNode):

    FILTER_WIDTH = 16
    NEW_NODE_PROBABILITY = 0.1

    def __init__(self, parents: List):
        super().__init__(parents)
        self.stride = random.randint(1, 2)
        self.channels_out = self.OUTPUT_INCREMENT

    @property
    def output_count(self):
        return self.channels_out

    def __getstate__(self):
        s = super().__getstate__()
        s.update({
            'stride': self.stride,
            'channels_out': self.channels_out,
        })
        return s

    def __setstate__(self, state):
        super().__setstate__(state)
        self.stride = state['stride']
        self.channels_out = state['channels_out']

    def grow(self):
        super().grow()
        create_new_node = random.random() < self.NEW_NODE_PROBABILITY
        if create_new_node:
            self._create_new_node()

    def _create_new_node(self):
        new_conv = ConvNode(self.parents)
        self.add_parent(new_conv)
        new_conv._notify_output_addition(new_conv.output_count)

    def _add_outputs(self, add_channels: int):
        if self._scope is None:
            return
        with tf.variable_scope(self._scope):
            new_filters = self.variable_initialization([1, self.FILTER_WIDTH, self.input_count, add_channels])
            new_bias = self.variable_initialization([add_channels])
            # Concat at channels_out
            self._filter_query = tf.concat([self._filter_query, new_filters], axis=-1)
            self._bias_query = tf.concat([self._bias_query, new_bias], axis=0)
        self.channels_out += add_channels
        self._notify_output_addition(add_channels)

    def _add_inputs(self, parent_node: Node, add_inputs_count: int):
        if self._scope is None:
            return
        with tf.variable_scope(self._scope):
            new_filters = self.variable_initialization([1, self.FILTER_WIDTH, add_inputs_count, self.channels_out])
            offset = self._get_input_index(parent_node) + parent_node.output_count - add_inputs_count
            new_tensors = [self._filter_query[:, :, :offset],
                           new_filters,
                           self._filter_query[:, :, offset:]]
            # Concat at input dimension
            self._filter_query = tf.concat(new_tensors, axis=2)

    def _remove_inputs(self, parent_node: Node, indices: np.ndarray):
        if self._scope is None:
            return
        with tf.variable_scope(self._scope):
            deletion_count = indices.shape[0]
            mask = np.ones([self.input_count + deletion_count], dtype=np.bool_)
            mask[indices] = 0
            transposed = tf.transpose(self._filter_query, perm=[2, 3, 0, 1])
            filtered = tf.boolean_mask(transposed, mask)
            self._filter_query = tf.transpose(filtered, perm=[2, 3, 0, 1])

    def _remove_outputs(self, indices: np.ndarray):
        if self._scope is None:
            return
        with tf.variable_scope(self._scope):
            mask = np.ones([self.output_count], dtype=np.bool_)
            mask[indices] = 0
            self._filter_query = tf.transpose(tf.boolean_mask(tf.transpose(self._filter_query), mask))
            self._bias_query = tf.boolean_mask(self._bias_query, mask)
        deletion_count = indices.shape[0]
        self.channels_out -= deletion_count
        self._notify_output_removal(indices)

    def _build(self):
        super()._build()
        # noinspection PyTypeChecker
        input_tensor = self._concat(tuple(node.output_tensor for node in self.parents))
        with tf.variable_scope('conv_node_' + self.uuid) as scope:
            self._scope = scope

            channels_in = input_tensor.get_shape()[-1]

            filter = tf.get_variable('filter', shape=(1, self.FILTER_WIDTH, channels_in, self.channels_out),
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
            self._below_del_threshold_indices = tf.where(below_del_threshold)
            self._below_del_threshold_count = tf.reduce_sum(tf.cast(below_del_threshold, tf.int16))
            tf.summary.scalar('below_del_threshold_count', self._below_del_threshold_count)
            tf.summary.scalar('output_count', self.output_count)

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
            fully_connected_node = FullyConnectedNode([first_conv_node])
            self.terminus_node = FullyConnectedNode([fully_connected_node], fixed_output_count=num_classes,
                                                    non_linearity=False)

        super().__init__(sample_length, learning_rate, num_classes, batch_size)

    def mutate(self, session: tf.Session):
        conv_nodes = [node for node in self.input_node.all_descendants() if isinstance(node, VariableNode)]
        with tf.variable_scope(self._nodes_scope):
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
        weights = tf.get_collection(VariableNode.WEIGHT_COLLECTION)
        l1 = tf.add_n([tf.reduce_sum(tf.abs(weight)) for weight in weights])
        l1_penalty = VariableNode.L1_NORM_PENALTY_STRENGTH * l1
        tf.summary.scalar('l1_penalty', l1_penalty)
        # noinspection PyTypeChecker
        return loss + l1_penalty

    def restore(self, session: tf.Session, checkpoint_dir: Path):
        # Init all values first, because not all are saved
        session.run(self.init)
        super().restore(session, checkpoint_dir)
        with tf.variable_scope(self._nodes_scope):
            for node in self.input_node.all_descendants():
                node.restore_variables(session)

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
        with tf.variable_scope('nodes') as scope:
            self._nodes_scope = scope
            self.input_node.build_dag(input_2d)
            output = self.terminus_node.output_tensor

        nodes = self.input_node.all_descendants()
        tf.summary.scalar('node_count', len(nodes))
        tf.summary.scalar('max_depth', self.terminus_node.max_depth)
        tf.summary.scalar('total_output_count', sum(n.output_count for n in nodes))

        logits = tf.identity(output, name='logits')

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
