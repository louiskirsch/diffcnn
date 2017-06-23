import math
import random
import tempfile
import uuid
from abc import abstractmethod, abstractproperty
from collections import deque
from functools import lru_cache
from pathlib import Path
from typing import List, Dict, Set, Tuple, Union

import itertools
import numpy as np
import scipy.misc
import tensorflow as tf
import tensorflow.contrib.distributions as tfd
import pickle
import logging
import graphviz
from tensorflow.contrib.layers import xavier_initializer


class Model:

    POST_TRAINING_UPDATE_COLLECTION = 'POST_TRAINING_UPDATE_COLLECTION'

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
            self._define_training()

        with tf.variable_scope('evaluation'):
            self._define_evaluation()

        self.init = tf.global_variables_initializer()
        self.summary = tf.summary.merge_all()
        self.summary_writer = None

    def _with_post_training_update(self, op: tf.Operation) -> tf.Operation:
        with tf.control_dependencies([op]):
            return tf.group(*tf.get_collection(self.POST_TRAINING_UPDATE_COLLECTION))

    def _define_training(self):
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.labels, logits=self.logits,
                                                                       name='crossentropy')
        self.loss = self._define_loss(cross_entropy)
        tf.summary.scalar('loss', self.loss)
        self.optimizer = tf.train.AdamOptimizer(self.learning_rate)
        train_op = self.optimizer.minimize(self.loss, global_step=self.global_step, name='train_op')
        self.train_op = self._with_post_training_update(train_op)

    def _define_evaluation(self):
        correct = tf.nn.in_top_k(self.logits, self.labels, 1, name='correct')
        self.correct_count = tf.reduce_sum(tf.cast(correct, tf.int32), name='correct_count')
        self.accuracy = self.correct_count / self.dynamic_batch_size
        self.testing_error = 1 - self.accuracy
        tf.summary.scalar('accuracy', self.accuracy)
        tf.summary.scalar('testing_error', self.testing_error)

    def _define_loss(self, cross_entropy: tf.Tensor) -> tf.Tensor:
        return tf.reduce_mean(cross_entropy, name='crossentropy_mean')

    def _vars_to_restore(self) -> Union[None, List[tf.Variable]]:
        # By default, restore all
        return None

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
            saver = tf.train.Saver(var_list=self._vars_to_restore())
            saver.restore(session, ckpt.model_checkpoint_path)
        else:
            raise FileNotFoundError('No checkpoint for evaluation found')

    def restore_or_create(self, session: tf.Session, checkpoint_dir: Path):
        try:
            self.restore(session, checkpoint_dir)
        except FileNotFoundError:
            self.create(session)

    def save(self, session: tf.Session, checkpoint_dir: Path):
        saver = tf.train.Saver()
        saver.save(session, str(checkpoint_dir / 'mcnn'), global_step=self.global_step.eval(session))

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
            return self._extract_summary(results, session)

        return results

    def _extract_summary(self, results: List, session: tf.Session) -> List:
        summary_result = results[-1]
        step = results[0]
        self.summary_writer.add_summary(summary_result, global_step=step)
        self.summary_writer.add_graph(session.graph, global_step=step)
        return results[:-1]

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


class NodeBuildConfiguration:

    def __init__(self):
        self.is_training = None
        self.depth_penalty = 'none'
        self.penalty_type = 'weigend'
        self.const_neuron_deletion_threshold = 0.0

    @classmethod
    def from_options(cls, options):
        config = NodeBuildConfiguration()
        config.depth_penalty = options.depth_penalty
        config.penalty_type = options.penalty_fnc
        config.const_neuron_deletion_threshold = options.neuron_deletion_threshold


class Node:

    def __init__(self, parents: List):
        self.children = []
        self.parents = list(parents)
        for node in parents:
            node.children.append(self)
        self.output_tensor = None
        self.max_depth = None
        self.vars_saved = False
        self._update_uuid()
        self.reset()

    def __getstate__(self):
        return {
            'children': self.children,
            'parents': self.parents,
            'uuid': self.uuid,
            'vars_saved': self.vars_saved
        }

    def __setstate__(self, state):
        self.reset()
        self.children = state['children']
        self.parents = state['parents']
        self.uuid = state['uuid']
        self.vars_saved = state['vars_saved']

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

    def reset(self):
        self.output_tensor = None

    def all_descendants(self, visited: Set = None) -> Set['Node']:
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

    def build_descendants(self, configuration: NodeBuildConfiguration):
        if not self.is_buildable() or self.is_built():
            return
        self._build(configuration)
        for child in self.children:
            assert isinstance(child, Node)
            child.build_descendants(configuration)

    def _build(self, configuration: NodeBuildConfiguration):
        if len(self.parents) == 0:
            self.max_depth = 0
        else:
            self.max_depth = max(p.max_depth for p in self.parents) + 1

    def _post_build(self, configuration: NodeBuildConfiguration, post_build_tensors: Dict[str, tf.Tensor]):
        pass

    def _get_input_index(self, parent_node) -> int:
        node_idx = self.parents.index(parent_node)
        return sum(p.output_count for p in self.parents[:node_idx])

    def to_graphviz(self, session: tf.Session, step: int, tmp_directory: Path) -> graphviz.Digraph:
        graph = graphviz.Digraph('mutating-cnn')
        graph.attr(rankdir='LR')
        nodes = self.all_descendants()
        for node in nodes:
            label = node.label
            for k, v in node.str_node_properties(session):
                label += '\n{} = {}'.format(k, v)
            graph.node(node.uuid, label)
            node._add_graph_misc(session, graph, step, tmp_directory)
        for node in nodes:
            for child in node.children:
                graph.edge(node.uuid, child.uuid)
        return graph

    def _add_graph_misc(self, session: tf.Session, graph: graphviz.Digraph, step: int, tmp_directory: Path):
        pass

    @property
    def label(self) -> str:
        return 'node'

    def str_node_properties(self, session: tf.Session) -> List[Tuple[str, str]]:
        return [
            ('outputs', str(self.output_count)),
            ('depth', str(self.max_depth))
        ]

    @property
    def penalty(self) -> Union[tf.Tensor, None]:
        return None

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

    @property
    def label(self) -> str:
        return 'input_node'

    def __setstate__(self, state):
        super().__setstate__(state)
        self._tmp_input = None

    def reset_all(self):
        self._concat.cache_clear()
        for node in self.all_descendants():
            node.reset()

    def post_build_all(self, configuration: NodeBuildConfiguration):
        post_build_tensors = {}
        for node in self.all_descendants():
            node._post_build(configuration, post_build_tensors)

    def build_dag(self, input_tensor: tf.Tensor, configuration: NodeBuildConfiguration):
        self._tmp_input = input_tensor
        self.reset_all()
        self.build_descendants(configuration)
        self.post_build_all(configuration)

    def _build(self, configuration):
        super()._build(configuration)
        with tf.variable_scope('input_node'):
            self.output_tensor = tf.identity(self._tmp_input, name='reshaped_input')
        self._tmp_input = None


class VariableNode(Node):

    SCALE_COLLECTION = 'SCALE_COLLECTION'

    # Parameters from 2016 Miconi
    NEURONS_BELOW_DEL_THRESHOLD = 1
    # TODO how to get rid of this hyperparameter?
    L1_NORM_PENALTY_STRENGTH = 1e-2

    DELETE_NODE_THRESHOLD = 1
    OUTPUT_INCREMENT = 16

    def __init__(self, parents: List, non_linearity: bool = True):
        super().__init__(parents)
        self.can_mutate = True
        self.non_linearity = non_linearity

    # noinspection PyAttributeOutsideInit
    def reset(self):
        super().reset()
        self._scope = None
        self._filter_var = None
        self._bias_var = None
        self._scale_var = None
        self._ema_mean_var = None
        self._ema_variance_var = None
        self._below_del_threshold_count = None
        self._below_del_threshold_indices = None
        self._penalty_per_output = None
        self._penalty = None

    def __getstate__(self):
        s = super().__getstate__()
        s.update({
            'can_mutate': self.can_mutate,
            'non_linearity': self.non_linearity,
            'output_count': self._output_count
        })
        return s

    def __setstate__(self, state):
        super().__setstate__(state)
        self._output_count = state['output_count']
        self.can_mutate = state['can_mutate']
        self.non_linearity = state['non_linearity']
        self.reset()

    def mutate(self, session: tf.Session, optimizer: tf.train.Optimizer, allow_node_creation: bool):
        if not self.can_mutate:
            return
        count = self._below_del_threshold_count.eval(session=session)
        logging.info('{} neurons below del threshold'.format(count))
        if count < self.NEURONS_BELOW_DEL_THRESHOLD:
            logging.info('Growing')
            self.grow(session, optimizer, allow_node_creation)
        elif count > self.NEURONS_BELOW_DEL_THRESHOLD:
            logging.info('Shrinking')
            deletion_indices = self._below_del_threshold_indices.eval(session=session)
            self.shrink(session, optimizer, deletion_indices[self.NEURONS_BELOW_DEL_THRESHOLD:])

    def grow(self, session: tf.Session, optimizer: tf.train.Optimizer, allow_node_creation: bool):
        self._add_outputs(session, optimizer, self.OUTPUT_INCREMENT)

    def shrink(self, session: tf.Session, optimizer: tf.train.Optimizer, deletion_indices: np.ndarray):
        active_outputs = self.output_count - deletion_indices.shape[0] - self.NEURONS_BELOW_DEL_THRESHOLD
        if active_outputs < self.DELETE_NODE_THRESHOLD:
            self.delete(session, optimizer)
        else:
            self._remove_outputs(session, optimizer, deletion_indices)

    def _notify_output_removal(self, session: tf.Session, optimizer: tf.train.Optimizer, indices: np.ndarray):
        for child in self.children:
            assert isinstance(child, VariableNode)
            child._remove_inputs(session, optimizer, indices)

    def _notify_output_addition(self, session: tf.Session, optimizer: tf.train.Optimizer, add_outputs_count: int):
        for child in self.children:
            assert isinstance(child, VariableNode)
            child._add_inputs(session, optimizer, self, add_outputs_count)

    def delete(self, session: tf.Session, optimizer: tf.train.Optimizer):
        for parent in self.parents:
            assert isinstance(parent, Node)
            parent.children.remove(self)
        self.parents.clear()
        self._remove_outputs(session, optimizer, np.arange(self.output_count))
        for child in self.children:
            assert isinstance(child, VariableNode)
            if len(child.parents) == 1:
                child.delete(session, optimizer)
            else:
                child.parents.remove(self)
        self.children.clear()

        logging.info('Deleted node {}'.format(self.uuid))

    def weight_initialization(self, shape: List[int]):
        # TODO does it make sense to use xavier here? shape doesn't have the correct fan_in and fan_out
        weights = xavier_initializer()(shape, dtype=tf.float32)
        return weights

    def bias_initialization(self, shape: List[int]):
        return tf.zeros(shape)

    @property
    def penalty(self) -> Union[tf.Tensor, None]:
        return self._penalty

    def get_usage(self, session: tf.Session) -> float:
        unused = self._below_del_threshold_count.eval(session=session)
        return 1 - (unused / self.output_count)

    def render_penalties(self, session: tf.Session, output_dir: Path, step: int, minimum_width: int = 200) -> Path:
        if self._scope is None:
            raise AssertionError('Node not created in graph')
        penalties = self._penalty_per_output.eval(session=session)
        img_width = len(penalties)
        img_height = math.ceil(img_width / 3)

        img = np.broadcast_to(penalties, (img_height, img_width))
        img = np.broadcast_to(img, (3, img_height, img_width))
        img = np.transpose(img, axes=(1, 2, 0)) * 255
        img = img.astype(np.uint8)

        if img_width < minimum_width:
            scale = minimum_width / img_width
            img = scipy.misc.imresize(img, scale, interp='nearest')

        filename = 'penalties_{}_{}.png'.format(self.uuid, step)
        file = output_dir / filename
        scipy.misc.imsave(str(file), img)
        return file

    def _add_graph_misc(self, session: tf.Session, graph: graphviz.Digraph, step: int, tmp_directory: Path):
        if self.penalty is None:
            return
        try:
            img_path = self.render_penalties(session, tmp_directory, step)
            node_name = self.uuid + '_penalties'
            graph.node(node_name, label='', image=str(img_path), imagescale='false', shape='box')
            graph.edge(self.uuid, node_name, arrowhead='none')
        except AssertionError:
            pass

    def str_node_properties(self, session: tf.Session) -> List[Tuple[str, str]]:
        properties = super().str_node_properties(session)
        if self._scope is not None:
            # TODO don't access via name
            penalty_sum = session.graph.get_tensor_by_name('training/penalty_sum:0')
            if self.penalty is not None:
                properties.append(('penalty-contrib', '{:.2f}%'.format(session.run(self.penalty / penalty_sum) * 100)))
            properties.extend([
                ('below_thres_count', self._below_del_threshold_count.eval(session=session))
            ])
        return properties

    @property
    def output_count(self):
        return self._output_count

    def _concat_outputs_to_var(self, var: tf.Variable, concat: tf.Tensor) -> tf.Operation:
        new_value = tf.concat([var, concat], axis=-1)
        self._override_shape(var, new_value.shape)
        return tf.assign(var, new_value, validate_shape=False).op

    def _mask_outputs(self, var: tf.Variable, mask: np.ndarray, transpose: bool = False) -> tf.Operation:
        if transpose:
            var_to_mask = tf.transpose(var)
        else:
            var_to_mask = var
        masked = tf.boolean_mask(var_to_mask, mask)
        if transpose:
            masked = tf.transpose(masked)
        self._override_shape(var, masked.shape)
        return tf.assign(var, masked, validate_shape=False).op

    def _add_outputs(self, session: tf.Session, optimizer: tf.train.Optimizer, add_outputs_count: int):
        if self._scope is None:
            return
        with tf.variable_scope(self._scope):
            weight_shape = list(tf.shape(self._filter_var)[:-1].eval(session=session))
            new_weights = self.weight_initialization(weight_shape + [add_outputs_count])
            new_bias = self.bias_initialization([add_outputs_count])
            new_scale = tf.ones([add_outputs_count])
            new_ema_mean = tf.zeros([add_outputs_count])
            new_ema_variance = tf.zeros([add_outputs_count])
            pairs = [
                (self._filter_var, new_weights),
                (self._bias_var, new_bias),
                (self._scale_var, new_scale),
                (self._ema_mean_var, new_ema_mean),
                (self._ema_variance_var, new_ema_variance)
            ]

            slots = optimizer.get_slot_names()
            for (var, concat), slot in itertools.product(list(pairs), slots):
                slot_var = optimizer.get_slot(var, slot)
                if slot_var is not None:
                    pairs.append((slot_var, tf.zeros_like(concat)))

            # Concat at output_count
            op = tf.group(*[self._concat_outputs_to_var(var, concat) for var, concat in pairs])
            op.run(session=session)
        self._output_count += add_outputs_count
        self._notify_output_addition(session, optimizer, add_outputs_count)

    def _remove_outputs(self, session: tf.Session, optimizer: tf.train.Optimizer, indices: np.ndarray):
        if self._scope is None:
            return
        with tf.variable_scope(self._scope):
            mask = np.ones([self.output_count], dtype=np.bool_)
            mask[indices] = 0

            vars = [self._filter_var, self._bias_var, self._scale_var, self._ema_mean_var, self._ema_variance_var]
            slots = optimizer.get_slot_names()
            for var, slot in itertools.product(list(vars), slots):
                slot_var = optimizer.get_slot(var, slot)
                if slot_var is not None:
                    vars.append(slot_var)

            op = tf.group(*[self._mask_outputs(var, mask, var.get_shape().ndims > 1) for var in vars])
            op.run(session=session)
        deletion_count = indices.shape[0]
        self._output_count -= deletion_count
        self._notify_output_removal(session, optimizer, indices)

    def _add_inputs(self, session: tf.Session, optimizer: tf.train.Optimizer, parent_node: Node, add_inputs_count: int):
        if self._scope is None:
            return
        with tf.variable_scope(self._scope):
            filter_shape = list(tf.shape(self._filter_var).eval(session=session))
            new_filters = self.weight_initialization(filter_shape[:-2] + [add_inputs_count] + [filter_shape[-1]])
            offset = self._get_input_index(parent_node) + parent_node.output_count - add_inputs_count

            prefix_slices = [slice(None) for _ in filter_shape[:-2]]
            from_slices = prefix_slices + [slice(None, offset)]
            to_slices = prefix_slices + [slice(offset, None)]
            new_tensors = [self._filter_var[from_slices],
                           new_filters,
                           self._filter_var[to_slices]]

            # Concat at input dimension
            new_value = tf.concat(new_tensors, axis=-2)
            self._override_shape(self._filter_var, new_value.shape)
            tf.assign(self._filter_var, new_value, validate_shape=False).op.run(session=session)

            for slot in optimizer.get_slot_names():
                slot_var = optimizer.get_slot(self._filter_var, slot)
                if slot_var is not None:
                    new_slot_values = [slot_var[from_slices],
                                       tf.zeros_like(new_filters),
                                       slot_var[to_slices]]
                    new_value = tf.concat(new_slot_values, axis=-2)
                    self._override_shape(slot_var, new_value.shape)
                    tf.assign(slot_var, new_value, validate_shape=False).op.run(session=session)

    def _override_shape(self, var: tf.Variable, new_shape: tf.TensorShape):
        # TensorFlow fails to infer when we change the shape of a variable, enforce new shape here
        var._ref()._shape = tf.TensorShape(new_shape)
        var.value()._shape = tf.TensorShape(new_shape)

    def _mask_input(self, var: tf.Variable, mask: np.ndarray) -> tf.Operation:
        filter_rank = var.get_shape().ndims
        if filter_rank > 2:
            transposed = tf.transpose(var, perm=[2, 3, 0, 1])
        else:
            transposed = var
        filtered = tf.boolean_mask(transposed, mask)
        if filter_rank > 2:
            filtered = tf.transpose(filtered, perm=[2, 3, 0, 1])
        self._override_shape(var, filtered.shape)
        return tf.assign(var, filtered, validate_shape=False).op

    def _remove_inputs(self, session: tf.Session, optimizer: tf.train.Optimizer, indices: np.ndarray):
        if self._scope is None:
            return
        with tf.variable_scope(self._scope):
            deletion_count = indices.shape[0]
            mask = np.ones([self.input_count + deletion_count], dtype=np.bool_)
            mask[indices] = 0
            self._mask_input(self._filter_var, mask).run(session=session)

            for slot in optimizer.get_slot_names():
                slot_var = optimizer.get_slot(self._filter_var, slot)
                if slot_var is not None:
                    self._mask_input(slot_var, mask).run(session=session)

    def _shape_input(self, input_tensor: tf.Tensor) -> tf.Tensor:
        return input_tensor

    @abstractmethod
    def get_weight_shape(self, input_tensor: tf.Tensor) -> List[int]:
        raise NotImplementedError()

    @abstractmethod
    def _apply_op(self, input_tensor: tf.Tensor, weight: tf.Tensor) -> tf.Tensor:
        raise NotImplementedError()

    def _post_build(self, configuration: NodeBuildConfiguration, post_build_tensors: Dict[str, tf.Tensor]):
        if 'deletion_threshold' not in post_build_tensors:
            post_build_tensors['deletion_threshold'] = self._calc_deletion_threshold(configuration)
        deletion_threshold = post_build_tensors['deletion_threshold']

        with tf.variable_scope(self._scope):
            below_del_threshold = tf.abs(self._scale_var) < deletion_threshold
            self._below_del_threshold_indices = tf.where(below_del_threshold)
            self._below_del_threshold_count = tf.reduce_sum(tf.cast(below_del_threshold, tf.int16))

    def _calc_deletion_threshold(self, configuration: NodeBuildConfiguration) -> tf.Tensor:
        scales = tf.get_collection(self.SCALE_COLLECTION)
        abs_scales = tf.abs(tf.concat(scales, axis=0), 'abs_scales')

        # Tensorflow does not infer shape right when shapes of scales change
        abs_scales._shape = tf.TensorShape([None])

        tf.summary.histogram('abs_scales', abs_scales)

        if configuration.const_neuron_deletion_threshold > 0:
            deletion_threshold = tf.constant(configuration.const_neuron_deletion_threshold)
        else:
            mean, variance = tf.nn.moments(abs_scales, axes=[0], name='scales_moments')
            std = tf.sqrt(variance, name='scales_std')
            # Delete all outliers further away than std
            deletion_threshold = mean - 2 * std

        assert isinstance(deletion_threshold, tf.Tensor)
        tf.summary.scalar('deletion_threshold', deletion_threshold)
        return deletion_threshold

    def _build(self, configuration: NodeBuildConfiguration):
        super()._build(configuration)
        # noinspection PyTypeChecker
        input_tensor = self._concat(tuple(node.output_tensor for node in self.parents))
        with tf.variable_scope(self.label + '_' + self.uuid) as scope:
            self._scope = scope

            input_tensor = self._shape_input(input_tensor)

            weight_shape = self.get_weight_shape(input_tensor)
            weight = tf.get_variable('weight', shape=weight_shape, dtype=tf.float32,
                                     initializer=xavier_initializer())
            bias = tf.get_variable('bias', shape=(self._output_count,), dtype=tf.float32,
                                   initializer=tf.constant_initializer(0.0))
            scale = tf.get_variable('scale', shape=(self._output_count,), dtype=tf.float32,
                                    initializer=tf.constant_initializer(1.0))
            tf.add_to_collection(self.SCALE_COLLECTION, scale)

            output = self._apply_op(input_tensor, weight)

            ema = tf.train.ExponentialMovingAverage(decay=0.9, name='moving_averages')
            batch_mean, batch_variance = tf.nn.moments(output, axes=list(range(len(weight_shape) - 1)), name='moments')
            update_ema = ema.apply([batch_mean, batch_variance])
            tf.add_to_collection(Model.POST_TRAINING_UPDATE_COLLECTION, update_ema)
            mean = tf.cond(configuration.is_training, lambda: batch_mean, lambda: ema.average(batch_mean))
            variance = tf.cond(configuration.is_training, lambda: batch_variance, lambda: ema.average(batch_variance))

            output = tf.nn.batch_normalization(output, mean, variance, bias, scale, 1e-8, name='batch_norm')
            if self.non_linearity:
                output = tf.nn.relu(output, name='activation')

            if self.can_mutate:
                # Calculate Weigend et al. 1990 regularizer
                if configuration.penalty_type == 'weigend':
                    squared_scales = tf.square(scale)
                    self._penalty_per_output = squared_scales / (1e-1 + squared_scales)
                elif configuration.penalty_type == 'linear':
                    self._penalty_per_output = tf.abs(scale)
                else:
                    raise ValueError('Unknown penalty type {}'.format(configuration.penalty_type))
                self._penalty = tf.reduce_sum(self._penalty_per_output)
                if configuration.depth_penalty == 'linear':
                    self._penalty *= self.max_depth
                elif configuration.depth_penalty == 'exponential':
                    self._penalty *= 2 ** self.max_depth
                elif configuration.depth_penalty != 'none':
                    raise ValueError('Unknown depth penalty {}'.format(configuration.depth_penalty))

            tf.summary.scalar('depth', self.max_depth)
            tf.summary.histogram('abs_scale', tf.abs(scale))

        self._filter_var = weight
        self._bias_var = bias
        self._scale_var = scale
        self._ema_mean_var = ema.average(batch_mean)
        self._ema_variance_var = ema.average(batch_variance)
        self.output_tensor = output
        return output


class FullyConnectedNode(VariableNode):

    def __init__(self, parents: List, fixed_output_count: int = None, non_linearity: bool = True):
        super().__init__(parents, non_linearity)
        self.can_mutate = fixed_output_count is None
        self._output_count = fixed_output_count or 16

    @property
    def label(self) -> str:
        return 'fullyconnected_node'

    def _shape_input(self, input_tensor: tf.Tensor) -> tf.Tensor:
        # Pool and reshape if not compatible
        if input_tensor.get_shape().ndims > 2:
            # TODO maxpool instead?
            input_tensor = tf.reduce_max(input_tensor, axis=2, keep_dims=True)
            filter_count = input_tensor.get_shape().as_list()[-1]
            input_tensor = tf.reshape(input_tensor, shape=(-1, filter_count))
        return input_tensor

    def get_weight_shape(self, input_tensor: tf.Tensor) -> List[int]:
        input_size = input_tensor.get_shape()[-1]
        return [input_size, self.output_count]

    def _apply_op(self, input_tensor: tf.Tensor, weight: tf.Tensor) -> tf.Tensor:
        return tf.matmul(input_tensor, weight)


class ConvNode(VariableNode):

    FILTER_WIDTH = 16
    NEW_NODE_PROBABILITY = 0.1

    def __init__(self, parents: List, fixed_output_channel_count: int = None, non_linearity: bool = True):
        super().__init__(parents, non_linearity)
        self.can_mutate = fixed_output_channel_count is None
        self.stride = random.randint(1, 2)
        self._output_count = fixed_output_channel_count or self.OUTPUT_INCREMENT

    @property
    def label(self) -> str:
        return 'conv_node'

    def str_node_properties(self, session: tf.Session) -> List[Tuple[str, str]]:
        properties = super().str_node_properties(session)
        properties.extend([
            ('stride', str(self.stride))
        ])
        return properties

    def __getstate__(self):
        s = super().__getstate__()
        s.update({
            'stride': self.stride,
        })
        return s

    def __setstate__(self, state):
        super().__setstate__(state)
        self.stride = state['stride']

    def grow(self, session: tf.Session, optimizer: tf.train.Optimizer, allow_node_creation: bool):
        create_probabilistically = random.random() < self.NEW_NODE_PROBABILITY
        if allow_node_creation and create_probabilistically:
            self.create_new_node(session, optimizer)
        else:
            super().grow(session, optimizer, allow_node_creation)

    def create_new_node(self, session: tf.Session, optimizer: tf.train.Optimizer):
        old_children = list(self.children)
        new_conv = ConvNode([self])
        for child in old_children:
            child.add_parent(new_conv)
        new_conv._notify_output_addition(session, optimizer, new_conv.output_count)

    def get_weight_shape(self, input_tensor: tf.Tensor) -> List[int]:
        channels_in = input_tensor.get_shape()[-1]
        return [1, self.FILTER_WIDTH, channels_in, self._output_count]

    def _apply_op(self, input_tensor: tf.Tensor, weight: tf.Tensor) -> tf.Tensor:
        return tf.nn.conv2d(input_tensor, weight, strides=[1, 1, self.stride, 1], padding='SAME')


class MutatingCnnModel(Model):

    VOLATILE_VARIABLES = 'VOLATILE_VARIABLES'

    def __init__(self, sample_length: int, learning_rate: float, num_classes: int, batch_size: int,
                 checkpoint_dir: Path, probabilistic_depth_strategy: bool = False, global_avg_pool: bool = True,
                 node_build_configuration: NodeBuildConfiguration = None):

        nodes_file = (checkpoint_dir / 'nodes.pickle')
        if nodes_file.exists():
            with nodes_file.open('rb') as infile:
                self.input_node, self.terminus_node = pickle.load(infile)
        else:
            self.input_node = InputNode()
            first_conv_node = ConvNode([self.input_node])
            if not global_avg_pool:
                fully_connected_node = FullyConnectedNode([first_conv_node])
                self.terminus_node = FullyConnectedNode([fully_connected_node], fixed_output_count=num_classes,
                                                        non_linearity=False)
            else:
                self.terminus_node = ConvNode([first_conv_node], fixed_output_channel_count=num_classes,
                                              non_linearity=False)

        self.probabilistic_depth_strategy = probabilistic_depth_strategy
        self.output_count_history = deque(maxlen=3)
        self.node_build_configuration = node_build_configuration or NodeBuildConfiguration()

        super().__init__(sample_length, learning_rate, num_classes, batch_size)

    def mutate(self, session: tf.Session):
        nodes = self.input_node.all_descendants()
        variable_nodes = [node for node in nodes if isinstance(node, VariableNode)]
        last_node = self.max_depth_mutatable_conv_node
        with tf.variable_scope(self._nodes_scope):
            for node in variable_nodes:
                assert isinstance(node, VariableNode) and node.is_built()
                # The last conv should not be mutated
                if node != last_node:
                    node.mutate(session, self.optimizer, allow_node_creation=self.probabilistic_depth_strategy)

        if not self.probabilistic_depth_strategy:
            self._add_node_if_majority_used(session)

    def _add_node_if_majority_used(self, session: tf.Session):
        last_node = self.max_depth_mutatable_conv_node
        if last_node.get_usage(session) > 0.5:
            logging.info('Create new node at last_node with depth {}'.format(last_node.max_depth))
            last_node.create_new_node(session, self.optimizer)

    def _add_node_in_equilibrium(self, session: tf.Session):
        nodes = self.input_node.all_descendants()
        history = self.output_count_history
        history.append(sum(node.output_count for node in nodes))
        # When the number of outputs didn't change over 3 iterations, create a new node
        if len(set(history)) <= 1:
            self.max_depth_mutatable_conv_node.create_new_node(session, self.optimizer)

    # noinspection PyTypeChecker,PyUnresolvedReferences
    @property
    def max_depth_mutatable_conv_node(self) -> ConvNode:
        nodes = self.input_node.all_descendants()
        conv_nodes = (node for node in nodes if isinstance(node, ConvNode) and node.can_mutate)
        return max(conv_nodes, key=lambda n: n.max_depth)

    def build(self):
        if self.input_node is not None:
            self.input_node.reset_all()
        super().build()
        self.init = tf.group(self.init, tf.variables_initializer(tf.get_collection(self.VOLATILE_VARIABLES)))

    def _define_loss(self, cross_entropy: tf.Tensor) -> tf.Tensor:
        loss = super()._define_loss(cross_entropy)
        tf.summary.scalar('cross_entropy', loss)
        nodes = self.input_node.all_descendants()
        penalties = [node.penalty for node in nodes if node.penalty is not None]
        penalty_sum = tf.add_n(penalties, name='penalty_sum')
        reg_penalty = VariableNode.L1_NORM_PENALTY_STRENGTH * penalty_sum
        tf.summary.scalar('l1_penalty', reg_penalty)
        # noinspection PyTypeChecker
        return loss + reg_penalty

    def _define_training(self):
        scales = tf.get_collection(VariableNode.SCALE_COLLECTION)
        self.scales = tf.concat(scales, axis=0)
        self._track_scales_diff()

        super()._define_training()

        self._track_scales_gradient(scales)
        self._define_scales_training(scales)

    def _define_scales_training(self, scales: List[tf.Variable]):
        train_op = self.optimizer.minimize(self.loss, global_step=self.global_step,
                                           var_list=scales, name='train_scales_op')
        self.train_scales_op = self._with_post_training_update(train_op)

    def _track_scales_gradient(self, scales: List[tf.Variable]):
        scale_gradients = tf.concat(tf.gradients(self.loss, scales), axis=0)
        tf.summary.histogram('scale_gradients', scale_gradients)

    def _track_scales_diff(self):
        self.previous_scales = tf.Variable(tf.zeros_like(self.scales), trainable=False,
                                           collections=[self.VOLATILE_VARIABLES], name='previous_scales')
        self.scales_diff = self.scales - self.previous_scales
        tf.summary.histogram('scales_diff', self.scales_diff)
        with tf.control_dependencies([self.scales_diff]):
            self.memorize_previous_scales = tf.assign(self.previous_scales, self.scales)
        tf.add_to_collection(self.POST_TRAINING_UPDATE_COLLECTION, self.memorize_previous_scales)

    def _vars_to_restore(self) -> Union[None, List[tf.Variable]]:
        not_saved_uuids = [n.uuid for n in self.input_node.all_descendants() if not n.vars_saved]
        restore_vars = [var for var in tf.global_variables() if all(uuid not in var.name for uuid in not_saved_uuids)]
        return restore_vars

    def restore(self, session: tf.Session, checkpoint_dir: Path):
        # Init all values first, because not all are saved
        session.run(self.init)
        super().restore(session, checkpoint_dir)

    def save(self, session: tf.Session, checkpoint_dir: Path):
        super().save(session, checkpoint_dir)
        built_nodes = (n for n in self.input_node.all_descendants() if n.is_built())
        for node in built_nodes:
            node.vars_saved = True
        with (checkpoint_dir / 'nodes.pickle').open('wb') as outfile:
            pickle.dump((self.input_node, self.terminus_node), outfile)

    def render_graph(self, session: tf.Session, render_dir: Path = None,
                     render_file: Path = None, render_format: str = 'png'):
        if render_dir is None and render_file is None:
            raise ValueError('Must specify either a render directory or render filepath.')
        with tempfile.TemporaryDirectory() as tmp_dir:
            step = self.global_step.eval(session=session)
            graph = self.input_node.to_graphviz(session, step, Path(tmp_dir))
            graph.format = render_format
            if render_file is None:
                render_file = render_dir / 'graph-{}'.format(step)
            else:
                render_file = render_file.with_name(render_file.stem)
            graph.render(str(render_file), cleanup=True)

    def _create_network(self, input_2d: tf.Tensor) -> tf.Tensor:
        self.is_training = tf.placeholder(tf.bool, name='is_training')
        self.node_build_configuration.is_training = self.is_training

        with tf.variable_scope('nodes') as scope:
            self._nodes_scope = scope
            self.input_node.build_dag(input_2d, self.node_build_configuration)
            output = self.terminus_node.output_tensor
            assert isinstance(output, tf.Tensor)

        # Remove width and height dimensions if no fully connected layer exists
        output_shape = output.get_shape()
        if output_shape.ndims == 4:
            ksize = [1] + output_shape.as_list()[1:3] + [1]
            output = tf.nn.avg_pool(output, ksize=ksize, strides=ksize, padding='VALID')
            output = tf.squeeze(output, axis=[1, 2])

        nodes = self.input_node.all_descendants()
        tf.summary.scalar('node_count', len(nodes))
        tf.summary.scalar('max_depth', self.terminus_node.max_depth)
        tf.summary.scalar('total_output_count', sum(n.output_count for n in nodes))

        logits = tf.identity(output, name='logits')

        return logits

    def step(self, session: tf.Session, feed_dict: Dict, loss=False, train=False, logits=False, correct_count=False,
             update_summary=False, train_switches=False):
        feed_dict[self.is_training] = train
        output_feed = [self.global_step]

        if loss:
            output_feed.append(self.loss)
        if train_switches:
            output_feed.append(self.train_scales_op)
        elif train:  # We either train switches or everything
            output_feed.append(self.train_op)
        if logits:
            output_feed.append(self.logits)
        if correct_count:
            output_feed.append(self.correct_count)
        if update_summary:
            output_feed.append(self.summary)

        results = session.run(output_feed, feed_dict=feed_dict)

        if update_summary:
            return self._extract_summary(results, session)

        return results


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
