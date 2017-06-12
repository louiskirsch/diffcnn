from collections import deque
from typing import Set

import tensorflow as tf
import numpy as np
from tensorflow.python.ops import nn_ops, gen_nn_ops, array_grad


def fprop_first(F, W, X, lowest, highest):
    # Propagate from last feedforward layer to input
    W, V, U = W, tf.maximum(0.0, W), tf.minimum(0.0, W)
    X, L, H = X, X * 0 + lowest, X * 0 + highest

    Z = tf.matmul(X, W) - tf.matmul(L, V) - tf.matmul(H, U) + 1e-9
    S = F / Z
    F = X * tf.matmul(S, tf.transpose(W)) - L * tf.matmul(S, tf.transpose(V)) - H * tf.matmul(S, tf.transpose(U))
    return F


def fprop(F, W, X):
    # Propagate over feedforward layer
    V = tf.maximum(0.0, W)
    Z = tf.matmul(X, V) + 1e-9
    S = F / Z
    C = tf.matmul(S, tf.transpose(V))
    F = X * C
    return F


def fprop_conv_first(F, W, X, lowest, highest, strides=None, padding='SAME'):
    # Propagate from last conv layer to input
    strides = [1, 1, 1, 1] if strides is None else strides

    Wn = tf.minimum(0.0, W)
    Wp = tf.maximum(0.0, W)

    X, L, H = X, X * 0 + lowest, X * 0 + highest

    c = tf.nn.conv2d(X, W, strides, padding)
    cp = tf.nn.conv2d(H, Wp, strides, padding)
    cn = tf.nn.conv2d(L, Wn, strides, padding)
    Z = c - cp - cn + 1e-9
    S = F / Z

    g = nn_ops.conv2d_backprop_input(tf.shape(X), W, S, strides, padding)
    gp = nn_ops.conv2d_backprop_input(tf.shape(X), Wp, S, strides, padding)
    gn = nn_ops.conv2d_backprop_input(tf.shape(X), Wn, S, strides, padding)
    F = X * g - L * gp - H * gn
    return F


def fprop_conv(F, W, X, strides=None, padding='SAME'):
    # Propagate over conv layer
    xshape = X.get_shape().as_list()
    fshape = F.get_shape().as_list()
    if len(xshape) != len(fshape):
        F = tf.reshape(F, (-1, xshape[1], xshape[2], fshape[-1] / (xshape[1] * xshape[2])))
    strides = [1, 1, 1, 1] if strides is None else strides
    W = tf.maximum(0.0, W)

    Z = tf.nn.conv2d(X, W, strides, padding) + 1e-9
    S = F / Z
    C = nn_ops.conv2d_backprop_input(tf.shape(X), W, S, strides, padding)
    F = X * C
    return F


def fprop_pool(F, X, strides=None, ksize=None, padding='SAME'):
    # Propagate over pool layer
    xshape = X.get_shape().as_list()
    fshape = F.get_shape().as_list()
    if len(xshape) != len(fshape):
        F = tf.reshape(F, (-1, int(np.ceil(xshape[1] / 2.0)),
                           int(np.ceil(xshape[2] / 2.0)), xshape[3]))
    ksize = [1, 2, 2, 1] if ksize is None else ksize
    strides = [1, 2, 2, 1] if strides is None else strides

    Z = tf.nn.max_pool(X, strides=strides, ksize=ksize, padding=padding) + 1e-9
    S = F / Z
    C = gen_nn_ops._max_pool_grad(X, Z, S, ksize, strides, padding)
    F = X * C
    return F


SUPPORTED_OPS = {'MatMul', 'Conv2D', 'MaxPool', 'Relu', 'BiasAdd', 'Identity', 'Max', 'Reshape', 'ConcatV2'}


def _all_parent_ops(child_op: tf.Operation, filter_ops: Set[str], ops: Set[tf.Operation] = None) -> Set[tf.Operation]:
    if ops is None:
        ops = set()
    if child_op in ops:
        return ops
    ops.add(child_op)
    parent_ops = [inp.op for inp in child_op.inputs if inp.op.type in filter_ops]
    for op in parent_ops:
        _all_parent_ops(op, filter_ops, ops)
    return ops


def lrp(input_tensor: tf.Tensor, logits: tf.Tensor, lowest: float, highest: float) -> tf.Tensor:

    tensors_to_process = deque([logits])
    all_used_ops = _all_parent_ops(logits.op, SUPPORTED_OPS)
    ops_processed = set()

    max_class = tf.argmax(logits, axis=-1)
    class_count = logits.get_shape()[-1]
    relevance_of_max_class = tf.nn.relu(logits * tf.one_hot(max_class, depth=class_count))
    signal_to_relevance = {logits: relevance_of_max_class}

    def add_tensor_for_processing(tensor: tf.Tensor):
        if tensor == input_tensor:
            return
        # Only add tensor for processing if all its consumers are processed
        consumers = set(tensor.consumers()).intersection(all_used_ops)
        if all(consumer in ops_processed for consumer in consumers):
            tensors_to_process.append(tensor)

    def update_signal_relevance(signal: tf.Tensor, relevance: tf.Tensor):
        if signal not in signal_to_relevance:
            signal_to_relevance[signal] = relevance
        else:
            # TODO Is adding up relevances in the case of multiple consumers correct?
            old_relevance = signal_to_relevance[signal]
            signal_to_relevance[signal] = old_relevance + relevance

    while len(tensors_to_process) > 0:

        processing_tensor = tensors_to_process.pop()
        relevance_tensor = signal_to_relevance[processing_tensor]
        last_op = processing_tensor.op
        assert isinstance(last_op, tf.Operation)
        ops_processed.add(last_op)

        if last_op.type == 'MatMul':
            signal, weights = last_op.inputs
            if signal == input_tensor:
                relevance_tensor = fprop_first(relevance_tensor, weights, signal, lowest, highest)
            else:
                relevance_tensor = fprop(relevance_tensor, weights, signal)
        elif last_op.type == 'Conv2D':
            signal, weights = last_op.inputs
            strides = last_op.get_attr('strides')
            padding = last_op.get_attr('padding')
            if signal == input_tensor:
                relevance_tensor = fprop_conv_first(relevance_tensor, weights, signal, lowest, highest, strides, padding)
            else:
                relevance_tensor = fprop_conv(relevance_tensor, weights, signal, strides, padding)
        elif last_op.type == 'MaxPool':
            signal = last_op.inputs[0]
            strides = last_op.get_attr('strides')
            ksize = last_op.get_attr('ksize')
            padding = last_op.get_attr('padding')
            relevance_tensor = fprop_pool(relevance_tensor, signal, strides, ksize, padding)
        elif last_op.type in ['Relu', 'BiasAdd', 'Identity']:
            signal = last_op.inputs[0]
        elif last_op.type == 'Max':
            # TODO this could be implemented more efficiently
            signal = last_op.inputs[0]
            signal_shape = signal.get_shape().as_list()
            strides = [1, 1, 1, 1]
            ksize = [1, signal_shape[1], signal_shape[2], 1]
            relevance_tensor = fprop_pool(relevance_tensor, signal, strides, ksize, padding='VALID')
        elif last_op.type == 'Reshape':
            signal = last_op.inputs[0]
            relevance_tensor = tf.reshape(relevance_tensor, tf.shape(signal))
        elif last_op.type == 'StridedSlice':
            signal = last_op.inputs[0]
            relevance_tensor = array_grad._StridedSliceGrad(last_op, relevance_tensor)[0]
        elif last_op.type == 'ConcatV2':
            signals = last_op.inputs[:-1]
            relevances = array_grad._ConcatGradV2(last_op, relevance_tensor)[:-1]
            for signal, relevance in zip(signals, relevances):
                update_signal_relevance(signal, relevance)
                add_tensor_for_processing(signal)
            continue
        else:
            raise ValueError('Operation type {} not supported'.format(last_op.type))

        update_signal_relevance(signal, relevance_tensor)
        add_tensor_for_processing(signal)

    assert input_tensor in signal_to_relevance
    return signal_to_relevance[input_tensor]
