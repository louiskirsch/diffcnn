from collections import deque
from pathlib import Path
from typing import Callable, Tuple

import tensorflow as tf
import numpy as np
import itertools
import logging

from mcnn import lrp
from mcnn.model import Model, MutatingCnnModel
from mcnn.samples import Dataset


def create_session() -> tf.Session:
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    session = tf.Session(config=config)
    return session


def evaluate(model: Model, dataset: Dataset, checkpoint_dir: Path, log_dir: Path,
             feature_name: str):

    with create_session() as session:

        model.restore(session, checkpoint_dir)
        model.setup_summary_writer(session, log_dir)

        # Some datasets have even fewer samples than batch_size
        if dataset.test_sample_count:
            model.batch_size = min(model.batch_size, dataset.test_sample_count)

        test_sample_generator = dataset.data_generator('test', model.batch_size, feature_name=feature_name, loop=False,
                                                       sample_length=model.sample_length)

        sample_total_count = 0
        correct_total_count = 0
        batches_count = 0
        loss_sum = 0
        global_step = None

        for input, labels in test_sample_generator:
            feed_dict = {
                model.input: input,
                model.labels: labels
            }
            global_step, loss, correct_count = model.step(session, feed_dict, loss=True, correct_count=True)
            sample_total_count += input.shape[0]
            correct_total_count += correct_count
            batches_count += 1
            loss_sum += loss

        accuracy = correct_total_count / sample_total_count
        testing_error = 1 - accuracy
        loss = loss_sum / batches_count
        summary = tf.Summary()
        summary.value.add(tag='evaluation/accuracy', simple_value=accuracy)
        summary.value.add(tag='evaluation/testing_error', simple_value=testing_error)
        summary.value.add(tag='training/loss', simple_value=loss)
        model.summary_writer.add_summary(summary, global_step)

        print('[Test] Loss {:.2f} Accuracy {:.2f}%'.format(loss, accuracy * 100))
        return accuracy


def visualize_lrp(model: Model, dataset: Dataset, checkpoint_dir: Path, feature_name: str,
                  heatmap_save_path: Path = None):
    import matplotlib.colors
    import matplotlib.pyplot as plt

    relevance = lrp.lrp(model.input, model.logits, 0.0, 1.0)

    with create_session() as session:

        model.restore(session, checkpoint_dir)

        test_sample_generator = dataset.data_generator('test', model.batch_size, feature_name=feature_name, loop=False,
                                                       sample_length=model.sample_length)

        reconstructions = []

        for input, labels in test_sample_generator:
            feed_dict = {
                model.input: input
            }
            heatmap, confidences = session.run([relevance, tf.nn.softmax(model.logits)], feed_dict=feed_dict)

            for batch_idx in range(input.shape[0]):
                orig = input[batch_idx, ...]
                heat = heatmap[batch_idx, ...]
                label = labels[batch_idx]
                confidence = confidences[batch_idx, label]
                reconstructions.append((orig, heat, label, confidence))

        color_map = matplotlib.colors.LinearSegmentedColormap.from_list('heat', [(0, 0, 0), (1, 0, 0)])
        size = np.ceil(np.sqrt(dataset.target_classes_count))
        plt.figure(figsize=(size * 6, size * 4))

        group_by_label = lambda r: r[2]
        reconstructions.sort(key=group_by_label)
        for idx, (label, items) in enumerate(itertools.groupby(reconstructions, key=group_by_label)):
            items = list(items)
            # noinspection PyTypeChecker
            max_item = int(np.argmax([i[3] for i in items]))
            orig, heat, label, confidence = items[max_item]
            heat = np.clip((heat - np.mean(heat)) / np.std(heat), 0, 1)
            length = orig.shape[0]
            x = range(length)
            y = orig
            plt.subplot(size, size, idx + 1)
            for other_item in items:
                plt.plot(other_item[0], c='gray', alpha=0.2)
            plt.plot(y, c='black')
            plt.scatter(x, y, c=heat, cmap=color_map)
            plt.title('Class {} with {:.2f}% confidence'.format(label, confidence * 100))

        plt.tight_layout(pad=2.0, h_pad=3.0)
        if heatmap_save_path is not None:
            plt.savefig(str(heatmap_save_path))
        else:
            plt.show()
        plt.close()


class ReduceLROnPlateau:

    def __init__(self, initial_learning_rate: float, factor: float = 0.5, patience: int = 50,
                 min_lr: float = 1e-4, threshold_epsilon: float = 1e-4):
        self.learning_rate = initial_learning_rate
        self.factor = factor
        self.patience = patience
        self.min_lr = min_lr
        self.threshold_epsilon = threshold_epsilon
        self.lr_epsilon = self.min_lr * 1e-4
        self.best = None
        self.wait = 0

    def update(self, loss: float):
        # TODO actually we should store the learning rate in a tf variable to reuse on restart
        if self.learning_rate < self.min_lr + self.lr_epsilon:
            return
        if self.best is not None:
            if loss < self.best - self.threshold_epsilon:
                self.best = loss
                self.wait = 0
            else:
                if self.wait >= self.patience:
                    new_lr = self.learning_rate * self.factor
                    new_lr = max(new_lr, self.min_lr)
                    logging.info('Setting learning rate to {}'.format(new_lr))
                    self.learning_rate = new_lr
                    self.wait = 0
                self.wait += 1
        else:
            self.best = loss


def train(model: Model, dataset: Dataset, step_count: int, checkpoint_dir: Path, log_dir: Path,
          steps_per_checkpoint: int, feature_name: str, checkpoint_written_callback: Callable,
          summary_every_step: bool, save: bool = True):

    with create_session() as session:

        model.restore_or_create(session, checkpoint_dir)
        model.setup_summary_writer(session, log_dir)

        train_sample_generator = dataset.data_generator('train', model.batch_size, feature_name=feature_name,
                                                        sample_length=model.sample_length, loop=True)
        train_sample_iterator = iter(train_sample_generator)

        plateau_reducer = ReduceLROnPlateau(model.default_learning_rate)

        for train_step in range(step_count):
            is_checkpoint_step = (train_step + 1) % steps_per_checkpoint == 0
            input, labels = next(train_sample_iterator)
            feed_dict = {
                model.input: input,
                model.labels: labels,
                model.learning_rate: plateau_reducer.learning_rate
            }
            if is_checkpoint_step:
                global_step, loss, _, correct_count = model.step(session, feed_dict, loss=True, train=True,
                                                                 correct_count=True, update_summary=True)
                accuracy = correct_count / model.batch_size
                print('[Train] Step {} Loss {:.2f} Accuracy {:.2f}%'.format(global_step, loss, accuracy * 100))
                if save:
                    model.save(session, checkpoint_dir)
                    logging.info('Model saved')
                    if checkpoint_written_callback is not None:
                        # noinspection PyCallingNonCallable
                        checkpoint_written_callback()
            else:
                global_step, loss, _ = model.step(session, feed_dict, loss=True, train=True,
                                                  update_summary=summary_every_step)
            # Run roughly every epoch
            if train_step % (dataset.train_sample_count // model.batch_size) == 0:
                plateau_reducer.update(loss)


def train_and_mutate(model: MutatingCnnModel, dataset: Dataset, step_count: int, checkpoint_dir: Path, log_dir: Path,
                     plot_dir: Path, steps_per_checkpoint: int, feature_name: str,
                     checkpoint_written_callback: Callable, render_graph_steps: int,
                     train_only_switches_fraction: float, summary_every_step: bool, freeze_on_delete: bool,
                     delete_shrinking_last_node: bool, only_switches_lr: float, checkpoints_after_frozen: int,
                     freeze_on_shrinking_total_outputs: bool):

    checkpoint_dir_mutated = checkpoint_dir.with_name(checkpoint_dir.name + '_mutated')

    # Some datasets have even fewer samples than batch_size
    if dataset.train_sample_count:
        model.batch_size = min(model.batch_size, dataset.train_sample_count)

    train_sample_generator = dataset.data_generator('train', model.batch_size, feature_name=feature_name,
                                                    sample_length=model.sample_length, loop=True)
    train_sample_iterator = iter(train_sample_generator)

    plateau_reducer = ReduceLROnPlateau(model.default_learning_rate)
    graph_file = plot_dir / 'graph.png'
    steps_left = step_count

    while steps_left > 0:
        iterate_step_count = min(steps_left, steps_per_checkpoint)
        steps_left -= iterate_step_count

        with create_session() as session:
            model.restore_or_create(session, checkpoint_dir_mutated)
            model.setup_summary_writer(session, log_dir)

            logging.info('Started training')
            for step in range(iterate_step_count):
                input, labels = next(train_sample_iterator)
                feed_dict = {
                    model.input: input,
                    model.labels: labels,
                    model.learning_rate: plateau_reducer.learning_rate
                }
                train_only_switches = not model.architecture_frozen and \
                                      (0.5 - train_only_switches_fraction / 2) * iterate_step_count \
                                      <= float(step) < \
                                      (0.5 + train_only_switches_fraction / 2) * iterate_step_count
                if train_only_switches:
                    feed_dict[model.learning_rate] = only_switches_lr
                if step < iterate_step_count - 1:
                    global_step, loss, _ = model.step(session, feed_dict, loss=True, train=True,
                                                      update_summary=summary_every_step,
                                                      train_switches=train_only_switches,
                                                      train_wo_penalty=model.architecture_frozen)
                else:
                    global_step, loss, _, correct_count = model.step(session, feed_dict, loss=True, train=True,
                                                                     correct_count=True, update_summary=True,
                                                                     train_switches=train_only_switches,
                                                                     train_wo_penalty=model.architecture_frozen)
                    accuracy = correct_count / model.batch_size
                    print('[Train] Step {} Loss {:.2f} Accuracy {:.2f}%'.format(global_step, loss, accuracy * 100))

                    model.save(session, checkpoint_dir)
                    logging.info('Model saved')
                    if checkpoint_written_callback is not None:
                        # noinspection PyCallingNonCallable
                        checkpoint_written_callback()

                    architecture_frozen_previously = model.architecture_frozen
                    model.mutate(session, freeze_on_delete, delete_shrinking_last_node,
                                 freeze_on_shrinking_total_outputs)
                    logging.info('Model mutated')
                    if model.architecture_frozen and not architecture_frozen_previously:
                        # Stop training soon
                        steps_left = checkpoints_after_frozen * steps_per_checkpoint
                    model.save(session, checkpoint_dir_mutated)
                    logging.info('Model saved')
                if render_graph_steps and step % render_graph_steps == 0:
                    model.render_graph(session, render_file=graph_file)
                    with graph_file.open(mode='rb') as f:
                        graph_bytes = f.read()
                    img_summary = tf.Summary.Image(encoded_image_string=graph_bytes)
                    summary = tf.Summary(value=[tf.Summary.Value(tag='graph_rendering', image=img_summary)])
                    model.summary_writer.add_summary(summary, global_step)
                # Run roughly every epoch
                if global_step % (dataset.train_sample_count // model.batch_size) == 0:
                    plateau_reducer.update(loss)

        model.build()
        logging.info('Model rebuilt')

