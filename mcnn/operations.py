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


def _evaluate_in_session(session: tf.Session, model: Model, dataset: Dataset) -> Tuple[int, float, float, tf.Summary]:

    batch_size = model.batch_size
    if dataset.test_sample_count:
        batch_size = min(batch_size, dataset.test_sample_count)

    test_sample_generator = dataset.data_generator('test', batch_size)

    sample_total_count = 0
    correct_total_count = 0
    batches_count = 0
    loss_sum = 0
    global_step = 0

    for input, labels in test_sample_generator:
        feed_dict = {
            model.input: input,
            model.labels: labels
        }
        global_step, test_loss, correct_count = model.step(session, feed_dict, loss=True, correct_count=True)
        sample_total_count += input.shape[0]
        correct_total_count += correct_count
        batches_count += 1
        loss_sum += test_loss

    test_accuracy = correct_total_count / sample_total_count
    testing_error = 1 - test_accuracy
    test_loss = loss_sum / batches_count
    test_summary = tf.Summary()
    test_summary.value.add(tag='evaluation/accuracy', simple_value=test_accuracy)
    test_summary.value.add(tag='evaluation/testing_error', simple_value=testing_error)
    test_summary.value.add(tag='training/loss', simple_value=test_loss)

    return global_step, test_loss, test_accuracy, test_summary


def evaluate(model: Model, dataset: Dataset, checkpoint_dir: Path, log_dir: Path):

    with create_session() as session:

        model.restore(session, checkpoint_dir)
        model.setup_summary_writer(session, log_dir)

        global_step, loss, accuracy, summary = _evaluate_in_session(session, model, dataset)
        model.summary_writer.add_summary(summary, global_step)

        print('[Test] Loss {:.2f} Accuracy {:.2f}%'.format(loss, accuracy * 100))
        return accuracy


def visualize_lrp(model: Model, dataset: Dataset, checkpoint_dir: Path, heatmap_save_path: Path = None):
    import matplotlib.colors
    import matplotlib.pyplot as plt

    relevance = lrp.lrp(model.input, model.logits, 0.0, 1.0)

    with create_session() as session:

        model.restore(session, checkpoint_dir)

        test_sample_generator = dataset.data_generator('test', model.batch_size)

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


class ReduceOnPlateau:

    def __init__(self, tf_var: tf.Variable, factor: float = 0.5, patience: int = 50,
                 min_value: float = 1e-4, threshold_epsilon: float = 1e-4):
        self.tf_var = tf_var
        self.factor = factor
        self.patience = patience
        self.min_value = min_value
        self.threshold_epsilon = threshold_epsilon
        self.value_epsilon = self.min_value * 1e-4
        self.best = None
        self.wait = 0

    def update(self, session: tf.Session, loss: float):
        value = self.tf_var.eval(session=session)
        if value < self.min_value + self.value_epsilon:
            return
        if self.best is not None:
            if loss < self.best - self.threshold_epsilon:
                self.best = loss
                self.wait = 0
            else:
                if self.wait >= self.patience:
                    new_value = value * self.factor
                    new_value = max(new_value, self.min_value)
                    self.tf_var.assign(new_value).op.run(session=session)
                    self.wait = 0
                self.wait += 1
        else:
            self.best = loss


class TrainingResult:

    def __init__(self):
        self.best_item = None

    def update(self, training_loss: float, training_accuracy: float, test_loss: float, test_accuracy: float):
        if self.best_item is None or self.best_item[0] > training_loss:
            self.best_item = (training_loss, training_accuracy, test_loss, test_accuracy)

    def clear(self):
        self.best_item = None

    @property
    def best_test_accuracy(self):
        return self.best_item[3]

    @property
    def best_test_loss(self):
        return self.best_item[2]


def train(model: Model, dataset: Dataset, epoch_count: int, checkpoint_dir: Path, log_dir_train: Path,
          log_dir_test: Path, steps_per_checkpoint: int, steps_per_summary: int, checkpoint_written_callback: Callable,
          save: bool = True) -> TrainingResult:

    batch_size = model.batch_size
    if dataset.train_sample_count:
       batch_size = min(batch_size, dataset.train_sample_count // 10)

    result = TrainingResult()
    with create_session() as session:

        model.restore_or_create(session, checkpoint_dir)
        train_writer = tf.summary.FileWriter(str(log_dir_train))
        test_writer = tf.summary.FileWriter(str(log_dir_test))

        plateau_reducer = ReduceOnPlateau(model.learning_rate)
        global_step = 0

        for epoch in range(epoch_count):
            train_loss = None
            train_accuracy = None
            is_summary_step = None
            for x, y in dataset.data_generator('train', batch_size):
                is_checkpoint_step = (global_step + 1) % steps_per_checkpoint == 0
                is_summary_step = (global_step + 1) % steps_per_summary == 0
                feed_dict = {
                    model.input: x,
                    model.labels: y
                }
                step_result = model.step(session, feed_dict, loss=True, train=True, accuracy=True,
                                         update_summary=is_summary_step, alternative_summary_writer=train_writer)
                global_step, train_loss, _, train_accuracy = step_result

                if is_checkpoint_step:
                    print('[Train] Step {} Loss {:.2f} Accuracy {:.2f}%'.format(global_step, train_loss,
                                                                                train_accuracy * 100))
                    if save:
                        model.save(session, checkpoint_dir)
                        logging.info('Model saved')
                        if checkpoint_written_callback is not None:
                            # noinspection PyCallingNonCallable
                            checkpoint_written_callback()

                if is_summary_step:
                    global_step, test_loss, test_accuracy, test_summary = _evaluate_in_session(session, model, dataset)
                    test_writer.add_summary(test_summary, global_step)
                    result.update(train_loss, train_accuracy, test_loss, test_accuracy)

            if not is_summary_step:
                global_step, test_loss, test_accuracy, test_summary = _evaluate_in_session(session, model, dataset)
                result.update(train_loss, train_accuracy, test_loss, test_accuracy)
            plateau_reducer.update(session, train_loss)

        train_writer.add_graph(session.graph, global_step=global_step)
    return result


class MutationTrainer:

    def __init__(self, model: MutatingCnnModel, dataset: Dataset, checkpoint_dir: Path, log_dir_train: Path,
                 log_dir_test: Path, plot_dir: Path, steps_per_checkpoint: int, steps_per_summary: int,
                 train_only_switches_fraction: float, freeze_on_delete: bool,
                 delete_shrinking_last_node: bool, only_switches_lr: float, epochs_after_frozen: int,
                 freeze_on_shrinking_total_outputs: bool):
        self.model = model
        self.dataset = dataset
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_dir_mutated = checkpoint_dir.with_name(checkpoint_dir.name + '_mutated')
        self.plot_dir = plot_dir
        self.steps_per_checkpoint = steps_per_checkpoint
        self.steps_per_summary = steps_per_summary
        self.train_only_switches_fraction = train_only_switches_fraction
        self.freeze_on_delete =freeze_on_delete
        self.delete_shrinking_last_node = delete_shrinking_last_node
        self.only_switches_lr = only_switches_lr
        self.epochs_after_frozen = epochs_after_frozen
        self.freeze_on_shrinking_total_outputs = freeze_on_shrinking_total_outputs

        self.train_writer = tf.summary.FileWriter(str(log_dir_train))
        self.test_writer = tf.summary.FileWriter(str(log_dir_test))

        self.session = None
        self.epochs_left = 0

        self.batch_size = model.batch_size
        if dataset.train_sample_count:
            self.batch_size = min(self.batch_size, dataset.train_sample_count // 10)

    def _restart_session(self):
        if self.session is not None:
            self.session.close()
        new_session = create_session()
        # TODO first load probably must be non mutated version
        self.model.restore_or_create(new_session, self.checkpoint_dir_mutated)
        self.session = new_session

    def _close_session(self):
        self.session.close()
        self.session = None

    def _mutate(self, result: TrainingResult):
        architecture_frozen_previously = self.model.architecture_frozen
        self.model.mutate(self.session, self.freeze_on_delete, self.delete_shrinking_last_node,
                          self.freeze_on_shrinking_total_outputs)
        logging.info('Model mutated')
        if self.model.architecture_frozen and not architecture_frozen_previously and self.epochs_after_frozen != -1:
            # Only keep results after freezing
            result.clear()
            # Stop training soon
            self.epochs_left = self.epochs_after_frozen
        self.model.save(self.session, self.checkpoint_dir_mutated)
        logging.info('Model saved')
        self._close_session()
        self.model.build()
        logging.info('Model rebuilt')
        self._restart_session()

    @staticmethod
    def _parse_summary(serialized: bytes) -> tf.Summary:
        summary = tf.Summary()
        summary.ParseFromString(serialized)
        return summary

    def train(self, epoch_count: int, checkpoint_written_callback: Callable = None) -> TrainingResult:

        result = TrainingResult()
        lr_plateau_reducer = ReduceOnPlateau(self.model.learning_rate)
        graph_file = self.plot_dir / 'graph.png'
        global_step = 0
        self.epochs_left = epoch_count
        self._restart_session()

        while self.epochs_left > 0:
            self.epochs_left -= 1
            train_loss = None
            train_ce = None
            train_accuracy = None
            is_summary_step = None
            for x, y in self.dataset.data_generator('train', self.batch_size):
                is_checkpoint_step = (global_step + 1) % self.steps_per_checkpoint == 0
                is_summary_step = (global_step + 1) % self.steps_per_summary == 0
                checkpoint_progress = (global_step % self.steps_per_checkpoint) / self.steps_per_checkpoint
                feed_dict = {
                    self.model.input: x,
                    self.model.labels: y
                }
                train_only_switches = not self.model.architecture_frozen and \
                                      (0.5 - self.train_only_switches_fraction / 2) \
                                      <= checkpoint_progress < \
                                      (0.5 + self.train_only_switches_fraction / 2)
                if train_only_switches:
                    feed_dict[self.model.learning_rate] = self.only_switches_lr

                step_result = self.model.step(self.session,
                                              feed_dict,
                                              loss=True,
                                              cross_entropy=True,
                                              train=True,
                                              accuracy=True,
                                              return_summary=is_summary_step,
                                              alternative_summary_writer=self.train_writer,
                                              train_switches=train_only_switches,
                                              train_wo_penalty=self.model.architecture_frozen)
                global_step, train_loss, train_ce, _, train_accuracy = step_result[:5]
                train_summary = self._parse_summary(step_result[-1]) if is_summary_step else None

                if is_summary_step:
                    self.model.render_graph(self.session, render_file=graph_file)
                    with graph_file.open(mode='rb') as f:
                        graph_bytes = f.read()
                    img_summary = tf.Summary.Image(encoded_image_string=graph_bytes)
                    train_summary.value.add(tag='graph_rendering', image=img_summary)

                    step_result = _evaluate_in_session(self.session, self.model, self.dataset)
                    global_step, test_loss, test_accuracy, test_summary = step_result
                    result.update(train_ce, train_accuracy, test_loss, test_accuracy)

                    test_summary.value.add(tag='evaluation/best_accuracy', simple_value=result.best_test_accuracy)
                    test_summary.value.add(tag='evaluation/best_loss', simple_value=result.best_test_loss)

                    self.train_writer.add_summary(train_summary, global_step)
                    self.test_writer.add_summary(test_summary, global_step)

                    print('[Train] Step {} Loss {:.2f} Accuracy {:.2f}%'.format(global_step,
                                                                                train_loss,
                                                                                train_accuracy * 100))
                    print('[Test ] Step {} Loss {:.2f} Accuracy {:.2f}%'.format(global_step,
                                                                                test_loss,
                                                                                test_accuracy * 100))
                    print('[Test ] Best {} Loss {:.2f} Accuracy {:.2f}%'.format(global_step,
                                                                                result.best_test_loss,
                                                                                result.best_test_accuracy * 100))

                if is_checkpoint_step:
                    self.model.save(self.session, self.checkpoint_dir)
                    logging.info('Model saved')
                    if checkpoint_written_callback is not None:
                        # noinspection PyCallingNonCallable
                        checkpoint_written_callback()

                    if not self.model.architecture_frozen:
                        self._mutate(result)
                        lr_plateau_reducer.tf_var = self.model.learning_rate

            # Evaluate at end of epoch
            if not is_summary_step:
                step_result = _evaluate_in_session(self.session, self.model, self.dataset)
                global_step, test_loss, test_accuracy, test_summary = step_result
                result.update(train_ce, train_accuracy, test_loss, test_accuracy)
            lr_plateau_reducer.update(self.session, train_loss)

        self.train_writer.add_graph(self.session.graph, global_step=global_step)
        self.train_writer.flush()
        self.test_writer.flush()
        self._close_session()
        return result
