import multiprocessing
from multiprocessing import Process

import logging

import mcnn.operations as operations
from mcnn.model import McnnModel, McnnConfiguration, Model, MutatingCnnModel, NodeBuildConfiguration
from mcnn.samples import HorizontalDataset, Dataset


def create_mcnn(dataset: Dataset) -> Model:
    sample_length = 128
    pooling_factor = sample_length // 32
    local_filter_width = sample_length // 32
    # full_filter_width = pooling_factor // 4
    full_filter_width = pooling_factor
    cfg = McnnConfiguration(downsample_strides=[2, 4, 8, 16],
                            smoothing_window_sizes=[8, 16, 32, 64],
                            pooling_factor=pooling_factor,
                            channel_count=256,
                            local_filter_width=local_filter_width,
                            full_filter_width=full_filter_width,
                            layer_size=256,
                            full_pool_size=4)
    model = McnnModel(batch_size=64,
                      num_classes=dataset.target_classes_count,
                      learning_rate=1e-3,
                      sample_length=sample_length,
                      mcnn_configuration=cfg)
    return model


def create_mutating_cnn(dataset: HorizontalDataset, options) -> MutatingCnnModel:
    model = MutatingCnnModel(batch_size=options.batch_size,
                             num_classes=dataset.target_classes_count,
                             learning_rate=options.learning_rate,
                             sample_length=dataset.sample_length,
                             checkpoint_dir=options.checkpoint_dir,
                             global_avg_pool=options.global_avg_pool,
                             node_build_configuration=NodeBuildConfiguration.from_options(options))
    return model


def evaluate(options):
    logging.info('Evaluating with options {}'.format(options))
    eval_dataset = HorizontalDataset(options.dataset_train, options.dataset_test)
    eval_model = create_mutating_cnn(eval_dataset, options)
    operations.evaluate(eval_model, eval_dataset, options.checkpoint_dir, options.log_dir_test, feature_name='')


def visualize(options):
    logging.info('Running LRP visualization with options {}'.format(options))
    dataset = HorizontalDataset(options.dataset_train, options.dataset_test)
    model = create_mutating_cnn(dataset, options)
    heatmap_save_path = options.plot_dir / 'heatmap.pdf'
    operations.visualize_lrp(model, dataset, options.checkpoint_dir,
                             feature_name='', heatmap_save_path=heatmap_save_path)


def train(options):
    multiprocessing.set_start_method('spawn')
    logging.info('Training with options {}'.format(options))
    dataset = HorizontalDataset(options.dataset_train, options.dataset_test)

    def evaluate_process():
        proc = Process(target=evaluate, args=(options,))
        proc.start()

    model = create_mutating_cnn(dataset, options)
    operations.train_and_mutate(model,
                                dataset,
                                step_count=options.step_count,
                                checkpoint_dir=options.checkpoint_dir,
                                log_dir=options.log_dir_train,
                                steps_per_checkpoint=options.steps_per_checkpoint,
                                feature_name='',
                                checkpoint_written_callback=evaluate_process,
                                render_graph_steps=options.render_graph_steps,
                                train_only_switches_fraction=options.train_only_switches_fraction,
                                summary_every_step=options.summary_every_step)

