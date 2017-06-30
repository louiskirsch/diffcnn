import multiprocessing
from multiprocessing import Process

import logging
from pathlib import Path

import mcnn.operations as operations
from mcnn.model import McnnModel, McnnConfiguration, Model, MutatingCnnModel, NodeBuildConfiguration, \
    NodeMutationConfiguration, FCNModel
from mcnn.samples import HorizontalDataset, Dataset, AugmentedDataset


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


def create_mutating_cnn(dataset: Dataset, options) -> MutatingCnnModel:
    model = MutatingCnnModel(batch_size=options.batch_size,
                             num_classes=dataset.target_classes_count,
                             learning_rate=options.learning_rate,
                             sample_length=dataset.sample_length,
                             checkpoint_dir=options.checkpoint_dir,
                             penalty_factor=options.penalty_factor,
                             new_layer_penalty_multiplier=options.new_layer_penalty_multiplier,
                             global_avg_pool=options.global_avg_pool,
                             node_build_configuration=NodeBuildConfiguration.from_options(options),
                             node_mutate_configuration=NodeMutationConfiguration.from_options(options))
    return model


def create_fcn(dataset: Dataset, options) -> FCNModel:
    return FCNModel(sample_length=dataset.sample_length,
                    learning_rate=options.learning_rate,
                    num_classes=dataset.target_classes_count,
                    batch_size=options.batch_size)


def _write_results(dataset_name: str, write_result_file: Path, accuracy: float):
    should_add_header = not write_result_file.exists()
    with write_result_file.open('a') as f:
        if should_add_header:
            f.write('dataset_name, accuracy\n')
        f.write('{}, {}\n'.format(dataset_name, accuracy))


def _evaluate_with_result(options) -> float:
    logging.info('Evaluating with options {}'.format(options))
    eval_dataset = HorizontalDataset(options.dataset_train, options.dataset_test, options.z_normalize)
    if options.use_fcn_architecture:
        eval_model = create_fcn(eval_dataset, options)
    else:
        eval_model = create_mutating_cnn(eval_dataset, options)
    return operations.evaluate(eval_model, eval_dataset, options.checkpoint_dir, options.log_dir_test, feature_name='')


def evaluate(options):
    accuracy = _evaluate_with_result(options)
    if options.write_result_file is not None:
        _write_results(options.dataset_name, options.write_result_file, accuracy)


def visualize(options):
    logging.info('Running LRP visualization with options {}'.format(options))
    dataset = HorizontalDataset(options.dataset_train, options.dataset_test, options.z_normalize)
    if options.use_fcn_architecture:
        model = create_fcn(dataset, options)
    else:
        model = create_mutating_cnn(dataset, options)
    heatmap_save_path = options.plot_dir / 'heatmap.pdf'
    operations.visualize_lrp(model, dataset, options.checkpoint_dir,
                             feature_name='', heatmap_save_path=heatmap_save_path)


def train(options):
    multiprocessing.set_start_method('spawn')
    logging.info('Training with options {}'.format(options))
    dataset = HorizontalDataset(options.dataset_train, options.dataset_test, options.z_normalize)

    def evaluate_process():
        proc = Process(target=_evaluate_with_result, args=(options,))
        proc.start()

    if options.use_fcn_architecture:
        model = create_fcn(dataset, options)
        operations.train(model,
                         dataset,
                         step_count=options.step_count,
                         checkpoint_dir=options.checkpoint_dir,
                         log_dir=options.log_dir_train,
                         steps_per_checkpoint=options.steps_per_checkpoint,
                         feature_name='',
                         checkpoint_written_callback=evaluate_process,
                         save=True)
    else:
        model = create_mutating_cnn(dataset, options)
        operations.train_and_mutate(model,
                                    dataset,
                                    step_count=options.step_count,
                                    checkpoint_dir=options.checkpoint_dir,
                                    log_dir=options.log_dir_train,
                                    plot_dir=options.plot_dir,
                                    steps_per_checkpoint=options.steps_per_checkpoint,
                                    feature_name='',
                                    checkpoint_written_callback=evaluate_process,
                                    render_graph_steps=options.render_graph_steps,
                                    train_only_switches_fraction=options.train_only_switches_fraction,
                                    only_switches_lr=options.only_switches_learning_rate,
                                    summary_every_step=options.summary_every_step,
                                    freeze_on_delete=options.freeze_on_delete,
                                    delete_shrinking_last_node=options.delete_shrinking_last_node,
                                    checkpoints_after_frozen=options.checkpoints_after_frozen,
                                    freeze_on_shrinking_total_outputs=options.freeze_on_shrinking_total_outputs)

    if options.write_result_file is not None:
        accuracy = _evaluate_with_result(options)
        _write_results(options.dataset_name, options.write_result_file, accuracy)
