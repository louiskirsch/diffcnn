import multiprocessing
from multiprocessing import Process

import logging
from pathlib import Path

import mcnn.operations as operations
from mcnn.model import McnnModel, McnnConfiguration, Model, MutatingCnnModel, NodeBuildConfiguration, \
    NodeMutationConfiguration, FCNModel, ConvNodeCreateConfiguration
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
                             architecture_dir=options.architecture_dir,
                             penalty_factor=options.penalty_factor,
                             new_layer_penalty_multiplier=options.new_layer_penalty_multiplier,
                             initial_depth=options.initial_depth,
                             use_fully_connected=options.use_fully_connected,
                             node_build_configuration=NodeBuildConfiguration.from_options(options),
                             node_mutate_configuration=NodeMutationConfiguration.from_options(options),
                             conv_node_create_configuration=ConvNodeCreateConfiguration.from_options(options))
    model.architecture_frozen = options.freeze
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
    return operations.evaluate(eval_model, eval_dataset, options.checkpoint_dir, options.log_dir_test)


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
    operations.visualize_lrp(model, dataset, options.checkpoint_dir, heatmap_save_path=heatmap_save_path)


def train(options):
    multiprocessing.set_start_method('spawn')
    logging.info('Training with options {}'.format(options))
    dataset = HorizontalDataset(options.dataset_train, options.dataset_test, options.z_normalize)

    def evaluate_process():
        proc = Process(target=_evaluate_with_result, args=(options,))
        proc.start()

    if options.use_fcn_architecture:
        model = create_fcn(dataset, options)
        result = operations.train(model,
                                  dataset,
                                  epoch_count=options.epoch_count,
                                  checkpoint_dir=options.checkpoint_dir,
                                  log_dir_train=options.log_dir_train,
                                  log_dir_test=options.log_dir_test,
                                  steps_per_checkpoint=options.steps_per_checkpoint,
                                  steps_per_summary=options.steps_per_summary,
                                  checkpoint_written_callback=None,
                                  save=True)
    else:
        model = create_mutating_cnn(dataset, options)
        trainer = operations.MutationTrainer(model,
                                             dataset,
                                             checkpoint_dir=options.checkpoint_dir,
                                             log_dir_train=options.log_dir_train,
                                             log_dir_test=options.log_dir_test,
                                             plot_dir=options.plot_dir,
                                             steps_per_checkpoint=options.steps_per_checkpoint,
                                             steps_per_summary=options.steps_per_summary,
                                             train_only_switches_fraction=options.train_only_switches_fraction,
                                             only_switches_lr=options.only_switches_learning_rate,
                                             freeze_on_delete=options.freeze_on_delete,
                                             delete_shrinking_last_node=options.delete_shrinking_last_node,
                                             epochs_after_frozen=options.epochs_after_frozen,
                                             freeze_on_shrinking_total_outputs=options.freeze_on_shrinking_total_outputs,
                                             stagnant_abort_steps=options.stagnant_abort_steps)
        result = trainer.train(options.epoch_count)

    print('Test accuracy {} with minimum train loss'.format(result.best_test_accuracy))

    if options.write_result_file is not None:
        _write_results(options.dataset_name, options.write_result_file, result.best_test_accuracy)
