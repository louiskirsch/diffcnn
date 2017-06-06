import multiprocessing
from pathlib import Path
import logging
from multiprocessing import Process

import mcnn.operations as operations
from mcnn.model import McnnModel, McnnConfiguration, Model, AutoCnnModel, MutatingCnnModel
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


def create_auto_cnn(dataset: HorizontalDataset) -> Model:
    model = AutoCnnModel(batch_size=64,
                         num_classes=dataset.target_classes_count,
                         learning_rate=1e-3,
                         sample_length=dataset.sample_length,
                         filter_width=4)
    return model


def create_mutating_cnn(dataset: HorizontalDataset, checkpoint_dir: Path, sample_length: int) -> MutatingCnnModel:
    model = MutatingCnnModel(batch_size=64,
                             num_classes=dataset.target_classes_count,
                             learning_rate=1e-4,
                             sample_length=sample_length,
                             checkpoint_dir=checkpoint_dir)
    return model


def train(model: Model, dataset: Dataset, checkpoint_dir: Path, log_dir: Path):
    operations.train(model,
                     dataset,
                     step_count=500,
                     checkpoint_dir=checkpoint_dir,
                     log_dir=log_dir,
                     steps_per_checkpoint=100,
                     feature_name='')


def visualize(model: Model, dataset: Dataset, checkpoint_dir: Path):
    operations.deconv(model,
                      dataset,
                      sample_count=1000,
                      checkpoint_dir=checkpoint_dir,
                      feature_name='')


def evaluate(dataset_train: Path, dataset_test: Path, checkpoint_dir: Path, dataset_name: str):
    eval_dataset = HorizontalDataset(dataset_train, dataset_test)
    eval_model = create_mutating_cnn(eval_dataset, checkpoint_dir, eval_dataset.sample_length)
    log_dir_test = Path('logs') / (dataset_name + '_test')
    operations.evaluate(eval_model, eval_dataset, checkpoint_dir, log_dir_test, feature_name='')

def main():
    multiprocessing.set_start_method('spawn')
    logging.basicConfig(level=logging.INFO)
    root = Path.home() / 'data' / 'UCR_TS_Archive_2015'

    # for dataset_path in root.iterdir():
    for dataset_path in [root / 'Adiac']:
        dataset_name = dataset_path.name
        checkpoint_dir = Path('checkpoints') / dataset_name
        log_dir_train = Path('logs') / (dataset_name + '_train')

        dataset_train = root / dataset_name / (dataset_name + '_TRAIN')
        dataset_test = root / dataset_name / (dataset_name + '_TEST')
        dataset = HorizontalDataset(dataset_train, dataset_test)

        train_sample_length = int(0.9 * dataset.sample_length)
        model = create_mutating_cnn(dataset, checkpoint_dir, train_sample_length)

        def evaluate_process():
            proc = Process(target=evaluate, args=(dataset_train, dataset_test, checkpoint_dir, dataset_name))
            proc.start()

        operations.train_and_mutate(model,
                                    dataset,
                                    step_count=10000,
                                    checkpoint_dir=checkpoint_dir,
                                    log_dir=log_dir_train,
                                    steps_per_checkpoint=1000,
                                    feature_name='',
                                    checkpoint_written_callback=evaluate_process)


if __name__ == '__main__':
    main()
