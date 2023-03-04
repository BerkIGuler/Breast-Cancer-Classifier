from modules import Trainer, Dataset, TrainingArguments, Model

import torch.nn as nn
import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset',
                        type=str,
                        required=True)
    parser.add_argument('--log_path',
                        type=str,
                        required=True)
    parser.add_argument('--model', 
                        type=str, 
                        required=True)
    parser.add_argument('--num_steps',
                        type=int,
                        default=20)
    parser.add_argument('--feature_extract',
                        type=int,
                        default=0,
                        help="Flag for feature extracting. When False, we finetune the whole model, \
                              when True we only update the reshaped layer params")
    parser.add_argument('--num_classes',
                        type=int,
                        required=True)
    parser.add_argument('--batch_size',
                        type=int,
                        default=16)
    parser.add_argument('--gpu_id',
                        type=int,
                        default=0)
    parser.add_argument('--augment',
                        type=int,
                        default=0,
                        help="If 0 no augmentation, if 1 apply augmentation")
    parser.add_argument('--lr',
                        type=float,
                        default=0.001,
                        help="learning_rate")
    parser.add_argument('--num_workers',
                        type=int,
                        default=4,
                        help="number of workers used for dataloader")
    parser.add_argument('--patience',
                        type=int,
                        default=50,
                        help="patience for early stopping")
    parser.add_argument('--eval_freq',
                        type=int,
                        default=100,
                        help="evaluate model on test set every eval_freq iterations")
    _args = parser.parse_args()

    return _args


def main(args):
    model = Model(args)
    criterion = nn.CrossEntropyLoss()
    training_args = TrainingArguments(args, criterion, model)
    dataset = Dataset(args, training_args, model)
    test_dataloader = dataset.get_test_dataloader()
    trainer = Trainer(args, training_args, dataset, model)
    training_stats_database, best_acc = trainer.train()
    trainer.save_checkpoints(training_stats_database, best_acc, test_dataloader)


if __name__ == "__main__":
    cli_args = parse_args()
    main(cli_args)
