import os
import matplotlib.pyplot as plt

import torch
from torch.utils.data.dataset import ConcatDataset
from torch.utils.data.sampler import RandomSampler
import math
from torchvision import datasets, transforms


class CustomBatchSchedulerSampler(torch.utils.data.sampler.Sampler):
    """
    iterate over tasks and provide a random batch per task in each mini-batch
    """
    def __init__(self, dataset, batch_size, sub_batch_sizes):
        assert sum(sub_batch_sizes) == batch_size
        self.dataset = dataset
        self.batch_size = batch_size
        self.sub_batch_sizes = sub_batch_sizes
        self.number_of_datasets = len(dataset.datasets)
        self.dataset_sizes = [len(cur_dataset.samples) for cur_dataset in dataset.datasets]
        self.max_steps = self._get_max_steps()

    def _get_max_steps(self):
        max_steps = 0
        for i in range(self.number_of_datasets):
            cur_dataset_size = self.dataset_sizes[i]
            cur_sub_batch_sizes = self.sub_batch_sizes[i]
            cur_steps = math.ceil(cur_dataset_size / cur_sub_batch_sizes)
            if cur_steps > max_steps:
                max_steps = cur_steps
        return max_steps

    def __len__(self):
        return self.max_steps * self.batch_size

    def __iter__(self):
        samplers_list = []
        sampler_iterators = []
        for dataset_idx in range(self.number_of_datasets):
            cur_dataset = self.dataset.datasets[dataset_idx]
            sampler = RandomSampler(cur_dataset)
            samplers_list.append(sampler)
            cur_sampler_iterator = sampler.__iter__()
            sampler_iterators.append(cur_sampler_iterator)

        push_index_val = [0] + self.dataset.cumulative_sizes[:-1]
        step = self.batch_size * self.number_of_datasets
        samples_to_grab = self.sub_batch_sizes
        # for this case we want to get all samples in dataset, this force us to resample from the smaller datasets
        epoch_samples = self.max_steps

        final_samples_list = []  # this is a list of indexes from the combined dataset
        for _ in range(epoch_samples):
            for i in range(self.number_of_datasets):
                cur_batch_sampler = sampler_iterators[i]
                cur_samples = []
                for _ in range(samples_to_grab[i]):
                    try:
                        cur_sample_org = cur_batch_sampler.__next__()
                        cur_sample = cur_sample_org + push_index_val[i]
                        cur_samples.append(cur_sample)
                    except StopIteration:
                        # got to the end of iterator - restart the iterator and continue to get samples
                        # until reaching "epoch_samples"
                        sampler_iterators[i] = samplers_list[i].__iter__()
                        cur_batch_sampler = sampler_iterators[i]
                        cur_sample_org = cur_batch_sampler.__next__()
                        cur_sample = cur_sample_org + push_index_val[i]
                        cur_samples.append(cur_sample)
                final_samples_list.extend(cur_samples)

        return iter(final_samples_list)


if __name__ == "__main__":
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ])

    first_dataset = datasets.ImageFolder(
        os.path.join(
            os.path.dirname(os.path.dirname(os.getcwd())), "datasets", "retina", "test"
        ),
        transform
    )

    second_dataset = datasets.ImageFolder(
        os.path.join(
            os.path.dirname(os.path.dirname(os.getcwd())), "datasets", "breast_cancer", "test"
        ),
        transform
    )
    concat_dataset = ConcatDataset([first_dataset, second_dataset])

    batch_size = 5
    # dataloader with BatchSchedulerSampler
    dataloader = torch.utils.data.DataLoader(dataset=concat_dataset,
                                             sampler=BatchSchedulerSampler(dataset=concat_dataset,
                                                                           batch_size=batch_size,
                                                                           sub_batch_sizes=[2, 3]),
                                             batch_size=batch_size,
                                             shuffle=False,
                                             )

    for inputs, labels in dataloader:
        for input_im in inputs:
            plt.imshow(input_im.permute(1, 2, 0))
            plt.show()
        break
