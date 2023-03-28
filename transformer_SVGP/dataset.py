import random

import torch
import torch.utils.data as data
import json
import warnings
import numpy as np


def get_even_sampling(data, sample_data_size, split_reward=0.6):
    """
    evenly random sample from the data
    @param data:
    @param sample_data_size:
    @param split_reward:
    @return:
    """
    if len(data) == 0:
        return []
    data_low_idx, data_high_idx, sample_data = [], [], []
    for idx, data_item in enumerate(data):
        data_low_idx.append(idx) if data_item['reward'] < split_reward else data_high_idx.append(idx)
    low_length_ratio = len(data_low_idx) / len(data)
    sample_data.extend([data[idx] for idx in random.sample(data_low_idx, int(sample_data_size * low_length_ratio))])
    high_length_ratio = len(data_high_idx) / len(data)
    sample_data.extend([data[idx] for idx in random.sample(data_high_idx, int(sample_data_size * high_length_ratio))])
    return sample_data


# SEQ_LEN = None
class Dataset(data.Dataset):

    def __init__(self, data_file_name, vocab, ground_truth='simulation', max_seq_len=64, label_len=5):
        """Set the path for images, captions and vocabulary wrapper.
        
        Args:
            root: image directory.
            data: index file name.
            vocab: pre-processed vocabulary.
            label_len: the maximum length of path
        """
        # self.root = root
        if data_file_name:
            with open(data_file_name, 'r') as f:
                self.data = json.load(f)
        else:
            self.data = []
            warnings.warn('Warning Message: not data file, we init [] for .data!')

        self.ids = range(len(self.data))
        self.vocab = vocab
        self.ground_truth = ground_truth
        self.seq_len = max_seq_len
        self.label_len = label_len
        self.data_keys = []

    def __getitem__(self, index):
        """Returns one data pair (image and concatenated captions)."""
        data = self.data
        vocab = self.vocab
        id = self.ids[index]

        paths = data[id]['paths']

        if len(paths) >= self.seq_len:
            paths = paths[:self.seq_len]

        embedded_paths = []
        for path in paths:
            comps = path.split(' - ')
            embedded_path = [vocab(comp) for comp in comps]
            if len(embedded_path) < self.label_len:
                # make sure all paths have the same length
                # 0 is hardcoded to be <PAD>
                embedded_path += [0] * (self.label_len - len(embedded_path))

            embedded_paths.append(embedded_path)

        path_list = torch.Tensor(embedded_paths)

        if self.ground_truth == 'simulation':
            eff = torch.Tensor([data[id]['eff']]).float()
            vout = torch.Tensor([data[id]['vout']]).float()
        elif self.ground_truth == 'analytic':
            eff = torch.Tensor([data[id]['eff_analytic']]).float()
            vout = torch.Tensor([data[id]['vout_analytic']]).float()
        else:
            raise Exception(f"Unknown ground truth {self.ground_truth}")

        duty = torch.Tensor([data[id]['duty']]).float()
        valid = torch.Tensor([data[id]['valid']]).float()

        return path_list, eff, vout, duty, valid

    def __len__(self):
        return len(self.ids)

    def append_data(self, path_set, duties, effs, vouts, valids, rewards):
        for paths, duty, eff, vout, valid, reward in zip(path_set, duties, effs, vouts, valids, rewards):
            self.data.append({'paths': paths, 'duty': duty, 'eff': eff, 'vout': vout, 'valid': valid, 'reward': reward})
        self.ids = range(len(self.data))

    def merge_data(self, another_dataset):
        assert (isinstance(another_dataset, Dataset))
        self.data.extend(another_dataset.data)
        self.ids = range(len(self.data))

    def merge_data_with_data_idx(self, _data, idx):
        self.data.extend(_data)
        self.ids = range(len(self.data))

    # def random_sample_data(self, ratio=0.1):
    #     self.data = random.sample(self.data, int(ratio * len(self.data)))
    #     self.ids = range(len(self.data))

    def random_sample_data(self, ratio):
        return get_even_sampling(data=self.data, sample_data_size=int(ratio * len(self.data)), split_reward=0.6), \
               int(ratio * len(self.data))
        # return random.sample(self.data, int(ratio * len(self.data))), int(ratio * len(self.data))


def getRawData(data_path, vocab, max_seq_len):
    with open(data_path, 'r') as f:
        data = json.load(f)

    data_x = []

    for item in data:
        paths = item['paths']
        if len(paths) >= max_seq_len:
            paths = paths[:max_seq_len]
        data_x.append([vocab(x) for x in paths])

    data_x = torch.Tensor(data_x)

    return data_x[:500]


def collate_fn(data):
    """Creates mini-batch tensors from the list of tuples (image, caption).

    Args:
        data: list of tuple (image, caption). 
            - image: torch tensor of shape
            - caption: torch tensor of shape (?); variable length.

    Returns:
        images: torch tensor of images.
        targets: torch tensor of shape (batch_size, padded_length).
        lengths: list; valid length for each padded caption.
    """
    # Sort a data list by caption length (descending order).
    # print(data)
    path_list, effs, vouts, duties, valids = zip(*data)

    # Merge images (from tuple of 3D tensor to 4D tensor).
    # path_list = torch.stack(path_list, 0)
    # image1 = torch.stack(image1, 0)

    effs = torch.stack(effs, 0)
    vouts = torch.stack(vouts, 0)
    duties = torch.stack(duties, 0)
    valids = torch.stack(valids, 0)
    # Merge captions (from tuple of 1D tensor to 2D tensor).
    lengths = [len(cap) for cap in path_list]
    max_path_len = len(path_list[0][0])  # get the length of a path, they are all uniform

    path_tensor = torch.zeros(len(path_list), max(lengths), max_path_len).long()
    # captions_tgt = torch.zeros(len(captions), max(lengths)).long()
    for i, cap in enumerate(path_list):
        end = lengths[i]
        path_tensor[i, :end] = cap[:end]
        # captions_tgt[i, :end-1] = cap[1:end]
    padding_mask = (path_tensor != 0)

    # assert padding_mask.all()
    # assert (path_tensor == path_list[0].long()).all()
    return path_tensor, effs, vouts, duties, valids, padding_mask


# def collate_fn_test(data):
#     """Creates mini-batch tensors from the list of tuples (image, caption).
#     Args:
#         data: list of tuple (image, caption). 
#             - image: torch tensor of shape
#             - caption: torch tensor of shape (?); variable length.
#     Returns:
#         images: torch tensor of images.
#         targets: torch tensor of shape (batch_size, padded_length).
#         lengths: list; valid length for each padded caption.
#     """
#     # Sort a data list by caption length (descending order).
#     image0, image1, _, image0_label, image1_label = zip(*data)
#     # Merge images (from tuple of 3D tensor to 4D tensor).
#     image0 = torch.stack(image0, 0)
#     image1 = torch.stack(image1, 0)

#     image0_label = torch.stack(image0_label, 0)
#     image1_label = torch.stack(image1_label, 0)
#     # # Merge captions (from tuple of 1D tensor to 2D tensor).
#     # lengths = [len(cap) for cap in captions]
#     # captions_src = torch.zeros(len(captions), max(lengths)).long()
#     # captions_tgt = torch.zeros(len(captions), max(lengths)).long()
#     # for i, cap in enumerate(captions):
#     #     end = lengths[i]
#     #     captions_src[i, :end-1] = cap[:end-1]
#     #     captions_tgt[i, :end-1] = cap[1:end]
#     # # caption_padding_mask = (captions_src != 0)
#     # return target_images, candidate_images, captions_src, captions_tgt
#     return image0, image1, image0_label, image1_label


def get_loader(data, vocab, batch_size, shuffle, ground_truth='simulation', num_workers=1, max_seq_len=64,
               attribute_len=5):
    """Returns torch.utils.data.DataLoader for custom dataset."""

    # relative caption dataset
    if type(data) == str:
        print('Reading data from', data)
        dataset = Dataset(
            data_file_name=data,
            vocab=vocab,
            max_seq_len=max_seq_len,
            label_len=attribute_len,
            ground_truth=ground_truth,
        )
    else:
        # use preloaded data
        dataset = data

    print('data size', len(dataset))
    # Data loader for the dataset
    # This will return (images, captions, lengths) for each iteration.
    # images: a tensor of shape (batch_size, 3, 224, 224).
    # captions: a tensor of shape (batch_size, padded_length).
    # lengths: a list indicating valid length for each caption. length is (batch_size)
    data_loader = torch.utils.data.DataLoader(dataset=dataset,
                                              batch_size=batch_size,
                                              shuffle=shuffle,
                                              num_workers=num_workers,
                                              collate_fn=collate_fn)

    return data_loader


def load_ori_token_data(data_file_name):
    test_data_captions = []
    with open(data_file_name, 'r') as f:
        data = json.load(f)

        for line in data:
            caption_texts = line['captions']
            temp = []
            for c in caption_texts:
                # tokens = nltk.tokenize.word_tokenize(str(c).lower())
                temp.append(c)
            test_data_captions.append(temp)

    return test_data_captions


def load_ori_token_data_new(data_file_name):
    test_data_captions = {}
    with open(data_file_name, 'r') as f:
        data = json.load(f)
        count = 0
        for line in data:
            caption_texts = line['captions']
            caption_texts = ["it " + x for x in caption_texts]
            # temp = []
            # for c in caption_texts:
            #     # tokens = nltk.tokenize.word_tokenize(str(c).lower())
            #     temp.append(c)
            test_data_captions[count] = caption_texts
            count += 1

    return test_data_captions
