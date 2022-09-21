'''
Modified based on https://github.com/wengong-jin/icml18-jtnn.git.
'''

import torch
from torch.utils.data import Dataset, DataLoader
from .mod_tree import ModTree
from .jtnn_enc import JTNNEncoder
import os, random

class PairTreeFolder(object):

    def __init__(self, data_folder, vocab, batch_size, num_workers=4, shuffle=True, y_assm=True, replicate=None):
        self.data_folder = data_folder
        self.data_files = [fn for fn in os.listdir(data_folder)]
        self.batch_size = batch_size
        self.vocab = vocab
        self.num_workers = num_workers
        self.y_assm = y_assm
        self.shuffle = shuffle

        if replicate is not None: #expand is int
            self.data_files = self.data_files * replicate

    def __iter__(self):
        for fn in self.data_files:
            fn = os.path.join(self.data_folder, fn)
            with open(fn) as f:
                data = pickle.load(f)

            if self.shuffle: 
                random.shuffle(data) #shuffle data before batch

            batches = [data[i : i + self.batch_size] for i in xrange(0, len(data), self.batch_size)]
            if len(batches[-1]) < self.batch_size:
                batches.pop()

            dataset = PairTreeDataset(batches, self.vocab, self.y_assm)
            dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=self.num_workers, collate_fn=lambda x:x[0])

            for b in dataloader:
                yield b

            del data, batches, dataset, dataloader

class MolTreeFolder(object):

    def __init__(self, data_folder, vocab, batch_size, num_workers=4, shuffle=True, assm=True, replicate=None):
        self.data_folder = data_folder
        self.data_files = [fn for fn in os.listdir(data_folder)]
        self.batch_size = batch_size
        self.vocab = vocab
        self.num_workers = num_workers
        self.shuffle = shuffle
        self.assm = assm

        if replicate is not None: #expand is int
            self.data_files = self.data_files * replicate

    def __iter__(self):
        for fn in self.data_files:
            fn = os.path.join(self.data_folder, fn)
            with open(fn) as f:
                data = pickle.load(f)

            if self.shuffle: 
                random.shuffle(data) #shuffle data before batch

            batches = [data[i : i + self.batch_size] for i in xrange(0, len(data), self.batch_size)]
            if len(batches[-1]) < self.batch_size:
                batches.pop()

            dataset = MolTreeDataset(batches, self.vocab, self.assm)
            dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=self.num_workers, collate_fn=lambda x:x[0])

            for b in dataloader:
                yield b

            del data, batches, dataset, dataloader

class PairTreeDataset(Dataset):

    def __init__(self, data, vocab, y_assm):
        self.data = data
        self.vocab = vocab
        self.y_assm = y_assm

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        batch0, batch1 = zip(*self.data[idx])
        return tensorize(batch0, self.vocab, assm=False), tensorize(batch1, self.vocab, assm=self.y_assm)

class MolTreeDataset(Dataset):

    def __init__(self, data, vocab, assm=True):
        self.data = data
        self.vocab = vocab
        self.assm = assm

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return tensorize(self.data[idx], self.vocab, assm=self.assm)

# takes in sampled graph and output tensors
def tensorize(attr, conn):
    tree_batch = []
    for idx in range(len(attr)):
        mod_tree = ModTree(attr[idx], conn[idx])
        tree_batch.append(mod_tree)

    set_batch_nodeID(tree_batch)
    jtenc_holder, mess_dict = JTNNEncoder.tensorize(tree_batch)
    jtenc_holder = jtenc_holder

    return tree_batch, jtenc_holder


def set_batch_nodeID(mod_batch):
    tot = 0
    for mod_tree in mod_batch:
        for node in mod_tree.nodes:
            node.idx = tot
            node.wid = node.type
            tot += 1
