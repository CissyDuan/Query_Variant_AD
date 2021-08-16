
import os
import json
import sys
import pickle
import math
import numpy as np
import scipy
import scipy.sparse as sp
import torch
import torch.utils.data
import random
import copy
from tqdm import tqdm
from file import *
import torch_geometric
from  torch_geometric.utils import dense_to_sparse,to_dense_adj
PAD = 0
BOS = 1
EOS = 2
UNK = 3
def filt_sent_id(sent_id):
    filt_id = [0, 1, 3]
    result = []
    for id in sent_id:
        if id == 2:
            break
        elif id in filt_id:
            continue
        result.append(id)
    return result


class Dataset(torch.utils.data.Dataset):
    def __init__(self, config, vocab,data_path, train_type):
        self.train_type = train_type
        self.vocab = vocab
        self.config=config
        #self.graph, self.graph_weight= self.load_graph(config)
        self.graph= self.load_graph(config)


        self.data_path = data_path
        self.data_all=json.load(open(data_path, 'r'))
        self.number = len(self.data_all)


    def load_graph(self,config):
        #KG=json.load(open(config.KG))
        KG=json.load(open(config.KG_weight))
        print('get kg')

        return KG
    def __len__(self):
        return  self.number

    def __getitem__(self, index):
        data= self.data_all[index]

        ads_title, query_keywords, title_keywords, expose,click = data['adv'],data['query'],data['keyword'],data['expose'],data['click']
        query = [self.vocab.word2id(word) for word in data['query']]
        assert self.train_type in ['k','qk','qt']
        if self.train_type=='k':
            sub = SubGraph_k(query_keywords, title_keywords)
            sub.build_subgraph(self.graph)

            extend = ExtendGraph_k(sub)
            extend.extend(self.graph)
        elif self.train_type=='qk':
            sub = SubGraph(query_keywords, title_keywords)
            sub.build_subgraph(self.graph)

            extend = ExtendGraph(sub)
            extend.extend(self.graph)
        else :
            sub_ori = SubGraph(query_keywords, title_keywords)
            sub_ori.build_subgraph(self.graph)

            sub = SubGraph_full(query_keywords, ads_title)
            sub.build_subgraph(self.graph)

            extend = ExtendGraph_full(sub_ori, sub)
            extend.extend(self.graph)


        example = Example(sub, extend, ads_title, query,self.vocab,expose,click)
        example.to_tensor()

        return example

class Collate:
    def __init__(self, data_type):
        self.data_type=data_type

    def __call__(self, example_list):
        return Batch(example_list,self.data_type)

class Batch:
    """
    Each batch is a mini-batch of data
    """

    def __init__(self, example_list,data_type):
        assert data_type in ['train','eval']

        self.ads = [e.ads for e in example_list]
        self.mask = [e.mask for e in example_list]

        self.sub_node_idx= [e.sub_node_idx for e in example_list]
        self.extend_node_idx =  [e.extend_node_idx for e in example_list]
        self.neibour_node_idx=[e.neibour_node_idx for e in example_list]

        self.sub_adj =  [e.sub_adj for e in example_list]
        self.extend_adj = [e.extend_adj for e in example_list]
        self.sub_adj_weight = [e.sub_adj_weight for e in example_list]
        #self.extend_adj_weight=[e.extend_adj_weight for e in example_list]

        self.word_type =  [e.word_type for e in example_list]
        self.word_type_sub = [e.word_type_sub for e in example_list]

        self.expose =  [e.expose for e in example_list]
        self.click =  [e.click for e in example_list]
        self.query=[e.query for e in example_list]

        self.sub_neibour_mask=[e.sub_neibour_mask for e in example_list]


class Example:
    def __init__(self, subgraph, extend, ads,query, vocab,expose,click):
        self.ads,self.mask= self.padding(vocab.sent2id(ads))

        self.subgraph = subgraph
        self.extend = extend

        self.sub_node_num = len(self.subgraph.nodes)
        self.extend_node_num = len(self.extend.nodes)

        self.sub_adj = self.subgraph.sparse_to_dense()
        self.extend_adj = torch.from_numpy(np.array(self.extend.sparse_to_dense(),dtype=np.float32))

        self.sub_adj_weight=1

        self.sub_node_idx = self.subgraph.convert_to_idx(vocab)
        self.extend_node_idx = self.extend.convert_to_idx(vocab)
        self.neibour_node_idx=self.extend.convert_to_idx_nei(vocab)

        self.word_type_sub = self.subgraph.word_type
        self.word_type=self.extend.word_type_extend

        self.query=query
        self.expose=expose
        self.click=click

        #self.sub_neibour_mask=self.extend_adj[self.sub_node_num:, :self.sub_node_num].ne(0).float()
        self.sub_neibour_mask = self.extend_adj[self.sub_node_num:, :self.sub_node_num].float()

    def to_tensor(self):
        self.ads=torch.from_numpy(np.array(self.ads, dtype=np.long))

        self.mask = torch.from_numpy(np.array(self.mask, dtype=np.long))

        self.sub_node_idx = torch.from_numpy(np.array(self.sub_node_idx, dtype=np.long))
        self.extend_node_idx = torch.from_numpy(np.array(self.extend_node_idx, dtype=np.long))
        self.neibour_node_idx=torch.from_numpy(np.array(self.neibour_node_idx, dtype=np.long))

        self.sub_adj, self.sub_adj_weight= dense_to_sparse(torch.from_numpy(np.array(self.sub_adj, dtype=np.float32)))
        #self.extend_adj,self.extend_adj_weight=dense_to_sparse(torch.from_numpy(np.array(self.extend_adj, dtype=np.float32)))

        self.word_type_sub = torch.from_numpy(np.array(self.word_type_sub, dtype=np.long))
        self.word_type =torch.from_numpy(np.array(self.word_type, dtype=np.long))

        self.expose=torch.from_numpy(np.array(self.expose,dtype=np.float32))
        self.click=torch.from_numpy(np.array(self.click,dtype=np.float32))
        self.query = torch.from_numpy(np.array(self.query, dtype=np.long))

    def padding(self,tgt, max_len=22):
        if len(tgt)>22:
            print('tgt length error')
            print(tgt)
        mask = [1. for _ in range(len(tgt))]
        while len(tgt) < max_len:
            tgt.append(0)
            mask.append(0)
        return tgt,mask




class Vocab:
    def __init__(self, vocab_file, emb_size, use_pre_emb=False, vocab_size=50000):
        self.emb_size = emb_size
        '''self._word2id = {'[PADDING]': 0, '[START]': 1, '[END]': 2, '[OOV]': 3, '[COM]': 4}
        self._id2word = ['[PADDING]', '[START]', '[END]', '[OOV]', '[COM]']
        self._wordcount = {'[PADDING]': 1, '[START]': 1, '[END]': 1, '[OOV]': 1, '[COM]': 1}'''
        self._word2id = {'[PADDING]': 0, '[START]': 1, '[END]': 2, '[OOV]': 3}
        self._id2word = ['[PADDING]', '[START]', '[END]', '[OOV]']
        self._wordcount = {'[PADDING]': 1, '[START]': 1, '[END]': 1, '[OOV]': 1}

        if use_pre_emb:
            emb = self.load_glove()
            self.load_vocab(vocab_file, vocab_size=vocab_size, emb_size=emb_size, word_emb=emb)
            self.emb = np.array(self.emb, dtype=np.float32)
        else:
            self.emb = None
            self.load_vocab(vocab_file, emb_size=emb_size, vocab_size=vocab_size)
        self.voc_size = len(self._word2id)
        self.UNK_token = 3
        self.PAD_token = 0


    @staticmethod
    def load_glove(fname='glove.txt'):
        emb = {}
        for line in open(fname):
            tem = line.strip().split(' ')
            word = tem[0]
            vec = np.array([float(num) for num in tem[1:]], dtype=np.float32)
            emb[word] = vec
        return emb

    def load_vocab(self, vocab_file, emb_size, word_emb=None, vocab_size=None):
        if word_emb is not None:
            self.emb = [np.zeros(emb_size) for _ in range(5)]
        words = json.load(open(vocab_file))
        for word in words:
            if word_emb is not None and word in word_emb:
                self._word2id[word] = len(self._word2id)
                self._id2word.append(word)
                self.emb.append(word_emb[word])
            elif word_emb is None:
                self._word2id[word] = len(self._word2id)
                self._id2word.append(word)
            if vocab_size > 0 and len(self._word2id) >= vocab_size:
                break
        assert len(self._word2id) == len(self._id2word)

    def word2id(self, word):
        '''if word.upper() in self._company:
            return self._word2id['[COM]']'''
        if word in self._word2id:
            return self._word2id[word]
        return self._word2id['[OOV]']

    def sent2id(self, sent, add_start=True, add_end=True):
        result = [self.word2id(word) for word in sent]
        if add_start:
            result = [self._word2id['[START]']] + result

        if add_end:
            result = result + [self._word2id['[END]']]
        return result



    def id2word(self, word_id):
        return self._id2word[word_id]

    def id2sent(self, sent_id):
        result = []
        for id in sent_id:
            if id == self._word2id['[END]']:
                break
            elif id == self._word2id['[PADDING]'] or id==self._word2id['[OOV]']:
                continue
            result.append(self._id2word[id])
        return result

    def id2words(self, sent_id):
        result = []
        for id in sent_id:
            if id == self._word2id['[END]']:
                break
            elif id == self._word2id['[PADDING]'] or id==self._word2id['[OOV]']:
                continue
            result.append(self._id2word[id])
            result.append(',')
        return result

class DataLoader:

    def __init__(self, config, data_path, batch_size,vocab,data_type=None,train_type=None):
        self.data_type=data_type
        self.train_type=train_type
        self.batch_size=batch_size
        self.dataset = Dataset(config, vocab,data_path,train_type)
    def __call__(self):
        assert self.data_type in ['train','eval']
        if self.data_type=='train':
            dataloader=torch.utils.data.DataLoader(dataset=self.dataset, batch_size=self.batch_size, pin_memory=True,
                                        collate_fn=Collate(self.data_type), shuffle=True)
        else:
            dataloader = torch.utils.data.DataLoader(dataset=self.dataset, batch_size=self.batch_size, pin_memory=True,
                                                     collate_fn=Collate(self.data_type), shuffle=False)

        return dataloader



