import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
import models
from  torch_geometric.utils import dense_to_sparse,to_dense_adj

import torch_geometric
from torch_geometric.data import Batch,Data


def word_type_trans(word_type):
    mask1=word_type.eq(1)
    mask2 = word_type.eq(2)
    mask3 = word_type.eq(3)
    mask_nei = (word_type.eq(4).long())*2
    mask_ori=mask1+mask2+mask3
    mask_ori=mask_ori.ne(0).long()
    word_type=mask_ori+mask_nei
    return word_type

def pad_martix(martix_list):

    length=[martix.size(0) for martix in martix_list]
    wide=[martix.size(1) for martix in martix_list]

    max_length=max(length)
    #print(max_length)
    max_wide=max(wide)
    #print(max_wide)
    martix_pad=[]
    for martix,l,w in zip(martix_list,length,wide):
        pad_l=max_length-l
        #print(pad_l)
        pad_w=max_wide-w
        pad=nn.ZeroPad2d((0,pad_w,0,pad_l))
        martix=pad(martix)
        martix_pad.append(martix)
    #print(martix_pad)

    martix_pad=torch.stack(martix_pad,0)
    return martix_pad


def mask_trans(mask):
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask
class graph2seq_rl(nn.Module):
    def __init__(self, config, vocab,use_cuda,emb,pretrain=None):
        super(graph2seq_rl, self).__init__()
        self.vocab=vocab
        self.vocab_size=vocab.voc_size
        #self.occupy=nn.Linear(1,1)
        if pretrain is not None:
            self.embedding = pretrain['emb']            
        elif emb==0:
            self.embedding = nn.Embedding(self.vocab_size, config.emb_size)
        else:
            self.embedding=emb
            for p in self.parameters():
                p.requires_grad = False

        self.word_type_emb = nn.Embedding(config.type_num_sam, config.emb_size)
        self.encoder = models.Graph_encoder_sample_weight(config, embedding=self.embedding,
                                                          word_type_emb=self.word_type_emb)

        '''assert config.graph_model in ['gcn-uni','gat-lstm','gcn-lstm','gcn-lstm-n','gcn-gat-lstm']

        if config.graph_model == 'gcn-uni':
            self.encoder = models.Graph_encoder_sample_weight_uni(config, embedding=self.embedding,
                                                           word_type_emb=self.word_type_emb)
        elif config.graph_model == 'gat-lstm':
            self.encoder = models.Graph_encoder_sample(config, embedding=self.embedding,
                                                word_type_emb=self.word_type_emb)
        elif config.graph_model == 'gcn-lstm':
            self.encoder = models.Graph_encoder_sample_weight(config, embedding=self.embedding,
                                                       word_type_emb=self.word_type_emb)
        elif config.graph_model == 'gcn-lstm-n':
            self.encoder = models.Graph_encoder_sample_weight_n(config, embedding=self.embedding,
                                                         word_type_emb=self.word_type_emb)
        elif config.graph_model == 'gcn-gat-lstm':
            self.encoder = models.Graph_encoder_sample_combine(config, embedding=self.embedding,
                                                               word_type_emb=self.word_type_emb)'''

        self.rl_sample=models.rl_sample(config)
        self.use_cuda=use_cuda
        self.config=config




    def get_sample_neibour(self,sub_x,sub_graph,extend_node_idx,neibour_node_idx,extend_adj,word_type_extend,sub_neibour_mask,nei_pad_mask,nei_num,sub_num,sample_type=None):

        neibour_node_emb = self.embedding(neibour_node_idx)
        sample_ids, sample_prob_batch = self.rl_sample(sub_x, sub_graph, neibour_node_emb, sub_neibour_mask,nei_pad_mask,nei_num,sub_num,sample_type=sample_type)

        sub_num=sub_num.long()
        sample_ids=sample_ids.unbind(0)
        sample_adj_batch=[]
        sample_adj_weight_batch=[]
        sample_node_idx_batch=[]
        word_type_batch=[]
        nei_num=nei_num.squeeze(-1).unbind(0)
        sub_num=sub_num.squeeze(-1).unbind(0)

        #print(sample_ids)
        #print(nei_num)


        for sub_node_num,sample_id,extend_adj,nei_num,extend_node_idx,word_type_extend in zip(sub_num,sample_ids,extend_adj,nei_num,extend_node_idx,word_type_extend):
            if nei_num<self.config.sample_num:
                sample_id=sample_id[:nei_num]
            #print(sub_node_num)
            #print(sample_id)
            #print(extend_adj)
            #print(nei_num)
            sample_adj,sample_adj_weight=self.get_sample_adj(extend_adj, sub_node_num, sample_id)
            sub_id = torch.tensor(list(range(sub_node_num)),dtype=torch.long).cuda()
            #print(sub_id)
            id =torch.cat((sub_id , sample_id),0)
            #print(extend_node_idx)
            sample_node_idx = extend_node_idx[id]
            word_type=word_type_extend[id]
            sample_node_idx_batch.append(sample_node_idx)
            sample_adj_batch.append(sample_adj)
            sample_adj_weight_batch.append(sample_adj_weight)
            word_type_batch.append(word_type)

        return sample_node_idx_batch, sample_adj_batch, sample_adj_weight_batch,sample_prob_batch,word_type_batch


    def get_sample_adj(self,extend_adj, sub_node_num, sample_id):
        sample1 = extend_adj[sample_id, :]
        #print(sub_node_num)
        sample_adj_mid = torch.cat((extend_adj[:sub_node_num, :], sample1), 0)
        sample2 = sample_adj_mid[:, sample_id]
        sample_adj = torch.cat((sample_adj_mid[:, :sub_node_num], sample2), 1)
        sample_adj,sample_adj_weight=dense_to_sparse(sample_adj)

        return sample_adj,sample_adj_weight


    def get_match_reward(self,x,state,e):
        word_type=e.word_type
        if self.use_cuda:
            x=x.cuda()
            state=state.cuda()
            word_type=word_type.cuda()
        x=torch.squeeze(x,0)
        state=torch.squeeze(state[0][-1],0)
        #print(state)
        #print(x)
        query_num=torch.ne((torch.eq(word_type,1)+torch.eq(word_type,2)),0).sum()
        query=x[:query_num]
        query=F.softmax(query,-1)
        state=F.softmax(state,-1)
        match_reward=torch.mul(state,query).sum()
        match_reward=match_reward/query_num

        return match_reward




    def forward(self, batch,train_type=None,sample=True,sample_type=None):
        sub_node_idx, extend_node_idx,neibour_node_idx, sub_adj, sub_adj_weight, extend_adj, word_type, tgt, sub_neibour_mask, word_type_sub= \
            batch.sub_node_idx, batch.extend_node_idx, batch.neibour_node_idx,batch.sub_adj, batch.sub_adj_weight, batch.extend_adj, batch.word_type, batch.ads, batch.sub_neibour_mask, batch.word_type_sub
        query = batch.query
        #extend_adj_weight=batch.extend_adj_weight
        if self.use_cuda == True:
            sub_node_idx = [s.cuda() for s in sub_node_idx]
            extend_node_idx = [e.cuda() for e in extend_node_idx]
            neibour_node_idx = [n.cuda() for n in neibour_node_idx]
            sub_adj = [s.cuda() for s in sub_adj]
            sub_adj_weight = [s.cuda() for s in sub_adj_weight]
            extend_adj = [e.cuda() for e in extend_adj]
            word_type = [w.cuda() for w in word_type]
            word_type_sub = [w.cuda() for w in word_type_sub]
            tgt = [t.cuda() for t in tgt]
            sub_neibour_mask = [s.cuda() for s in sub_neibour_mask]
            # print(query)
            query = [q.cuda() for q in query]
            #extend_adj_weight=[e.cuda() for e in extend_adj_weight]

        # =====分解=====
        sub_node_idx_no = []
        sub_adj_no=[]
        sub_adj_weight_no=[]
        extend_node_idx_no = []
        extend_adj_no = []
        extend_adj_weight_no = []
        word_type_no = []
        word_type_sub_no=[]
        tgt_no = []
        query_no = []
        index_no = []

        sub_node_idx_s = []
        extend_node_idx_s = []
        neibour_node_idx_s = []
        sub_adj_s = []
        sub_adj_weight_s = []
        extend_adj_s = []
        word_type_s = []
        word_type_sub_s = []
        sub_neibour_mask_s = []
        tgt_s = []
        query_s = []
        index_s = []

        for i in range(len(sub_node_idx)):
            if neibour_node_idx[i].size(0) < self.config.sample_num:
                sub_node_idx_no.append(sub_node_idx[i])
                sub_adj_no.append(sub_adj[i])
                sub_adj_weight_no.append(sub_adj_weight[i])
                extend_node_idx_no.append(extend_node_idx[i])
                extend_adj_i, extend_adj_weight_i = dense_to_sparse(extend_adj[i])
                extend_adj_no.append(extend_adj_i)
                extend_adj_weight_no.append(extend_adj_weight_i)
                word_type_no.append(word_type[i])
                word_type_sub_no.append(word_type_sub[i])
                tgt_no.append(tgt[i])
                query_no.append(query[i])
                index_no.append(i)
            else:
                sub_node_idx_s.append(sub_node_idx[i])
                extend_node_idx_s.append(extend_node_idx[i])
                neibour_node_idx_s.append(neibour_node_idx[i])
                sub_adj_s.append(sub_adj[i])
                sub_adj_weight_s.append(sub_adj_weight[i])
                extend_adj_s.append(extend_adj[i])
                word_type_s.append(word_type[i])
                word_type_sub_s.append(word_type_sub[i])
                sub_neibour_mask_s.append(sub_neibour_mask[i])
                tgt_s.append(tgt[i])
                query_s.append(query[i])
                index_s.append(i)

        if sample==True:
            #==采样======
            data_batch = []
            sub_node_idx_s = pad_sequence(sub_node_idx_s, batch_first=True)
            word_type_sub_s = pad_sequence(word_type_sub_s, batch_first=True)
            mask_pad = sub_node_idx_s.ne(0).float().unsqueeze(-1)
            sub_num=mask_pad.sum(1)
            mask_score = mask_trans(sub_node_idx_s.ne(0).float())
            sub_node_idx_s = list(sub_node_idx_s)
            word_type_sub_s = list(word_type_sub_s)

            for x, a, aw,w in zip(sub_node_idx_s, sub_adj_s,sub_adj_weight_s, word_type_sub_s):
                data = Data(x=x, edge_index=a,edge_attr=aw, word_type=w[:x.size(0)])
                data_batch.append(data)

            data_batch = Batch.from_data_list(data_batch, follow_batch=[])
            sub_x, sub_graph = self.encoder(data_batch,mask_pad,mask_score)
            #extend_node_idx=pad_sequence(extend_node_idx,batch_first=True)
            neibour_node_idx_s=pad_sequence(neibour_node_idx_s,batch_first=True)
            nei_pad_mask=neibour_node_idx_s.ne(0)
            nei_num=nei_pad_mask.sum(1).unsqueeze(-1)
            nei_pad_mask=mask_trans(nei_pad_mask)
            #print(nei_pad_mask)
            sub_neibour_mask_s=pad_martix(sub_neibour_mask_s)

            sample_node_idx_batch, sample_adj_batch,sample_adj_weight_batch, sample_prob_batch, word_type_batch = self.get_sample_neibour(sub_x, sub_graph,
                                                                                          extend_node_idx_s,neibour_node_idx_s,
                                                                                          extend_adj_s, word_type_s,
                                                                                          sub_neibour_mask_s,nei_pad_mask,nei_num,sub_num,sample_type=sample_type)

            prob_avg = torch.mean(sample_prob_batch, 0, keepdim=False).data
            # print(prob_avg)
            prob_no = torch.ones(len(tgt_no)).fill_(prob_avg.detach())
            prob_no = prob_no.cuda()
            # print(sample_prob_batch)

            # ===合并====
            #if train_type == 'sample_sup' or train_type == 'generate':

            sample_node_idx_batch = sample_node_idx_batch + extend_node_idx_no
            sample_adj_batch = sample_adj_batch + extend_adj_no
            sample_adj_weight_batch = sample_adj_weight_batch + extend_adj_weight_no
            sample_prob_batch = torch.cat((sample_prob_batch, prob_no), 0)

            word_type_batch = word_type_batch + word_type_no
            sub_node_idx = sub_node_idx_s + sub_node_idx_no


        else:

            word_type_batch=word_type_sub_s+word_type_sub_no
            sample_node_idx_batch =sub_node_idx_s+sub_node_idx_no
            sample_adj_batch=sub_adj_s+sub_adj_no
            sample_adj_weight_batch=sub_adj_weight_s+sub_adj_weight_no
            sample_prob_batch=torch.ones(len(sample_node_idx_batch)).fill_(1).cuda()
            sub_node_idx=sub_node_idx_s + sub_node_idx_no


        tgt = tgt_s + tgt_no
        query = query_s + query_no
        index = index_s + index_no

        # ===排序====
        sample_node_idx_batch = sort_list(sample_node_idx_batch, index)
        sample_adj_batch = sort_list(sample_adj_batch, index)
        sample_adj_weight_batch = sort_list(sample_adj_weight_batch, index)
        sample_prob_batch = sample_prob_batch.index_select(dim=0, index=torch.tensor(index).cuda())
        tgt = sort_list(tgt, index)
        word_type_batch = sort_list(word_type_batch, index)
        sub_node_idx = sort_list(sub_node_idx, index)
        query = sort_list(query, index)


        tgt = torch.stack(tgt, dim=0)
        #print(sample_prob_batch)
        #word_type_batch = word_type
        word_type_batch = pad_sequence(word_type_batch, batch_first=True)
        word_type_batch = word_type_trans(word_type_batch)
        word_type_batch = word_type_batch.unbind(0)
        '''sample_node_idx_batch=extend_node_idx
        sample_adj_batch=extend_adj
        sample_adj_weight_batch=extend_adj_weight
        sample_prob_batch=torch.tensor([1])'''

        return sample_node_idx_batch, sample_adj_batch, sample_adj_weight_batch,sample_prob_batch, tgt, word_type_batch,query,sub_node_idx
def get_key(item):
    return item[1]
def sort_list(list,index):
    list_sort=[]
    for l,i in zip(list,index):
        list_sort.append((l,i))
    list_sort.sort(key=get_key)
    list_out=[]
    for item in list_sort:
        list_out.append(item[0])
    return list_out