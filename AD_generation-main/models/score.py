import torch
import torch.nn as nn
import torch.nn.functional as F
import models
import torch_geometric
from torch_geometric.nn import GCNConv

'''class Score(nn.Module):
    #随机
    def __init__(self,hidden_size):
        super(Score,self).__init__()
        self.n_w = nn.Linear(2*hidden_size,1)

    def compute_score(self,x,neibour,sub_graph,mask,nei_pad_mask):
        score=nei_pad_mask
        score=F.softmax(score,-1)
        return score'''
'''class Score(nn.Module):
    #pmi
    def __init__(self,hidden_size):
        super(Score,self).__init__()
        self.n_w = nn.Linear(hidden_size*2, 1)

    def compute_score(self,x,neibour,sub_graph,mask,nei_pad_mask):
        score = mask
        score = score.sum(2)
        #score=torch.sigmoid(score)
        score = score + nei_pad_mask
        score=F.softmax(score,-1)
        #print(score)
        return score'''

class Score(nn.Module):
    def __init__(self,hidden_size):
        super(Score,self).__init__()
        self.n_w = nn.Linear(hidden_size*2, 1)

    def compute_score(self,x,neibour,sub_graph,pmi,nei_pad_mask):

        assert x.size(1)!=0
        assert neibour.size(1)!=0
        assert sub_graph.size(0)!=0
        sub_graph=sub_graph.unsqueeze(1)
        sub_graph=sub_graph.expand(sub_graph.size(0),neibour.size(1),sub_graph.size(2))
        mid=torch.cat((neibour,sub_graph),2)
        score=self.n_w(mid)
        score = torch.sigmoid(score)
        score = score.sum(2)
        score=score+nei_pad_mask
        score=F.softmax(score,-1)

        return score

class rl_sample(nn.Module):
    def __init__(self,config):
        super(rl_sample, self).__init__()
        self.input_size=config.emb_size
        self.hidden_size=config.encoder_hidden_size
        self.score=Score(self.hidden_size)
        self.linear2 = nn.Linear(self.input_size, self.hidden_size)
        self.sample_num=config.sample_num

    def forward(self, x,sub_graph,neibour,mask,nei_pad_mask,nei_num,sub_num,sample_type=None):
        nei_pad_size=nei_pad_mask.size(1)
        neibour_id=torch.arange(0,nei_pad_size).unsqueeze(0).expand(neibour.size(0),nei_pad_size)
        neibour_id=neibour_id.cuda()
        sample_num=self.sample_num
        nei_num=nei_num.expand(nei_num.size(0),sample_num).float()
        sub_num=sub_num.expand(sub_num.size(0),sample_num).float()

        neibour=self.linear2(neibour)
        att= self.score.compute_score(x,neibour,sub_graph,mask,nei_pad_mask)

        assert sample_type in ['train','eval']
        if sample_type=='train':
            sample_index = torch.multinomial(att, sample_num, replacement=False)
            #sth, sample_index = torch.topk(att, sample_num)

        elif sample_type=='eval':
            sth,sample_index=torch.topk(att,sample_num)
            #sample_index = torch.multinomial(att, sample_num, replacement=False)
            #sample_index=torch.randint_like(att,low=0,high=att.size(0))

        one_hot = torch.zeros(att.size())
        one_hot=one_hot.cuda()
        one_hot=one_hot.scatter_(1, sample_index.long(), 1)  # 采样所在位置变成1
        one_hot=one_hot.bool()

        sample_prob = torch.masked_select(att, one_hot)  # 取出所在位置的概率

        sample_prob=sample_prob.view(att.size(0),-1).float()
        sample_prob=(sample_prob*nei_num).mean(1,False)

        sample_id = torch.masked_select(neibour_id, one_hot)
        sample_id=sample_id.view(att.size(0),-1).float()
        sample_id=sample_id+sub_num
        sample_id=sample_id.long()

        return sample_id, sample_prob




