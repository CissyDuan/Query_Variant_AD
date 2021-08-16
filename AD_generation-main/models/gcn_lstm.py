import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric
from torch_geometric.nn import GATConv,GINConv,GCNConv,GatedGraphConv




class Graph_encoder_sample_weight(nn.Module):
    def __init__(self, config, embedding=None, word_type_emb=None):
        super(Graph_encoder_sample_weight, self).__init__()
        self.config=config

        self.embedding = embedding
        self.word_type_emb = word_type_emb
        self.gcn1 = GCNConv(config.decoder_hidden_size, config.decoder_hidden_size)
        self.gcn2 = GCNConv(config.decoder_hidden_size, config.decoder_hidden_size)
        self.w = nn.Linear(config.emb_size, config.decoder_hidden_size)
        self.tanh = nn.Tanh()
        self.pool1 = nn.Linear(config.decoder_hidden_size, config.decoder_hidden_size)
        self.pool2 = nn.Linear(config.decoder_hidden_size, 1, bias=False)

        self.x_w = nn.Linear(config.decoder_hidden_size, config.decoder_hidden_size)
        self.g_w = nn.Linear(config.decoder_hidden_size, config.decoder_hidden_size)

        self.lstm_x = nn.LSTM(config.decoder_hidden_size, config.decoder_hidden_size, batch_first=True)
        self.lstm_g = nn.LSTM(config.decoder_hidden_size, config.decoder_hidden_size, batch_first=True)


    def pooling(self,x,mask):
        x=x.view(mask.size(0),mask.size(1),-1)
        h = torch.tanh(self.pool1(x))
        # print(h)
        score = torch.squeeze(self.pool2(h), -1)
        #print(score.size())
        #print(mask.size())

        score=score+mask
        alpha = F.softmax(score, -1)
        # print(alpha)
        x_weight=torch.unsqueeze(alpha, -1) * x
        #print(x_weight.size())
        graph_out= torch.sum(x_weight, 1).unsqueeze(1)
        #print(graph_out.size())

        return graph_out


    def forward(self,data_batch,mask_pad,mask_score):
        #length_x=mask_pad.sum(1).long()
        x = data_batch.x
        word_type = data_batch.word_type
        edge_index = data_batch.edge_index
        edge_attr = data_batch.edge_attr
        x = self.embedding(x)
        if self.config.use_wordtype==True:
            word_type = self.word_type_emb(word_type)
            x = x + word_type
        x = self.w(x)

        x1_mid = self.gcn1(x, edge_index,edge_weight=edge_attr)  # maybe use highway
        x1_cat = torch.cat((x.unsqueeze(1), x1_mid.unsqueeze(1)), 1)
        x1 = self.lstm_x(x1_cat)[0][:, -1, :]
        x = x1

        graph_out_tensor1 = self.pooling(x, mask_score)

        x2_mid = self.gcn2(x1, edge_index,edge_weight=edge_attr)

        x2_cat = torch.cat((x1.unsqueeze(1), x2_mid.unsqueeze(1)), 1)
        x2 = self.lstm_x(x2_cat)[0][:, -1, :]
        x = x2

        graph_out_tensor2 = self.pooling(x, mask_score)

        graph_out_tensor_cat = torch.cat((graph_out_tensor1, graph_out_tensor2), 1)

        graph_out_tensor = self.lstm_g(graph_out_tensor_cat)[0][:, -1, :]

        graph_out_tensor = torch.squeeze(graph_out_tensor, 1)

        x = self.tanh(x)
        x = x.view(mask_pad.size(0), mask_pad.size(1), -1)
        mask_pad = mask_pad.expand(x.size(0), x.size(1), x.size(2))

        x = x * mask_pad
        return x,graph_out_tensor


'''class Graph_encoder_sample_weight(nn.Module):
    def __init__(self, config, embedding=None, word_type_emb=None):
        super(Graph_encoder_sample_weight, self).__init__()
        self.config=config

        self.embedding = embedding
        self.word_type_emb = word_type_emb
        self.gcn1 = GCNConv(config.decoder_hidden_size, config.decoder_hidden_size)
        self.gcn2 = GCNConv(config.decoder_hidden_size, config.decoder_hidden_size)
        self.w = nn.Linear(config.emb_size, config.decoder_hidden_size)
        self.tanh = nn.Tanh()
        self.pool1 = nn.Linear(config.decoder_hidden_size, config.decoder_hidden_size)
        self.pool2 = nn.Linear(config.decoder_hidden_size, 1, bias=False)


    def pooling(self,x,mask):
        x=x.view(mask.size(0),mask.size(1),-1)
        h = torch.tanh(self.pool1(x))
        # print(h)
        score = torch.squeeze(self.pool2(h), -1)
        #print(score.size())
        #print(mask.size())

        score=score+mask
        alpha = F.softmax(score, -1)
        # print(alpha)
        x_weight=torch.unsqueeze(alpha, -1) * x
        #print(x_weight.size())
        graph_out= torch.sum(x_weight, 1).unsqueeze(1)
        #print(graph_out.size())

        return graph_out


    def forward(self,data_batch,mask_pad,mask_score):
        #length_x=mask_pad.sum(1).long()
        x = data_batch.x
        word_type = data_batch.word_type
        edge_index = data_batch.edge_index
        edge_attr = data_batch.edge_attr
        x = self.embedding(x)
        if self.config.use_wordtype==True:
            word_type = self.word_type_emb(word_type)
            x = x + word_type
        x = self.w(x)

        x1 = self.gcn1(x, edge_index, edge_weight=edge_attr)  # maybe use highway
        x1=F.relu(x1)
        x2 = self.gcn2(x1, edge_index, edge_weight=edge_attr)
        graph_out_tensor = self.pooling(x2, mask_score)
        graph_out_tensor = torch.squeeze(graph_out_tensor, 1)

        #x = self.tanh(x)
        x = x.view(mask_pad.size(0), mask_pad.size(1), -1)
        mask_pad = mask_pad.expand(x.size(0), x.size(1), x.size(2))

        x = x * mask_pad
        return x,graph_out_tensor'''

'''class Graph_encoder_sample_weight(nn.Module):
    def __init__(self, config, embedding=None, word_type_emb=None):
        super(Graph_encoder_sample_weight, self).__init__()
        self.config=config

        self.embedding = embedding
        self.word_type_emb = word_type_emb
        self.gcn1 = GATConv(config.decoder_hidden_size, config.decoder_hidden_size)
        self.gcn2 = GATConv(config.decoder_hidden_size, config.decoder_hidden_size)
        self.w = nn.Linear(config.emb_size, config.decoder_hidden_size)
        self.tanh = nn.Tanh()
        self.pool1 = nn.Linear(config.decoder_hidden_size, config.decoder_hidden_size)
        self.pool2 = nn.Linear(config.decoder_hidden_size, 1, bias=False)

        self.lstm_x = nn.LSTM(config.decoder_hidden_size, config.decoder_hidden_size, batch_first=True)
        self.lstm_g = nn.LSTM(config.decoder_hidden_size, config.decoder_hidden_size, batch_first=True)


    def pooling(self,x,mask):
        x=x.view(mask.size(0),mask.size(1),-1)
        h = torch.tanh(self.pool1(x))
        # print(h)
        score = torch.squeeze(self.pool2(h), -1)
        #print(score.size())
        #print(mask.size())

        score=score+mask
        alpha = F.softmax(score, -1)
        # print(alpha)
        x_weight=torch.unsqueeze(alpha, -1) * x
        #print(x_weight.size())
        graph_out= torch.sum(x_weight, 1).unsqueeze(1)
        #print(graph_out.size())

        return graph_out



    def forward(self,data_batch,mask_pad,mask_score):
        #length_x=mask_pad.sum(1).long()
        x = data_batch.x
        word_type = data_batch.word_type
        edge_index = data_batch.edge_index
        edge_attr = data_batch.edge_attr
        x = self.embedding(x)
        if self.config.use_wordtype==True:
            word_type = self.word_type_emb(word_type)
            x = x + word_type
        x = self.w(x)

        x1 = self.gcn1(x, edge_index)  # maybe use highway
        x1=F.relu(x1)
        x2 = self.gcn2(x1, edge_index)
        graph_out_tensor = self.pooling(x2, mask_score)
        graph_out_tensor = torch.squeeze(graph_out_tensor, 1)

        #x = self.tanh(x)
        x = x.view(mask_pad.size(0), mask_pad.size(1), -1)
        mask_pad = mask_pad.expand(x.size(0), x.size(1), x.size(2))

        x = x * mask_pad
        return x,graph_out_tensor'''