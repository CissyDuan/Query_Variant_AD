import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric
from torch_geometric.nn import GATConv,GINConv,GCNConv,GatedGraphConv



class Graph_encoder(nn.Module):
    def __init__(self, config):
        super(Graph_encoder, self).__init__()
        self.gcn1=GCNConv(config.decoder_hidden_size, config.decoder_hidden_size)
        self.gcn2 = GCNConv(config.decoder_hidden_size, config.decoder_hidden_size)
        #self.tanh = nn.Tanh()
        self.lstm_x=nn.LSTM(config.decoder_hidden_size,config.decoder_hidden_size,batch_first=True)

    def forward(self,data_batch,mask_pad):
        x=data_batch.x
        edge_index=data_batch.edge_index
        edge_attr=data_batch.edge_attr
        x1_mid = self.gcn1(x, edge_index,edge_weight=edge_attr)   # maybe use highway
        x1_cat=torch.cat((x.unsqueeze(1),x1_mid.unsqueeze(1)),1)
        x1=self.lstm_x(x1_cat)[0][:,-1,:]
        x2_mid = self.gcn2(x1, edge_index,edge_weight=edge_attr)
        x2_cat=torch.cat((x1.unsqueeze(1),x2_mid.unsqueeze(1)),1)
        x2=self.lstm_x(x2_cat)[0][:,-1,:]
        #x = self.tanh(x2)
        x=x2
        x=x.view(mask_pad.size(0),mask_pad.size(1),-1)
        mask_pad=mask_pad.expand(x.size(0),x.size(1),x.size(2))
        x=x*mask_pad

        return x

'''class Graph_encoder(nn.Module):
    def __init__(self, config):
        super(Graph_encoder, self).__init__()
        self.gcn1=GCNConv(config.decoder_hidden_size, config.decoder_hidden_size)
        self.gcn2 = GCNConv(config.decoder_hidden_size, config.decoder_hidden_size)
        self.tanh = nn.Tanh()


    def forward(self,data_batch,mask_pad):
        x=data_batch.x
        edge_index=data_batch.edge_index
        edge_attr=data_batch.edge_attr
        x1 = self.gcn1(x, edge_index,edge_weight=edge_attr)   # maybe use highway
        x1=F.relu(x1)
        x2 = self.gcn2(x1, edge_index,edge_weight=edge_attr)
        #x = self.tanh(x2)
        x=x2
        x=x.view(mask_pad.size(0),mask_pad.size(1),-1)
        mask_pad=mask_pad.expand(x.size(0),x.size(1),x.size(2))
        x=x*mask_pad

        return x'''


'''class Graph_encoder(nn.Module):
    def __init__(self, config):
        super(Graph_encoder, self).__init__()
        head=8
        self.gcn1=GATConv(config.decoder_hidden_size, int(config.decoder_hidden_size/head),heads=8)
        self.gcn2 = GATConv(config.decoder_hidden_size, int(config.decoder_hidden_size/head),heads=8)
        self.tanh = nn.Tanh()
        self.lstm_x = nn.LSTM(config.decoder_hidden_size, config.decoder_hidden_size, batch_first=True)

    def forward(self, data_batch, mask_pad):
        x = data_batch.x
        edge_index = data_batch.edge_index
        edge_attr = data_batch.edge_attr
        x1_mid = self.gcn1(x, edge_index)  # maybe use highway
        x1_cat = torch.cat((x.unsqueeze(1), x1_mid.unsqueeze(1)), 1)
        x1 = self.lstm_x(x1_cat)[0][:, -1, :]
        x2_mid = self.gcn2(x1, edge_index)
        x2_cat = torch.cat((x1.unsqueeze(1), x2_mid.unsqueeze(1)), 1)
        x2 = self.lstm_x(x2_cat)[0][:, -1, :]
        # x = self.tanh(x2)
        x = x2
        x = x.view(mask_pad.size(0), mask_pad.size(1), -1)
        mask_pad = mask_pad.expand(x.size(0), x.size(1), x.size(2))
        x = x * mask_pad
        return x'''

'''def forward(self,data_batch,mask_pad):
    x=data_batch.x
    edge_index=data_batch.edge_index
    edge_attr=data_batch.edge_attr
    x1 = self.gcn1(x, edge_index)   # maybe use highway
    x1=F.relu(x1)
    x2 = self.gcn2(x1, edge_index)
    #x = self.tanh(x2)
    x=x2
    x=x.view(mask_pad.size(0),mask_pad.size(1),-1)
    mask_pad=mask_pad.expand(x.size(0),x.size(1),x.size(2))
    x=x*mask_pad

    return x'''



