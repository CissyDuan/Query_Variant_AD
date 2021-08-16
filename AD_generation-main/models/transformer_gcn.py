
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import models
from torch.nn.utils.rnn import pad_sequence
from Data import *
import torch_geometric
from torch_geometric.data import Batch,Data



def padding_mask(seq_k, seq_q):
    len_q = seq_q.size(1)
    pad_mask = seq_k.eq(0)
    pad_mask = pad_mask.unsqueeze(1).expand(-1, len_q, -1)  # shape [B, L_q, L_k]
    return pad_mask

def sequence_mask(seq):
    batch_size, seq_len = seq.size()
    mask = torch.triu(torch.ones((seq_len, seq_len), dtype=torch.uint8),
                    diagonal=1)
    mask = mask.unsqueeze(0).expand(batch_size, -1, -1)  # [B, L, L]
    return mask

def mask_trans(mask):
    mask = mask.float().masked_fill(mask == 1, float('-inf')).masked_fill(mask == 0, float(0.0))
    return mask

def generate_square_subsequent_mask(sz):

    mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))


    return mask


class PositionalEncoding(nn.Module):
    "Implement the PE function."

    def __init__(self, d_model, dropout, max_len=20):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Compute the positional encodings once in log space.
        self.pe = torch.zeros(max_len, d_model)
        #print(self.pe)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0., d_model, 2) *
                             -(math.log(10000.0) / d_model))
        self.pe[:, 0::2] = torch.sin(position * div_term)
        self.pe[:, 1::2] = torch.cos(position * div_term)
        self.pe = self.pe.unsqueeze(0)
        self.pe=self.pe.cuda()

        #self.register_buffer('pe', pe)

    def forward(self, x):
        #print(x.size())
        #print(self.pe.size())
        x = x + self.pe[:, :x.size(1),:]
        return self.dropout(x)

class Transformer_gcn(nn.Module):
    def __init__(self, config, vocab, use_cuda, pretrain=None):
        super(Transformer_gcn, self).__init__()
        self.use_cuda=use_cuda
        self.vocab = vocab
        self.vocab_size = vocab.voc_size
        if pretrain is not None:
            self.embedding = pretrain['emb']
        else:
            self.embedding = nn.Embedding(self.vocab_size, config.emb_size)
        self.word_type_emb = nn.Embedding(config.type_num_gen, config.emb_size)
        self.w=nn.Linear(config.emb_size,config.decoder_hidden_size)
        #self.transformer=nn.Transformer(d_model=config.decoder_hidden_size, nhead=config.num_head, num_encoder_layers=6, num_decoder_layers=6, dim_feedforward=config.dim_feedforward, dropout=0.1, custom_encoder=None, custom_decoder=None)

        self.graph_encoder=models.Graph_encoder(config)
        num_encoder_layers=2
        encoder_layer =nn.TransformerEncoderLayer(config.decoder_hidden_size, config.num_head, dim_feedforward=config.dim_feedforward, dropout=0.1)
        encoder_norm = nn.LayerNorm(config.decoder_hidden_size)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)

        num_decoder_layers=3
        decoder_layer = nn.TransformerDecoderLayer(config.decoder_hidden_size, config.num_head, dim_feedforward=config.dim_feedforward, dropout=0.1)
        decoder_norm = nn.LayerNorm(config.decoder_hidden_size)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_decoder_layers, decoder_norm)

        self.o=nn.Linear(config.decoder_hidden_size,config.vocab_size)
        self.config=config
        self.pos_emb = PositionalEncoding(config.emb_size, 0.1, max_len=config.max_tgt_len+2)

    def forward(self, x, adj, adj_weight,tgt, word_type):
        x = [i.cuda() for i in x]
        word_type = [w.cuda() for w in word_type]
        tgt=tgt.cuda()

        tgt=tgt[:,:-1]

        x = pad_sequence(x,batch_first=True)
        word_type = pad_sequence(word_type,batch_first=True)

        mask_pad_g = x.ne(0).float().unsqueeze(-1)


        x_emb=self.embedding(x)
        word_type_emb=self.word_type_emb(word_type)
        x_emb=x_emb+word_type_emb

        tgt_emb=self.embedding(tgt)
        tgt_emb= self.pos_emb(tgt_emb)

        #src_mask=generate_square_subsequent_mask(x.size(1)).cuda()
        tgt_mask=generate_square_subsequent_mask(tgt.size(1)).cuda()

        src_pad_mask=x.eq(0).cuda()
        tgt_pad_mask=tgt.eq(0).cuda()

        memory_pad_mask=x.eq(0).cuda()
        #print(memory_pad_mask.size())

        x=self.w(x_emb)
        tgt=self.w(tgt_emb)

        x=x.unbind(0)
        data_batch = []
        for x, adj, adj_weight in zip(x, adj, adj_weight):
            data = Data(x=x, edge_index=adj, edge_attr=adj_weight)
            data_batch.append(data)

        data_batch = Batch.from_data_list(data_batch, follow_batch=[])
        x=self.graph_encoder(data_batch,mask_pad_g)

        x=x.transpose(0,1)
        tgt = tgt.transpose(0, 1)
        #print(tgt)
        #print(x.size())
        #print(tgt.size())

        memory = self.encoder(x,src_key_padding_mask=src_pad_mask)
        #print(memory.size())
        output = self.decoder(tgt=tgt, memory=memory,tgt_mask=tgt_mask,memory_mask=None,tgt_key_padding_mask=tgt_pad_mask,memory_key_padding_mask=memory_pad_mask)

        output=self.o(output)
        output=output.transpose(0,1)
        output = F.softmax(output, -1)
        #print(output.size())

        return output,[],[]

    def sample(self,x,adj,adj_weight,word_type):


        x = [i.cuda() for i in x]
        word_type = [w.cuda() for w in word_type]


        x = pad_sequence(x, batch_first=True)
        word_type = pad_sequence(word_type, batch_first=True)

        mask_pad_g = x.ne(0).float().unsqueeze(-1)
        tgt = torch.zeros(x.size(0)).long().fill_(self.vocab.word2id('[START]')).unsqueeze(1)

        tgt = tgt.cuda()

        x_emb = self.embedding(x)
        word_type_emb = self.word_type_emb(word_type)
        x_emb = x_emb + word_type_emb

        #src_mask = generate_square_subsequent_mask(x.size(1)).cuda()

        src_pad_mask = x.eq(0).cuda()
        memory_pad_mask = x.eq(0).cuda()
        x = self.w(x_emb)

        x = x.unbind(0)
        data_batch = []
        for x, adj, adj_weight in zip(x, adj, adj_weight):
            data = Data(x=x, edge_index=adj, edge_attr=adj_weight)
            data_batch.append(data)

        data_batch = Batch.from_data_list(data_batch, follow_batch=[])
        x = self.graph_encoder(data_batch, mask_pad_g)

        x = x.transpose(0, 1)


        memory = self.encoder(x,src_key_padding_mask=src_pad_mask)

        #print(tgt)

        for step in range(self.config.max_tgt_len+1):
            tgt_mask = generate_square_subsequent_mask(tgt.size(1)).cuda()
            input= self.embedding(tgt)
            input = self.pos_emb(input)
            input= self.w(input)
            input = input.transpose(0, 1)

            output = self.decoder(tgt=input, memory=memory, tgt_mask=tgt_mask,
                                   memory_key_padding_mask=memory_pad_mask)
            #output = self.decoder(input, memory)
            output = self.o(output)[-1,:,:]
            pred= output.max(dim=1)[1].unsqueeze(1)
            tgt=torch.cat((tgt,pred),1)

        tgt=tgt[:,1:]
        #pred_id=torch.unbind(tgt,0)

        return tgt



    '''def compute_loss(self, hidden_outputs, targets):
        #hidden_outputs=hidden_outputs[:,1:,:]
        targets=targets[:,1:]
        #print(hidden_outputs.size())
        #print(targets.size())
        assert hidden_outputs.size(1) == targets.size(1) and hidden_outputs.size(0) == targets.size(0)
        outputs = hidden_outputs.contiguous().view(-1, hidden_outputs.size(2))
        #print(outputs.size())
        targets = targets.contiguous().view(-1)
        #print(targets.size())
        weight = torch.ones(outputs.size(-1))
        weight[0] = 0
        weight[3] = 0
        weight = weight.to(outputs.device)
        loss = F.nll_loss(torch.log(outputs), targets, weight=weight, reduction='mean')
        #print(loss.size())
        pred = outputs.max(dim=1)[1]
        num_correct = pred.data.eq(targets.data).masked_select(targets.ne(PAD).data).sum()
        num_total = targets.ne(0).data.sum()
        #loss = loss.div(num_total.float())
        acc = num_correct.float() / num_total.float()
        return loss, acc'''
    def compute_loss(self, hidden_outputs, targets):
        targets = targets[:, 1:]
        assert hidden_outputs.size(1) == targets.size(1) and hidden_outputs.size(0) == targets.size(0)
        outputs = hidden_outputs.contiguous().view(-1, hidden_outputs.size(2))
        targets_l = targets.contiguous().view(-1)
        weight = torch.ones(outputs.size(-1))
        weight[0] = 0
        weight[3] = 0
        weight = weight.to(outputs.device)
        loss = F.nll_loss(torch.log(outputs), targets_l, weight=weight,reduction='none')
        loss=loss.view(targets.size(0),targets.size(1))
        loss=loss.sum(1)
        num_none = targets.eq(0) + targets.eq(3)
        num_total_loss = num_none.eq(0)
        num_total_loss = num_total_loss.ne(0).sum(1)
        num_total = num_total_loss.data.sum()
        loss = loss.div(num_total_loss.float())
        case_num=float(loss.size(0))
        loss=loss.div(case_num)

        #loss = loss.div(num_total.float())
        loss=loss.sum(0)

        pred = outputs.max(dim=1)[1]
        num_correct = pred.data.eq(targets_l.data).masked_select(targets_l.ne(PAD).data).sum()
        acc = num_correct.float() / num_total.float()

        return loss, acc

    def compute_loss_sample(self, hidden_outputs, targets):

        targets = targets[:, 1:]
        assert hidden_outputs.size(1) == targets.size(1) and hidden_outputs.size(0) == targets.size(0)
        outputs = hidden_outputs.contiguous().view(-1, hidden_outputs.size(2))
        targets_l = targets.contiguous().view(-1)
        weight = torch.ones(outputs.size(-1))
        weight[0] = 0
        weight[3] = 0
        weight = weight.to(outputs.device)
        loss = F.nll_loss(torch.log(outputs), targets_l, weight=weight,reduction='none')
        #print(loss.size())
        loss=loss.view(targets.size(0),targets.size(1))
        #print(loss.size())
        loss=loss.sum(1)
        num_none=targets.eq(0)+targets.eq(3)
        num_total_loss=num_none.eq(0)
        num_total_loss = num_total_loss.ne(0).sum(1)
        num_total = num_total_loss.data.sum()
        loss = loss.div(num_total_loss.float())
        #loss = loss.div(num_total.float())
        #case_num = float(loss.size(0))
        #loss = loss.div(case_num)

        pred = outputs.max(dim=1)[1]
        num_correct = pred.data.eq(targets_l.data).masked_select(targets_l.ne(PAD).data).sum()
        acc = num_correct.float() / num_total.float()

        return loss, acc

    '''def compute_loss_sample(self, hidden_outputs, targets):

        targets = targets[:, 1:]
        assert hidden_outputs.size(1) == targets.size(1) and hidden_outputs.size(0) == targets.size(0)
        outputs = hidden_outputs.contiguous().view(-1, hidden_outputs.size(2))
        targets_l = targets.contiguous().view(-1)
        weight = torch.ones(outputs.size(-1))
        weight[0] = 0
        weight[3] = 0
        weight = weight.to(outputs.device)
        loss = F.nll_loss(torch.log(outputs), targets_l, weight=weight,reduction='none')
        #print(loss.size())
        loss=loss.view(targets.size(0),targets.size(1))
        #print(loss.size())
        loss=loss.sum(1)
        num_none=targets.eq(0)+targets.eq(3)
        num_total_loss=num_none.eq(0)
        num_total_loss = num_total_loss.ne(0).sum(1)
        num_total = num_total_loss.data.sum()
        loss = loss.div(num_total.float())

        pred = outputs.max(dim=1)[1]
        num_correct = pred.data.eq(targets_l.data).masked_select(targets_l.ne(PAD).data).sum()
        acc = num_correct.float() / num_total.float()



        return loss, acc'''




