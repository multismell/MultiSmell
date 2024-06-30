import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import XLNetModel,BertModel
import transformers

print("transformers 版本:",transformers.__version__)
class config(object):
    def __init__(self, model_name,types):
        self.model_name = model_name
        self.types = types
        self.save_path = 'E:/save_dict/' + self.model_name + '.ckpt'
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.require_improvment = 1000
        self.pretain = 'xlnet'
        self.num_class = 3
        self.num_epochs = 5
        self.batch_size = 128
        self.padding_size = 20
        self.learning_rate = 1e-4
        self.xlnet_path = 'E:/Pre-training/XLNet'
        self.bert_path = 'D:/Bert/BERT12'
        self.gru_hidden = 256
        self.rnn_hidden = 256
        # self.hidden_size = 128
        self.hidden_size = 256
        self.num_filter = 256
        self.class_nums = 2
        self.pool_type = max
        self.num_layer = 2
        self.gat_layers = 3
        self.dropout = 0.5
        self.linear = 128
        self.k = 5 # 10折

class Bi_LSTM(nn.Module):
    def __init__(self):
        super(Bi_LSTM, self).__init__()
        self.lstm_1 = torch.nn.LSTM(
            input_size=768,
            hidden_size=128,
            num_layers=2,
            batch_first=True,
            dropout=0.4,
            bidirectional=True
        )
        self.dropout = torch.nn.Dropout(0.4)
        self.linear_1 = torch.nn.Linear(256,128)
        self.act = torch.nn.Tanh()
        self.flatten = torch.nn.Flatten()

    def forward(self,inputs):
        inputs = inputs.reshape(inputs.shape[0],1,-1)
        out,_ = self.lstm_1(inputs)
        out = self.dropout(out)
        out = out[:, -1, :]
        out = self.linear_1(out)
        return out

class CNN(nn.Module):
    def __init__(self,config):
        super(CNN, self).__init__()
        self.config = config
        self.cnn_method = nn.Sequential(
            torch.nn.Conv1d(20,128,kernel_size=6,padding='same'),
            torch.nn.ReLU(),
            torch.nn.BatchNorm1d(128),
            torch.nn.Conv1d(128,128,kernel_size=6,padding='same'),
            torch.nn.ReLU(),
        )
        self.cnn_class = nn.Sequential(
            torch.nn.Conv1d(10, 128, kernel_size=6, padding='same'),
            torch.nn.ReLU(),
            torch.nn.BatchNorm1d(128),
            torch.nn.Conv1d(128, 128, kernel_size=6, padding='same'),
            torch.nn.ReLU(),
        )
  
        self.flatten = nn.Flatten()

    def forward(self,inputs):
        inputs = inputs.view(inputs.shape[0],-1,1)
        if self.config.types == 'method':
            out = self.cnn_method(inputs)
        else:
            out = self.cnn_class(inputs)

        out = self.flatten(out)

        return out

class MultiSmell(nn.Module):
    def __init__(self, config):
        super(MultiSmell, self).__init__()
        self.config = config
        self.xlnet = XLNetModel.from_pretrained(config.xlnet_path)
        for param in self.xlnet.parameters():
            param.requires_grad = False
        self.bi_lstm = Bi_LSTM()
        self.cnn = CNN(config)
        self.dropout = torch.nn.Dropout(0.2)
        self.act = torch.nn.ReLU()
        self.flatten = nn.Flatten()
        self.fc = torch.nn.Linear(256,128)
        self.gat = HGAT(config)

    def forward(self, x1, mask, x2, label):
        encoder_out = self.xlnet(x1, attention_mask=mask)[0]
        encoder_out, _ = torch.max(encoder_out, dim=1)
        out_1 = self.bi_lstm(encoder_out)


        x2 = x2.view(x2.shape[0],-1,1)
        out_2 = self.cnn(x2)



        out = torch.cat([out_1,out_2],dim=1)

        logits = self.gat(out)
        return logits

class HGAT(nn.Module):
    def __init__(self, config):
        super(HGAT, self).__init__()
        self.config = config
        hidden_size = 256 
        self.embeding = nn.Embedding(config.num_class, hidden_size)
        self.relation = nn.Linear(hidden_size, hidden_size)
        self.fc1 = nn.Linear(hidden_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 1)
        self.layers = nn.ModuleList([GATLayer(hidden_size) for _ in range(config.gat_layers)])

    def forward(self, x, mask=None):
        p = torch.arange(self.config.num_class, device=x.device).long()
        p = self.embeding(p)
        p = self.relation(p)
        p = p.unsqueeze(0).expand(x.size(0), p.size(0), p.size(1))
        label, p = self.gat_layer(x.unsqueeze(1), p,mask)
        p = self.fc1(p)
        p = torch.tanh(p)
        p = self.fc2(p).squeeze(2)
        p = torch.sigmoid(p)
        return p

    def gat_layer(self, x, p, mask=None):
        for m in self.layers:
            x, p = m(x, p, mask)
        return x, p

class GATLayer(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.ra1 = RelationAttention(hidden_size)
        self.ra2 = RelationAttention(hidden_size)

    def forward(self, x, p, mask=None):
        x = self.ra1(x, p) + x
        p = self.ra2(p,x, mask) + p
        return x, p


class RelationAttention(nn.Module):
    def __init__(self, hidden_size):
        super(RelationAttention, self).__init__()
        self.query = nn.Linear(hidden_size, hidden_size)
        self.key = nn.Linear(hidden_size, hidden_size)
        self.value = nn.Linear(hidden_size, hidden_size)
        self.score = nn.Linear(2 * hidden_size, 1)
        self.gate = nn.Linear(hidden_size * 2, 1)

    def forward(self, p, x, mask=None):
        q = self.query(p)
        k = self.key(x)
        score = self.fuse(q, k)
        if mask is not None:
            mask = 1 - mask[:, None, :].expand(-1, score.size(1), -1)
            score = score.masked_fill(mask == 1, -1e9)
        score = F.softmax(F.leaky_relu(score,0.2), 2)
        v = self.value(x)
        out = torch.einsum('bcl,bld->bcd', score, v) + p
        return out

    def fuse(self, x, y):
        x = x.unsqueeze(2).expand(-1, -1, y.size(1), -1)
        y = y.unsqueeze(1).expand(-1, x.size(1), -1, -1)
        temp = torch.cat([x, y], 3)
        return self.score(temp).squeeze(3)