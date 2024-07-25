import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet152, resnet101
from torch.nn.utils.rnn import pack_padded_sequence
from torch.nn.utils.rnn import pad_packed_sequence
from lib.config import cfg
import lib.utils as utils
from model.att_basic_model import AttBasicModel
import blocks
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#print(torch.cuda.is_available())

class Encoder(nn.Module):
    def __init__(self, architecture='resnet152'):
        super(Encoder, self).__init__()
        self.architecture = architecture
        if architecture == 'resnet152':
            self.net = resnet152(pretrained=True)
            ## pretrained 设置为 True，会自动下载模型 所对应权重，并加载到模型中
            # 假设 我们的 分类任务只需要 分 100 类，那么我们应该做的是
            # 1. 查看 resnet 的源码
            # 2. 看最后一层的 名字是啥
            # 3. 替换掉这个层

            '''对于网络的修改怎样可以快速的去除model本身的网络层呢？
一个继承nn.module的model它包含一个叫做children()的函数，这个函数可以用来提取出model每一层的网络结构，在此基础上进行修改即可，修改方法如下(去除后两层)：
nn.Sequential定义的网络中各层会按照定义的顺序进行级联，因此需要保证各层的输入和输出之间要衔接。
而且里面的模块必须是按照顺序进行排列的，所以我们必须确保前一个模块的输出大小和下一个模块的输入大小是一致的，
            '''


            self.net = nn.Sequential(*list(self.net.children())[:-2])
            self.dim = 2048
        else:
            self.net = resnet101(pretrained=True)
            self.net = nn.Sequential(*list(self.net.children())[:-2])
            self.dim = 2048
        
        self.fine_tune()
    #前向计算
    def forward(self, img):
        feats = self.net(img)
        feats = feats.permute(0, 2, 3, 1)#Tensor.permute(a,b,c,d, ...)：permute函数可以对任意高维矩阵进行转置
        #print(feats)
        feats = feats.view(feats.size(0), -1, feats.size(-1))
        #print(feats)
        return feats
    #模型微调
    def fine_tune(self, fine_tune=False):
        # 如果只想训练 最后某几层的话，应该做的是：
        # 1. 将其它层的参数 requires_grad 设置为 False
            # 这一步可以节省大量的时间，因为多数的参数不需要计算梯度
        if not fine_tune:
            for param in self.net.parameters():
                param.requires_grad = False


class Attention(nn.Module):
    
    def __init__(self, encoder_dim, decoder_dim, attention_dim):
        super(Attention, self).__init__()
        self.W1 = nn.Linear(encoder_dim, attention_dim)#线性层对编码图像进行变换将cnn输出的feature转换成特定维度的线性层  #nn.Linear（）是用于设置网络中的全连接层的，需要注意的是全连接层的输入与输出都是二维张量，一般形状为[batch_size, size]，不同于卷积层要求输入输出是四维张量
        self.W2 = nn.Linear(decoder_dim, attention_dim)#线性层用于变换解码器的输出将decode输出的hidden_state转换成特定维度的线性层
        self.V = nn.Linear(attention_dim, 1)#线性层，用于计算要最大化的值
        self.tanh = nn.Tanh()#激活函数
        self.softmax = nn.Softmax(dim=1)#当nn.Softmax的输入是一个二维张量时，其参数dim = 0，是让列之和为1；dim = 1，是让行之和为1。
    
    def forward(self, img_feats, hidden):
        #(batch_size, num_pixels, attention_dim)
        x = self.W1(img_feats)
        # (batch_size, attention_dim)
        y = self.W2(hidden)
        # (batch_size, num_pixels)
        x = self.V(self.tanh(x + y.unsqueeze(1))).squeeze(2)
        '''#torch.unsqueeze(input, dim, out=None)
         作用：扩展维度
        返回一个新的张量，对输入的既定位置插入维度 1
        torch.squeeze 详解
        作用：降维
        torch.squeeze(input, dim=None, out=None)

        将输入张量形状中的1 去除并返回。 如果输入是形如(A×1×B×1×C×1×D)，那么输出形状就为： (A×B×C×D)

        当给定dim时，那么挤压操作只在给定维度上。例如，输入形状为: (A×1×B), squeeze(input, 0) 将会保持张量不变，只有用 squeeze(input, 1)，形状会变成 (A×B)。
        '''
        # (batch_size, num_pixels)
        alphas = self.softmax(x)#每个像素的概率被计算出来了
        # (batch_size, encoder_dim)
        weighted_feats = (img_feats * alphas.unsqueeze(2)).sum(dim=1) #每个像素点加权求和
        
        return weighted_feats, alphas

'''
class Generator(AttBasicModel):
    def __init__(self):
        super(Generator, self).__init__()
        self.num_layers = 2

        # First LSTM layer
        rnn_input_size = cfg.MODEL.RNN_SIZE + cfg.MODEL.BILINEAR.DIM
        self.att_lstm = nn.LSTMCell(rnn_input_size, cfg.MODEL.RNN_SIZE)
        self.ctx_drop = nn.Dropout(cfg.MODEL.DROPOUT_LM)

        self.attention = blocks.create(
            cfg.MODEL.BILINEAR.DECODE_BLOCK,
            embed_dim=cfg.MODEL.BILINEAR.DIM,
            att_type=cfg.MODEL.BILINEAR.ATTTYPE,
            att_heads=cfg.MODEL.BILINEAR.HEAD,
            att_mid_dim=cfg.MODEL.BILINEAR.DECODE_ATT_MID_DIM,
            att_mid_drop=cfg.MODEL.BILINEAR.DECODE_ATT_MID_DROPOUT,
            dropout=cfg.MODEL.BILINEAR.DECODE_DROPOUT,
            layer_num=cfg.MODEL.BILINEAR.DECODE_LAYERS
        )
        # context vector
        self.att2ctx = nn.Sequential(
            nn.Linear(cfg.MODEL.BILINEAR.DIM + cfg.MODEL.RNN_SIZE, 2 * cfg.MODEL.RNN_SIZE),
            nn.GLU()
        )

    # state[0] -- h, state[1] -- c
    def Forward(self, **kwargs):
        wt = kwargs[cfg.PARAM.WT]
        att_feats = kwargs[cfg.PARAM.ATT_FEATS]
        att_mask = kwargs[cfg.PARAM.ATT_FEATS_MASK]
        state = kwargs[cfg.PARAM.STATE]
        gv_feat = kwargs[cfg.PARAM.GLOBAL_FEAT]
        p_att_feats = kwargs[cfg.PARAM.P_ATT_FEATS]

        if gv_feat.shape[-1] == 1:  # empty gv_feat
            if att_mask is not None:
                gv_feat = torch.sum(att_feats * att_mask.unsqueeze(-1), 1) / torch.sum(att_mask.unsqueeze(-1), 1)
            else:
                gv_feat = torch.mean(att_feats, 1)
        xt = self.word_embed(wt.long())

        h_att, c_att = self.att_lstm(torch.cat([xt, gv_feat + self.ctx_drop(state[0][1])], 1),
                                     (state[0][0], state[1][0]))
        # def forward(self, gv_feat, att_feats, att_mask, p_att_feats=None, precompute=False)
        # att_feats为V(1+M)
        att, _ = self.attention(h_att, att_feats, att_mask, p_att_feats, precompute=True)  # vˆd
        # shuchu gv_feat, att_feats
        ctx_input = torch.cat([att, h_att], 1)

        preds = self.att2ctx(ctx_input)  # current context vector ct
        state = [torch.stack((h_att, preds)), torch.stack((c_att, state[1][1]))]

        return preds, state
'''

class GRUDiscriminator(nn.Module):

    def __init__(self, embedding_dim, encoder_dim, gru_units, vocab_size):

        super(GRUDiscriminator, self).__init__()

        self.encoder_dim = encoder_dim
        self.embedding_dim = embedding_dim
        self.gru_units = gru_units
        self.vocab_size = vocab_size
        #print(self.vocab_size)
        self.embedding = nn.Embedding(self.vocab_size, embedding_dim,padding_idx=-1)
        self.gru = nn.GRU(input_size=embedding_dim, hidden_size=gru_units, batch_first=True)

        self.fc1 = nn.Linear(encoder_dim, embedding_dim)
        self.fc2 = nn.Linear(gru_units, 1)
        self.sigmoid = nn.Sigmoid()


    def forward(self, img_feats, target_seq,max_len):
        img_feats = img_feats.permute(0, 2, 1)
        img_feats = F.avg_pool1d(img_feats, img_feats.shape[-1]).squeeze(-1)#平均池化
        img_feats = self.fc1(img_feats)
        #print(target_seq)
        #print(target_seq.size() )
        embeddings = self.embedding(target_seq.long())
        inputs = torch.cat((img_feats.unsqueeze(1), embeddings), 1).to(device)
        #inputs_packed = pack_padded_sequence(inputs, caplen+1, batch_first=True ,enforce_sorted=False)
        outputs, _ = self.gru(inputs)
        #try:
        #outputs = pad_packed_sequence(outputs, batch_first=True)[0]
        #except:
            #pass
        row_indices = torch.arange(0, target_seq.size(0)).long()#返回一个1维张量，长度为floor((end-start)/step)，以step`为步长的一组序列值。
        last_hidden = outputs.permute(0, 2, 1)
        last_hidden = F.avg_pool1d(last_hidden, last_hidden.shape[-1]).squeeze(-1)
        #print(last_hidden.size())
        pred = self.sigmoid(self.fc2(last_hidden))
        #print(pred)
        #print(pred.squeeze(-1).size())
        return pred.squeeze(-1)



class Evaluator(nn.Module):

    def __init__(self, embedding_dim, encoder_dim, gru_units, vocab_size):

        super(Evaluator, self).__init__()

        self.encoder_dim = encoder_dim
        self.embedding_dim = embedding_dim
        self.gru_units = gru_units
        self.vocab_size = vocab_size
        # print(self.vocab_size)
        self.embedding = nn.Embedding(self.vocab_size, embedding_dim, padding_idx=-1)
        self.gru = nn.GRU(input_size=embedding_dim, hidden_size=gru_units, batch_first=True)

        self.fc1 = nn.Linear(encoder_dim, embedding_dim)
        self.fc2 = nn.Linear(gru_units, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, img_feats, target_seq,max_len):
        img_feats = img_feats.permute(0, 2, 1)
        img_feats = F.avg_pool1d(img_feats, img_feats.shape[-1]).squeeze(-1)#平均池化
        img_feats = self.fc1(img_feats)
        #print(target_seq)
        #print(target_seq.size() )
        embeddings = self.embedding(target_seq.long())
        inputs = torch.cat((img_feats.unsqueeze(1), embeddings), 1).to(device)
        #inputs_packed = pack_padded_sequence(inputs, caplen+1, batch_first=True ,enforce_sorted=False)
        outputs, _ = self.gru(inputs)
        #try:
        #outputs = pad_packed_sequence(outputs, batch_first=True)[0]
        #except:
            #pass
        row_indices = torch.arange(0, target_seq.size(0)).long()#返回一个1维张量，长度为floor((end-start)/step)，以step`为步长的一组序列值。
        last_hidden = outputs.permute(0, 2, 1)
        last_hidden = F.avg_pool1d(last_hidden, last_hidden.shape[-1]).squeeze(-1)
        #print(last_hidden.size())
        pred = self.sigmoid(self.fc2(last_hidden))
        #print(pred)
        #print(pred.squeeze(-1).size())
        return pred.squeeze(-1)




