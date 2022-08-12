import torch
import os
import random
import torch.nn as nn
import pandas as pd
import numpy as np
from tqdm import tqdm


class DecoderBlock(nn.Module):
    def __init__(self, embed_size, heads, forward_expansion, dropout, device):
        super(DecoderBlock, self).__init__()
        self.attention = SelfAttention(embed_size, heads)
        self.norm = nn.LayerNorm(embed_size)
        self.transformer_block = TransformerBlock(
            embed_size, heads, dropout, forward_expansion
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, value, key, src_mask, trg_mask):
        attention = self.attention(x, x, x, trg_mask)
        query = self.dropout(self.norm(attention + x))
        out = self.transformer_block(value, key, query, src_mask)
        return out


class Decoder(nn.Module):
    def __init__(
            self,
            trg_vocab_size,
            embed_size,
            num_layers,
            heads,
            forward_expansion,
            dropout,
            device,
            max_length,
    ):
        super(Decoder, self).__init__()
        self.device = device
        self.word_embedding = nn.Embedding(trg_vocab_size, embed_size)
        self.position_embedding = nn.Embedding(max_length, embed_size)
        self.layers = nn.ModuleList(
            [DecoderBlock(embed_size, heads, forward_expansion, dropout, device)
             for _ in range(num_layers)]
        )
        self.fc_out = nn.Linear(embed_size, trg_vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, enc_out, src_mask, trg_mask):
        N, seq_length = x.shape
        positions = torch.arange(0, seq_length).expand(N, seq_length).to(self.device)
        x = self.dropout((self.word_embedding(x) + self.position_embedding(positions)))

        for layer in self.layers:
            x = layer(x, enc_out, enc_out, src_mask, trg_mask)

        out = self.fc_out(x)
        return out


class Transformer(nn.Module):
    def __init__(
            self,
            src_vocab_size,
            trg_vocab_size,
            src_pad_idx,
            trg_pad_idx,
            embed_size=256,
            num_layers=6,
            forward_expansion=4,
            heads=8,
            dropout=0,
            device="cuda",
            max_length=100
    ):
        super(Transformer, self).__init__()
        self.encoder = Encoder(
            src_vocab_size,
            embed_size,
            num_layers,
            heads,
            device,
            forward_expansion,
            dropout,
            max_length
        )
        self.decoder = Decoder(
            trg_vocab_size,
            embed_size,
            num_layers,
            heads,
            forward_expansion,
            dropout,
            device,
            max_length
        )

        self.src_pad_idx = src_pad_idx
        self.trg_pad_idx = trg_pad_idx
        self.device = device

    def make_src_mask(self, src):
        src_mask = (src != self.src_pad_idx).unsqueeze(1).unsqueeze(2)
        # (N, 1, 1, src_len)
        return src_mask.to(self.device)

    def make_trg_mask(self, trg):
        N, trg_len = trg.shape
        trg_mask = torch.tril(torch.ones((trg_len, trg_len))).expand(
            N, 1, trg_len, trg_len
        )
        return trg_mask.to(self.device)

    def forward(self, src, trg):
        src_mask = self.make_src_mask(src)
        trg_mask = self.make_trg_mask(trg)
        enc_src = self.encoder(src, src_mask)
        out = self.decoder(trg, enc_src, src_mask, trg_mask)
        return out


class SelfAttention(nn.Module):
    def __init__(self, embed_size):
        super(SelfAttention, self).__init__()
        self.embed_size = embed_size


        self.values = nn.Linear(self.embed_size, self.embed_size, bias=False)
        self.keys = nn.Linear(self.embed_size, self.embed_size, bias=False)
        self.queries = nn.Linear(self.embed_size, self.embed_size, bias=False)
        self.fc_out = nn.Linear(self.embed_size, self.embed_size)

    def forward(self, values, keys, query, mask):
        N = query.shape[0]
        value_len, key_len, query_len = values.shape[1], keys.shape[1], query.shape[1]

        # split embedding into self.heads pieces
        # values = values.reshape(N, value_len, self.heads, self.head_dim)
        # keys = keys.reshape(N, key_len, self.heads, self.head_dim)
        # queries = query.reshape(N, query_len, self.heads, self.head_dim)

        values = self.values(values)
        keys = self.keys(keys)
        queries = self.queries(query)

        energy = torch.mm(queries, keys.T)
        # queries shape: (N, query_len, heads, heads_dim)
        # keys shape : (N, key_len, heads, heads_dim)
        # energy shape: (N, heads, query_len, key_len)



        attention = torch.softmax(energy / (self.embed_size ** (1 / 2)), dim=1)

        out = torch.mm(attention, values)
        # attention shape: (N, heads, query_len, key_len)
        # values shape: (N, value_len, heads, heads_dim)
        # (N, query_len, heads, head_dim)

        out = self.fc_out(out)
        return out


class TransformerBlock(nn.Module):
    def __init__(self, embed_size, heads, dropout, forward_expansion):
        super(TransformerBlock, self).__init__()
        self.attention = SelfAttention(embed_size, heads)
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)

        self.feed_forward = nn.Sequential(
            nn.Linear(embed_size, forward_expansion * embed_size),
            nn.ReLU(),
            nn.Linear(forward_expansion * embed_size, embed_size)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, value, key, query, mask):
        attention = self.attention(value, key, query, mask)

        x = self.dropout(self.norm1(attention + query))
        forward = self.feed_forward(x)
        out = self.dropout(self.norm2(forward + x))
        return out


class Encoder(nn.Module):
    def __init__(
            self,
            src_vocab_size,
            embed_size,
            num_layers,
            heads,
            device,
            forward_expansion,
            dropout,
            max_length,
    ):
        super(Encoder, self).__init__()
        self.embed_size = embed_size
        self.device = device
        self.word_embedding = nn.Embedding(src_vocab_size, embed_size)
        self.position_embedding = nn.Embedding(max_length, embed_size)

        self.layers = nn.ModuleList(
            [
                TransformerBlock(
                    embed_size,
                    heads,
                    dropout=dropout,
                    forward_expansion=forward_expansion,
                )
                for _ in range(num_layers)]
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        N, seq_length = x.shape
        positions = torch.arange(0, seq_length).expand(N, seq_length).to(self.device)
        out = self.dropout(self.word_embedding(x) + self.position_embedding(positions))
        for layer in self.layers:
            out = layer(out, out, out, mask)
        return out


class stockEncoder(nn.Module):
    def __init__(self, features_in, nums_layers, embeding_size, dropout, period):
        super(stockEncoder, self).__init__()
        self.embeding_size = embeding_size
        self.nums_layers = nums_layers
        self.period = period
        self.dropout = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(self.embeding_size)
        self.norm2 = nn.LayerNorm(self.embeding_size * self.period)

        self.fc_in = nn.Linear(features_in, self.embeding_size)
        self.act1 = nn.ReLU()
        self.layers = nn.ModuleList(
            [
                SelfAttention(
                    self.embeding_size
                )
                for _ in range(self.nums_layers)]
        )
        self.feed_back1 = nn.Linear(self.embeding_size * self.period, self.embeding_size * self.period//2)
        self.feed_back2 = nn.Linear(self.embeding_size * self.period//2, 512)
        self.feed_back3 = nn.Linear(512, 256)
        self.feed_back4 = nn.Linear(256, 128)
        self.feed_back5 = nn.Linear(128, 68)
        self.feed_back6 = nn.Linear(68, 32)
        self.feed_back7 = nn.Linear(32, 16)
        self.feed_back8 = nn.Linear(16, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.act1(self.fc_in(x))
        for layer in self.layers:
            x = layer(x, x, x, mask = None)
        x = x.reshape(1, -1)
        x = self.dropout(self.norm2(x))
        x = self.act1(self.feed_back1(x))
        x = self.act1(self.feed_back2(x))
        x = self.act1(self.feed_back3(x))
        x = self.act1(self.feed_back4(x))
        x = self.act1(self.feed_back5(x))
        x = self.act1(self.feed_back6(x))
        x = self.act1(self.feed_back7(x))
        x = self.act1(self.feed_back8(x))
        out = self.sigmoid(x)

        return out


def sample(feature, index, period_len):
    x = feature[index : index + period_len, :]
    return x


if __name__ == '__main__':
    period_len = 60
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    my_model = stockEncoder(4, 6, 32, 0.3, 60).to(device)
    optimizer = torch.optim.SGD(my_model.parameters(), lr=0.01)
    criterion = torch.nn.BCELoss(size_average=False)

    ##### use my data ####
    filePath = 'selected_data'
    debug = 0
    bad_list = []

    for i in tqdm(os.listdir(filePath)):
        # try:
        name, frame = i.split('.')
        npy = np.load("selected_data/{}.npy".format(name))
        feature = npy[:, 0:4]
        y_list = npy[:, 4]
        lenl = len(y_list)
        debug += 1
        for epoch_in_a_stock in tqdm(range(lenl - period_len)):
            index = epoch_in_a_stock
            x = sample(feature, index, period_len)
            x = torch.tensor(x, dtype=torch.float).to(device)
            y_pre = my_model(x)
            torch.squeeze(y_pre)
            y = torch.tensor(y_list[index], dtype=torch.float).to(device)
            y = y.reshape(1, -1)
            loss = criterion(y_pre, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        if debug == 60:
            break
        # except RuntimeError:
            # bad_list.append(name)
            # continue
    print(bad_list)
    print(len(bad_list))


    vol = np.load('vol/CVX.npy')
    feature_vol = vol[:, 0:4]
    y_vol = torch.tensor(vol[:, 4], dtype=torch.float).to(device)
    right = 0
    for index_vol in range(len(y_vol) - period_len):
        x_vol = sample(feature_vol, index_vol, period_len)
        x_vol = torch.tensor(x_vol, dtype=torch.float).to(device)
        y_vol_pre = my_model(x_vol)
        if abs(y_vol_pre-y_vol[index]) < 0.5:
            right += 1
    print('the right rate is {}'.format(right/(len(y_vol) - period_len)))




