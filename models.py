import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from GlobalAttention import GlobalAttention


def embedded_dropout(embed, words, dropout=0.1, scale=None):
    # codes from https://github.com/salesforce/awd-lstm-lm/blob/master/embed_regularize.py
    if dropout:
        mask = embed.weight.data.new().resize_((embed.weight.size(0), 1)).bernoulli_(1 - dropout).expand_as(
            embed.weight) / (1 - dropout)
        masked_embed_weight = mask * embed.weight
    else:
        masked_embed_weight = embed.weight
    if scale:
        masked_embed_weight = scale.expand_as(masked_embed_weight) * masked_embed_weight

    padding_idx = embed.padding_idx
    if padding_idx is None:
        padding_idx = -1

    X = F.embedding(words, masked_embed_weight,
                    padding_idx, embed.max_norm, embed.norm_type,
                    embed.scale_grad_by_freq, embed.sparse
                    )
    return X


class MulEmbed(nn.Module):
    def __init__(self, loc_size, tim_size, loc_emb_size, tim_emb_size, drop_p=0.0, poi_size=None, poi_emb_size=None):
        super(MulEmbed, self).__init__()
        self.loc_size = loc_size
        self.tim_size = tim_size
        self.loc_emb_size = loc_emb_size
        self.tim_emb_size = tim_emb_size
        self.drop_p = drop_p

        self.emb_loc = nn.Embedding(self.loc_size, self.loc_emb_size, padding_idx=0)
        self.emb_tim = nn.Embedding(self.tim_size, self.tim_emb_size, padding_idx=0)
        self.init_weights()

        if poi_size is not None:
            self.poi_size = poi_size
            self.poi_emb_size = poi_emb_size
            self.emb_poi = nn.Linear(self.poi_size, self.poi_emb_size)
            self.init_weights_poi()

    def init_weights(self):
        nn.init.uniform_(self.emb_loc.weight.data, a=-0.5, b=0.5)
        nn.init.uniform_(self.emb_tim.weight.data, a=-0.5, b=0.5)
        # padding_idx=0
        self.emb_loc.weight.data[0].zero_()
        self.emb_tim.weight.data[0].zero_()

    def init_weights_poi(self):
        nn.init.uniform_(self.emb_poi.weight.data, a=-0.5, b=0.5)

    def forward(self, loc, tim, poi=None):
        # loc_emb = self.emb_loc(loc)
        # tim_emb = self.emb_tim(tim)

        loc_emb = embedded_dropout(self.emb_loc, loc, dropout=self.drop_p if self.training else 0)
        tim_emb = embedded_dropout(self.emb_tim, tim, dropout=self.drop_p if self.training else 0)
        if poi is None:
            x = torch.cat((loc_emb, tim_emb), 2)
        else:
            poi_emb = self.emb_poi(poi)
            x = torch.cat((loc_emb, tim_emb, poi_emb), 2)
        x = F.tanh(x)
        return x


class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size, mod, device, batch_size, layers):
        super(Encoder, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.mod = mod
        self.layers = layers
        self.batch_size = batch_size
        self.device = device

        if self.mod == 'GRU':
            self.rnn = nn.GRU(self.input_size, self.hidden_size, self.layers, batch_first=True)
        elif self.mod == 'LSTM':
            self.rnn = nn.LSTM(self.input_size, self.hidden_size, self.layers, batch_first=True)
        elif self.mod == 'RNN':
            self.rnn = nn.RNN(self.input_size, self.hidden_size, self.layers, batch_first=True)
        self.init_weights()

    def init_weights(self):
        """
        Keras default initialization weights
        """
        ih = (param.data for name, param in self.named_parameters() if 'weight_ih' in name)
        hh = (param.data for name, param in self.named_parameters() if 'weight_hh' in name)
        b = (param.data for name, param in self.named_parameters() if 'bias' in name)

        for t in ih:
            nn.init.xavier_uniform_(t)
        for t in hh:
            nn.init.orthogonal_(t)
        for t in b:
            nn.init.constant_(t, 0)

    def forward(self, x):
        h = torch.zeros(size=(self.layers, self.batch_size, self.hidden_size), device=self.device)
        if self.mod == 'GRU' or self.mod == 'RNN':
            states, h = self.rnn(x, h)
        elif self.mod == 'LSTM':
            c = torch.zeros(size=(self.layers, self.batch_size, self.hidden_size), device=self.device)
            states, (h, c) = self.rnn(x, (h, c))
        return states, h


class SiameseNet(nn.Module):
    def __init__(self, loc_size, tim_size, loc_emb_size, tim_emb_size, hidden_size, batch_size, device,
                 loss_mode='BCELoss', layers=1, mod='GRU', attn_mod='dot', dropout_p=0.5, fusion='S',
                 poi_size=None, poi_emb_size=None):
        super(SiameseNet, self).__init__()
        self.loc_size = loc_size
        self.tim_size = tim_size
        self.loc_emb_size = loc_emb_size
        self.tim_emb_size = tim_emb_size
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.layers = layers
        self.rnn_mod = mod
        self.device = device
        self.loss_mode = loss_mode
        self.attn_mod = attn_mod
        self.fusion = fusion

        self.dropout = nn.Dropout(p=dropout_p, inplace=False)
        self.drop_emb = 0.2

        if poi_size is None:
            self.input_size = loc_emb_size + tim_emb_size
        else:
            self.poi_size = poi_size
            self.poi_emb_size = poi_emb_size
            self.input_size = loc_emb_size + tim_emb_size + poi_emb_size

        if poi_size is None:
            self.embed = MulEmbed(self.loc_size, self.tim_size, self.loc_emb_size, self.tim_emb_size, self.drop_emb)
        else:
            self.embed = MulEmbed(self.loc_size, self.tim_size, self.loc_emb_size, self.tim_emb_size, self.drop_emb,
                                  self.poi_size, self.poi_emb_size)
        self.encoder_top = Encoder(input_size=self.input_size, hidden_size=self.hidden_size, mod=self.rnn_mod,
                                   device=self.device, batch_size=self.batch_size, layers=self.layers)
        self.encoder_down = Encoder(input_size=self.input_size, hidden_size=self.hidden_size, mod=self.rnn_mod,
                                    device=self.device, batch_size=self.batch_size, layers=self.layers)

        self.attn_top = GlobalAttention(self.hidden_size, attn_type=self.attn_mod)
        self.attn_down = GlobalAttention(self.hidden_size, attn_type=self.attn_mod)

        # E: embedding; R: RNN; C: co-attention; P:pooling;
        if self.fusion not in ["ERPC"]:
            pass
        else:
            self.fc_top = nn.Linear(self.hidden_size * 2, self.hidden_size)
            self.fc_down = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.fc_final1 = nn.Linear(self.hidden_size * 2, 50)
        self.fc_final2 = nn.Linear(50, 1)

    def sort_lens(self, lens):
        # https://github.com/facebookresearch/InferSent/blob/master/models.py#L46
        # https://discuss.pytorch.org/t/torch-from-numpy-not-support-negative-strides/3663
        lens, idx_sort = np.flip(np.sort(lens), axis=0).copy(), np.argsort(-lens)
        idx_unsort = np.argsort(idx_sort)
        # https://discuss.pytorch.org/t/runtimeerror-no-grad-accumulator-for-a-saved-leaf/17827
        idx_sort = idx_sort.to(self.device)  # .requires_grad_(True)
        idx_unsort = idx_unsort.to(self.device)
        lens = torch.from_numpy(lens).to(self.device)
        return lens, idx_sort, idx_unsort

    def forward(self, loc_top, tim_top, loc_down, tim_down, lens_top, lens_down, poi_top=None, poi_down=None):
        # remember batch first
        if poi_top is None:
            seq_top, seq_down = self.embed(loc_top, tim_top), self.embed(loc_down, tim_down)
        else:
            seq_top, seq_down = self.embed(loc_top, tim_top, poi_top), self.embed(loc_down, tim_down, poi_down)
        seq_top, seq_down = self.dropout(seq_top), self.dropout(seq_down)

        # sort by length
        lens_top, idx_sort_top, idx_unsort_top = self.sort_lens(lens_top)
        seq_top = seq_top.index_select(0, idx_sort_top)
        # Handling padding in Recurrent Networks
        seq_top_packed = nn.utils.rnn.pack_padded_sequence(seq_top, lens_top, batch_first=True)
        seq_top_out, seq_top_hidden = self.encoder_top(seq_top_packed)
        seq_top_out, _ = nn.utils.rnn.pad_packed_sequence(seq_top_out, batch_first=True)
        # U-sort
        lens_top = lens_top.index_select(0, idx_unsort_top)
        seq_top_out = seq_top_out.index_select(0, idx_unsort_top)
        seq_top_hidden = seq_top_hidden.squeeze(0)  # remove layers
        seq_top_hidden = seq_top_hidden.index_select(0, idx_unsort_top)
        seq_top_hidden = seq_top_hidden.unsqueeze(1)

        # down side
        lens_down, idx_sort_down, idx_unsort_down = self.sort_lens(lens_down)
        seq_down = seq_down.index_select(0, idx_sort_down)
        seq_down_packed = nn.utils.rnn.pack_padded_sequence(seq_down, lens_down, batch_first=True)
        seq_down_out, seq_down_hidden = self.encoder_top(seq_down_packed)
        seq_down_out, _ = nn.utils.rnn.pad_packed_sequence(seq_down_out, batch_first=True)
        lens_down = lens_down.index_select(0, idx_unsort_down)
        seq_down_out = seq_down_out.index_select(0, idx_unsort_down)
        seq_down_hidden = seq_down_hidden.squeeze(0)  # remove layers
        seq_down_hidden = seq_down_hidden.index_select(0, idx_unsort_down)
        seq_down_hidden = seq_down_hidden.unsqueeze(1)

        # co-attention
        if "C" in self.fusion:
            context_down, _ = self.attn_down(seq_top_hidden, seq_down_out, lens_down)
            context_top, _ = self.attn_top(seq_down_hidden, seq_top_out, lens_top)

        if "P" in self.fusion:
            # pooling
            lens_top2 = lens_top.to(dtype=torch.float32)
            lens_down2 = lens_down.to(dtype=torch.float32)
            lens_top2 = lens_top2.unsqueeze(1)
            lens_down2 = lens_down2.unsqueeze(1)
            seq_top_emb = torch.sum(seq_top_out, 1).squeeze(1)
            seq_top_out_pooling = seq_top_emb / lens_top2.expand_as(seq_top_emb)
            seq_down_emb = torch.sum(seq_down_out, 1).squeeze(1)
            seq_down_out_pooling = seq_down_emb / lens_down2.expand_as(seq_down_emb)

        if self.fusion in ["ERC"]:
            feature_top = context_top.squeeze(0)
            feature_down = context_down.squeeze(0)
        elif self.fusion in ["ERPC"]:
            feature_top2 = torch.cat((context_top.squeeze(0), seq_top_out_pooling), 1)
            feature_down2 = torch.cat((context_down.squeeze(0), seq_down_out_pooling), 1)
            feature_top = self.fc_top(F.selu(self.dropout(feature_top2)))
            feature_down = self.fc_down(F.selu(self.dropout(feature_down2)))
        feature_fusion = self.fc_final2(self.fc_final1(torch.cat((feature_top, feature_down), 1)))
        score = F.sigmoid(feature_fusion)

        return score, feature_top, feature_down
