import copy
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.lockedropout import LockedDropout

from all_packages import *


def stable_masked_softmax(x, mask):
    x = x - x.max()
    x = x.masked_fill_(mask[:,None] == 0, float("-inf"))
    x = torch.softmax(x, -1)

    return x


class SelfAttention(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.input_linear = nn.Linear(input_size, 1, bias=False)
        self.dot_scale = nn.Parameter(torch.Tensor(input_size).uniform_(1.0 / (input_size ** 0.5)))

        self.ff1 = nn.Sequential(
            nn.Linear(120, 120),
            nn.Tanh(),
            nn.LayerNorm(120)
        )
        self.ff2 = nn.Sequential(
            nn.Linear(120, 120),
            nn.Tanh(),
            nn.LayerNorm(120)
        )

    def forward(self, input, memory, mask):

        input_dot = self.input_linear(input)  # nan: cal the weight for the same word

        inp = self.ff1(input * self.dot_scale)
        memory_ = self.ff2(memory).permute(0, 2, 1).contiguous()
        cross_dot = inp @ memory_

        att = input_dot + cross_dot

        ## Numerically stable version
        weight_one = stable_masked_softmax(att, mask)

        # att = att - 1e30 * (1 - mask[:,None])
        # weight_one = F.softmax(att, dim=-1)

        if NaNReporter.check_abnormal(weight_one, "weight_one"):
            print(f"att: max: {att.max()} - min: {att.min()}")

        output_one = torch.bmm(weight_one, memory)

        return torch.cat([input, output_one], dim=-1)


class BiAttention(nn.Module):
    def __init__(self, input_size, dropout):
        super().__init__()
        self.dropout = LockedDropout(dropout)
        self.input_linear = nn.Linear(input_size, 1, bias=False)
        self.memory_linear = nn.Linear(input_size, 1, bias=False)

        self.dot_scale = nn.Parameter(torch.Tensor(input_size).uniform_(1.0 / (input_size ** 0.5)))

    def forward(self, input, memory, mask):
        bsz, input_len, memory_len = input.size(0), input.size(1), memory.size(1)

        input = self.dropout(input)
        memory = self.dropout(memory)

        input_dot = self.input_linear(input)
        memory_dot = self.memory_linear(memory).view(bsz, 1, memory_len)
        cross_dot = torch.bmm(input * self.dot_scale, memory.permute(0, 2, 1).contiguous())
        att = input_dot + memory_dot + cross_dot
        att = att - 1e30 * (1 - mask[:, None])

        weight_one = F.softmax(att, dim=-1)
        output_one = torch.bmm(weight_one, memory)
        weight_two = F.softmax(att.max(dim=-1)[0], dim=-1).view(bsz, 1, input_len)
        output_two = torch.bmm(weight_two, input)

        return torch.cat([input, output_one, input * output_one, output_two * output_one], dim=-1)


def attention(query, key, mask=None, dropout=None):
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)

    p_attn = F.softmax(scores, dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)

    return p_attn


def clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class MultiHeadAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        assert d_model % h == 0

        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 2)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, mask=None):
        if mask is not None:
            mask = mask.unsqueeze(1).unsqueeze(1)

        nbatches = query.size(0)

        query, key = [
            l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
            for l, x in zip(self.linears, (query, key))
        ]
        attn = attention(query, key, mask=mask, dropout=self.dropout)

        return attn
