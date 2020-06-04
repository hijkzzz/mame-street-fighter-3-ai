import torch
import torch.nn as nn

import math


class GRUCell(nn.Module):
    """
    An implementation of GRUCell.

    """
    def __init__(self, input_size, hidden_size, bias=True):
        super(GRUCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        self.x2h = nn.Linear(input_size, 3 * hidden_size, bias=bias)
        self.h2h = nn.Linear(hidden_size, 3 * hidden_size, bias=bias)
        self.reset_parameters()

    def reset_parameters(self):
        std = 1.0 / math.sqrt(self.hidden_size)
        for w in self.parameters():
            w.data.uniform_(-std, std)

    def forward(self, x, hidden):

        gate_x = self.x2h(x)
        gate_h = self.h2h(hidden)
        
        i_r, i_i, i_n = gate_x.chunk(3, 2)
        h_r, h_i, h_n = gate_h.chunk(3, 2)

        resetgate = torch.sigmoid(i_r + h_r)
        inputgate = torch.sigmoid(i_i + h_i)
        newgate = torch.tanh(i_n + resetgate * h_n)

        hy = (1 - inputgate) * newgate + inputgate * hidden
        return hy


class GRU(nn.Module):
    def __init__(self, input_dim, hidden_dim, bias=True):
        super(GRU, self).__init__()

        self.hidden_dim = hidden_dim
        self.gru_cell = GRUCell(input_dim, hidden_dim, bias)

    def forward(self, x, hn=None, masks=None):
        if hn is None:
            hn = torch.zeros(1, x.size(1), self.hidden_dim).to(device=x.device)

        outs = []
        for seq in range(x.size(0)):
            hn = self.gru_cell(x[seq].unsqueeze(0), hn)
            outs.append(hn)

            if not masks is None:
                hn *= masks[seq].view(-1, 1)

        return torch.cat(outs), hn