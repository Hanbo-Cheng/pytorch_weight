import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# DenseNet-B
class Bottleneck(nn.Module):
    def __init__(self, nChannels, growthRate, use_dropout):
        super(Bottleneck, self).__init__()
        interChannels = 4 * growthRate
        self.bn1 = nn.BatchNorm2d(interChannels)
        self.conv1 = nn.Conv2d(nChannels, interChannels, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(growthRate)
        self.conv2 = nn.Conv2d(interChannels, growthRate, kernel_size=3, padding=1, bias=False)
        self.use_dropout = use_dropout
        self.dropout = nn.Dropout(p=0.2)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        if self.use_dropout:
            out = self.dropout(out)
        out = F.relu(self.bn2(self.conv2(out)), inplace=True)
        if self.use_dropout:
            out = self.dropout(out)
        out = torch.cat((x, out), 1)
        return out


# single layer
class SingleLayer(nn.Module):
    def __init__(self, nChannels, growthRate, use_dropout):
        super(SingleLayer, self).__init__()
        self.bn1 = nn.BatchNorm2d(nChannels)
        self.conv1 = nn.Conv2d(nChannels, growthRate, kernel_size=3, padding=1, bias=False)
        self.use_dropout = use_dropout
        self.dropout = nn.Dropout(p=0.2)

    def forward(self, x):
        out = self.conv1(F.relu(x, inplace=True))
        if self.use_dropout:
            out = self.dropout(out)
        out = torch.cat((x, out), 1)
        return out


# transition layer
class Transition(nn.Module):
    def __init__(self, nChannels, nOutChannels, use_dropout):
        super(Transition, self).__init__()
        self.bn1 = nn.BatchNorm2d(nOutChannels)
        self.conv1 = nn.Conv2d(nChannels, nOutChannels, kernel_size=1, bias=False)
        self.use_dropout = use_dropout
        self.dropout = nn.Dropout(p=0.2)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        if self.use_dropout:
            out = self.dropout(out)
        out = F.avg_pool2d(out, 2, ceil_mode=True)
        return out


class TapGRU(nn.Module):
    def __init__(self, params, suffix=""):
        super(TapGRU, self).__init__()
        self.dim = params["tap_gru_Ux"][1]
        self.tap_gru_W = nn.Linear(params['tap_gru_W' + suffix][0], params['tap_gru_W' + suffix][1])
        self.tap_gru_Wx = nn.Linear(params['tap_gru_Wx' + suffix][0], params['tap_gru_Wx' + suffix][1])

        self.tap_gru_U = nn.Linear(params['tap_gru_U'][0], params['tap_gru_U'][1], bias=False)
        self.tap_gru_Ux = nn.Linear(params['tap_gru_Ux'][0], params['tap_gru_Ux'][1], bias=False)

    def forward(self, params, state_below, mask=None):
        nsteps = state_below.shape[0]
        if state_below.ndim == 3:
            n_samples = state_below.shape[1]
        else:
            n_samples = 1

        if mask is None:
            mask = torch.ones(state_below.shape[0], 1).cuda()

        state_below_ = self.tap_gru_W(state_below)
        state_belowx = self.tap_gru_Wx(state_below)
        init_states = torch.zeros(n_samples, self.dim).cuda()
        result = torch.zeros(nsteps, n_samples, self.dim).cuda()

        for i in range(nsteps):
            init_states = self._step_slice(mask[i], state_below_[i], state_belowx[i], init_states, self.tap_gru_U,
                                           self.tap_gru_Ux)
            result[i] = init_states
        return result

    def _slice(self, _x, n, dim):
        if _x.ndim == 3:
            return _x[:, :, n * dim:(n + 1) * dim]
        return _x[:, n * dim:(n + 1) * dim]

    # step function to be used by scan
    # arguments    | sequences |outputs-info| non-seqs
    # x_ = 8 × 500
    # xx_ = 8 × 250
    # h_ = 8 × 250
    # U = 250 × 500
    # Ux = 250× 250
    def _step_slice(self, m_, x_, xx_, h_, U, Ux):
        preact = U(h_)
        preact = preact + x_

        # reset and update gates
        r = torch.sigmoid(self._slice(preact, 0, self.dim))
        u = torch.sigmoid(self._slice(preact, 1, self.dim))

        # compute the hidden state proposal
        preactx = Ux(h_)
        preactx = preactx * r
        preactx = preactx + xx_

        # hidden state proposal
        h = torch.tanh(preactx)

        # leaky integrate and obtain next hidden state
        h = u * h_ + (1. - u) * h
        h = m_[:, None] * h + (1. - m_)[:, None] * h_

        return h
