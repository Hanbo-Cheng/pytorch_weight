import torch
import torch.nn as nn
from encoder import TapGRU
from decoder import Tap_gru_cond_layer
import numpy as np


def getReverse(a):
    # a = a.cpu().numpy()[::-1]
    # return torch.from_numpy(a.copy()).cuda()
    return a[range(len(a))[::-1]]


# Embedding
class My_Tap_Embedding(nn.Module):
    def __init__(self, params):
        super(My_Tap_Embedding, self).__init__()
        self.embedding = nn.Embedding(params['num_target'], params['word_dim'])

    def forward(self, params, y):
        if y.sum() < 0.:
            emb = torch.zeros(1, params['word_dim']).cuda()
        else:
            emb = self.embedding(y)
            if emb.ndim == 3:  # only for training stage
                emb_shifted = torch.zeros([emb.shape[0], emb.shape[1], params['word_dim']], dtype=torch.float32).cuda()
                emb_shifted[1:] = emb[:-1]
                emb = emb_shifted
        return emb


class Encoder_Decoder(nn.Module):
    def __init__(self, params):
        super(Encoder_Decoder, self).__init__()
        self.tap_encoder_0_1 = self.rnn = nn.LSTM(params["dim_feature"], 250, 2,
                                                  bidirectional=True)  # ->(input_size,hidden_size,num_layers)
        self.tap_encoder_2 = self.rnn = nn.LSTM(500, 250, 1,
                                                bidirectional=True)  # ->(input_size,hidden_size,num_layers)

        self.tap_encoder_3 = self.rnn = nn.LSTM(500, 250, 1,
                                                bidirectional=True)  # ->(input_size,hidden_size,num_layers)

        self.ff_state_W = nn.Linear(params['ff_state_W'][0], params['ff_state_W'][1])
        self.tap_emb_model = My_Tap_Embedding(params)
        self.tap_decoder = Tap_gru_cond_layer(params)
        self.ff_logit_lstm = nn.Linear(params["ff_logit_lstm"][0], params["ff_logit_lstm"][1])
        self.ff_logit_prev = nn.Linear(params["ff_logit_prev"][0], params["ff_logit_prev"][1])
        self.ff_logit_ctx = nn.Linear(params["ff_logit_ctx"][0], params["ff_logit_ctx"][1])
        self.ff_logit = nn.Linear(params["ff_logit"][0], params["ff_logit"][1])
        self.test_logit = nn.Linear(params["test_logit"][0], params["test_logit"][1])
        self.tap_dec_alpha_bias = torch.from_numpy(np.array(1e-20).astype('float32')).cuda()
        self.tap_a_bias = torch.from_numpy(np.array(1e-20).astype('float32')).cuda()

    def forward(self, params, tap_x, tap_x_mask, tap_a, tap_a_mask, tap_y, tap_y_mask,
                one_step=False):
        # output(seq_len, batch, hidden_size * num_directions)
        # hn(num_layers * num_directions, batch, hidden_size)
        # cn(num_layers * num_directions, batch, hidden_size)
        self.tap_encoder_0_1.flatten_parameters()
        self.tap_encoder_2.flatten_parameters()
        self.tap_encoder_3.flatten_parameters()
        h, (hn, cn) = self.tap_encoder_0_1(tap_x)

        h, (hn, cn) = self.tap_encoder_2(h)
        h = h[0::2]
        tap_x_mask = tap_x_mask[0::2]
        tap_a = tap_a[0::2]
        tap_a_mask = tap_a_mask[0::2]
        h, (hn, cn) = self.tap_encoder_3(h)
        h = h[0::2]
        tap_x_mask = tap_x_mask[0::2]
        tap_a = tap_a[0::2]
        tap_a_mask = tap_a_mask[0::2]

        tap_a = tap_a / (torch.sum(tap_a, axis=0, keepdims=True) + self.tap_a_bias)

        tap_ctx = h

        # test_logit = self.test_logit(h)
        # return test_logit

        # x_mask[:, :, None] = seq_x × batch_size × 1
        # (ctx * x_mask[:, :, None]).sum(0) = batch_size × 500
        # x_mask.sum(0)[:, None] = batch_size ×1
        tap_ctx_mean = (tap_ctx * tap_x_mask[:, :, None]).sum(0) / tap_x_mask.sum(0)[:, None]
        # init_state = batch_size × 256
        tap_init_state = torch.tanh(self.ff_state_W(tap_ctx_mean))
        # tparams['Wemb_dec'] = 111 × 256
        # y.flatten = y_length * batch_size
        # tparams['Wemb_dec'][y.flatten()] = y.flatten × 256

        # tap decoder
        tap_emb = self.tap_emb_model(params, tap_y)
        tap_proj = self.tap_decoder(params, tap_emb, mask=tap_y_mask, context=tap_ctx, context_mask=tap_x_mask,
                                    one_step=False,
                                    init_state=tap_init_state)

        tap_proj_h = tap_proj[0]
        tap_ctxs = tap_proj[1]
        tap_dec_alphas = tap_proj[2].permute(2, 1, 0) + self.tap_dec_alpha_bias
        cost_alphas = - tap_a * torch.log(tap_dec_alphas) * tap_a_mask
        logit_lstm = self.ff_logit_lstm(tap_proj_h)
        logit_prev = self.ff_logit_prev(tap_emb)
        logit_ctx = self.ff_logit_ctx(tap_ctxs)

        logit = logit_lstm + logit_prev + logit_ctx

        shape = logit.shape
        shape2 = int(shape[2] / 2)
        shape3 = 2
        logit = torch.reshape(logit, [shape[0], shape[1], shape2, shape3])
        logit = logit.max(3)[0]  # seq*batch*128
        logit = self.ff_logit(logit)

        # ***************************************************
        return logit, cost_alphas
        # return test_logit

    # decoding: encoder part
    def tap_f_init(self, params, tap_x):
        self.tap_encoder_0_1.flatten_parameters()
        self.tap_encoder_2.flatten_parameters()
        self.tap_encoder_3.flatten_parameters()
        h, (hn, cn) = self.tap_encoder_0_1(tap_x)

        h, (hn, cn) = self.tap_encoder_2(h)
        h = h[0::2]
        h, (hn, cn) = self.tap_encoder_3(h)
        h = h[0::2]
        tap_ctx = h
        tap_ctx_mean = tap_ctx.mean(0)
        tap_init_state = torch.tanh(self.ff_state_W(tap_ctx_mean))
        return tap_init_state, tap_ctx

    # decoding: decoder part
    def tap_f_next(self, params, y, tap_ctx, init_state, alpha_past):
        tap_emb = self.tap_emb_model(params, y)
        tap_proj = self.tap_decoder(params, tap_emb, context=tap_ctx,
                                    one_step=True,
                                    init_state=init_state, alpha_past=alpha_past)
        tap_next_state = tap_proj[0]
        tap_ctxs = tap_proj[1]
        next_alpha_past = tap_proj[3]
        logit_lstm = self.ff_logit_lstm(tap_next_state)
        logit_prev = self.ff_logit_prev(tap_emb)
        logit_ctx = self.ff_logit_ctx(tap_ctxs)

        logit = logit_lstm + logit_prev + logit_ctx

        shape = logit.shape
        shape1 = int(shape[1] / 2)
        shape2 = 2
        logit = torch.reshape(logit, [shape[0], shape1, shape2])
        logit = logit.max(2)[0]  # seq*batch*128

        logit = self.ff_logit(logit)
        next_probs = torch.softmax(logit, logit.ndim - 1)

        return next_probs, tap_next_state, next_alpha_past
