import torch
import torch.nn as nn


class Tap_gru_cond_layer(nn.Module):
    def __init__(self, params):
        super(Tap_gru_cond_layer, self).__init__()
        self.dim = params['tap_decoder_Wcx'][1]
        self.tap_decoder_Wc_att = nn.Linear(params['tap_decoder_Wc_att'][0], params['tap_decoder_Wc_att'][1])
        self.tap_decoder_Wx = nn.Linear(params['tap_decoder_Wx'][0], params['tap_decoder_Wx'][1])
        self.tap_decoder_W = nn.Linear(params['tap_decoder_W'][0], params['tap_decoder_W'][1])
        self.tap_decoder_Wyg = nn.Linear(params['tap_decoder_Wyg'][0], params['tap_decoder_Wyg'][1])

        self.tap_decoder_U = nn.Linear(params['tap_decoder_U'][0], params['tap_decoder_U'][1], bias=False)
        self.tap_decoder_Ux = nn.Linear(params['tap_decoder_Ux'][0],
                                        params['tap_decoder_Ux'][1], bias=False)
        self.tap_decoder_Whg = nn.Linear(params['tap_decoder_Whg'][0],
                                         params['tap_decoder_Whg'][1])
        self.tap_decoder_Umg = nn.Linear(params['tap_decoder_Umg'][0],
                                         params['tap_decoder_Umg'][1], bias=False)
        self.tap_decoder_W_comb_att = nn.Linear(params['tap_decoder_W_comb_att'][0],
                                                params['tap_decoder_W_comb_att'][1], bias=False)
        self.tap_decoder_conv_Uf = nn.Linear(params['tap_decoder_conv_Uf'][0],
                                             params['tap_decoder_conv_Uf'][1])
        self.tap_decoder_U_att = nn.Linear(params['tap_decoder_U_att'][0],
                                           params['tap_decoder_U_att'][1])
        self.tap_decoder_W_m_att = nn.Linear(params['tap_decoder_W_m_att'][0],
                                             params['tap_decoder_W_m_att'][1], bias=False)
        self.tap_decoder_U_when_att = nn.Linear(params['tap_decoder_U_when_att'][0],
                                                params['tap_decoder_U_when_att'][1])
        self.tap_decoder_U_nl = nn.Linear(params['tap_decoder_U_nl'][0],
                                          params['tap_decoder_U_nl'][1])
        self.tap_decoder_Wc = nn.Linear(params['tap_decoder_Wc'][0], params['tap_decoder_Wc'][1], bias=False)

        self.tap_decoder_Ux_nl = nn.Linear(params['tap_decoder_Ux_nl'][0],
                                           params['tap_decoder_Ux_nl'][1])

        self.tap_decoder_Wcx = nn.Linear(params['tap_decoder_Wcx'][0],
                                         params['tap_decoder_Wcx'][1], bias=False)

        self.tap_decoder_Conv2d = torch.nn.Conv2d(1, 256, (121, 1), padding=(121 // 2, 1 // 2))

    def forward(self, params, state_below, mask=None, context=None, context_mask=None, one_step=False, init_state=None,
                alpha_past=None):
        assert context != None

        if one_step:
            assert init_state is not None, 'previous state must be provided'

        nsteps = state_below.shape[0]
        if state_below.ndim == 3:
            n_samples = state_below.shape[1]
        else:
            n_samples = 1
        if mask is None:
            mask = torch.ones(state_below.shape[0]).cuda()

        if init_state is None:
            init_state = torch.zeros(n_samples, self.dim).cuda()

        assert context.ndim == 3, \
            'Context must be 3-d: #annotation x #sample x dim'

        if alpha_past is None:
            alpha_past = torch.zeros(n_samples, context.shape[0]).cuda()

        pctx_ = self.tap_decoder_Wc_att(context)

        # state_belowx = seq_y × batch_size × 256 = seq_y × batch_size × 256 dot 256 × 256
        state_belowx = self.tap_decoder_Wx(state_below)

        # state_below_ = seq_y × batch_size × 512 = seq_y × batch_size × 256 dot 256 × 512
        state_below_ = self.tap_decoder_W(state_below)

        # state_belowyg = seq_y × batch_size × 500 = seq_y × batch_size × 256 dot 256 × 500
        state_belowyg = self.tap_decoder_Wyg(state_below)

        if one_step == True:

            h2s, ctx_s, alphas, alpha_pasts, betas = self._step_slice(mask, state_below_, state_belowx, state_belowyg,
                                                                      init_state, alpha_past, pctx_, context,
                                                                      self.tap_decoder_U, self.tap_decoder_Wc,
                                                                      self.tap_decoder_W_comb_att,
                                                                      self.tap_decoder_U_att, self.tap_decoder_Ux,
                                                                      self.tap_decoder_Wcx, self.tap_decoder_U_nl,
                                                                      self.tap_decoder_Ux_nl, self.tap_decoder_Conv2d,
                                                                      self.tap_decoder_conv_Uf,
                                                                      self.tap_decoder_Whg, self.tap_decoder_Umg,
                                                                      self.tap_decoder_W_m_att,
                                                                      self.tap_decoder_U_when_att, context_mask)
            result = [h2s, ctx_s, alphas, alpha_pasts, betas]
        else:

            h2ts = torch.zeros(nsteps, n_samples, self.dim).cuda()
            cts = torch.zeros(nsteps, n_samples, context.shape[2]).cuda()
            alphas_list = (torch.zeros(nsteps, n_samples, context.shape[0])).cuda()
            alpha_pasts_list = torch.zeros(nsteps, n_samples, context.shape[0]).cuda()
            for i in range(nsteps):
                h2s, ctx_s, alphas, alpha_pasts, betas = self._step_slice(mask[i], state_below_[i], state_belowx[i],
                                                                          state_belowyg[i], init_state, alpha_past,
                                                                          pctx_, context,
                                                                          self.tap_decoder_U, self.tap_decoder_Wc,
                                                                          self.tap_decoder_W_comb_att,
                                                                          self.tap_decoder_U_att, self.tap_decoder_Ux,
                                                                          self.tap_decoder_Wcx, self.tap_decoder_U_nl,
                                                                          self.tap_decoder_Ux_nl,
                                                                          self.tap_decoder_Conv2d,
                                                                          self.tap_decoder_conv_Uf,
                                                                          self.tap_decoder_Whg, self.tap_decoder_Umg,
                                                                          self.tap_decoder_W_m_att,
                                                                          self.tap_decoder_U_when_att, context_mask)
                h2ts[i] = h2s
                init_state = h2s
                alpha_past = alpha_pasts
                cts[i] = ctx_s
                alphas_list[i] = alphas
                alpha_pasts_list[i] = alpha_pasts
            result = [h2ts, cts, alphas_list, alpha_pasts_list]
        return result

    def _slice(self, _x, n, dim):
        if _x.ndim == 3:
            return _x[:, :, n * dim:(n + 1) * dim]
        return _x[:, n * dim:(n + 1) * dim]

    def _step_slice(self, m_, x_, xx_, yg, h_, alpha_past_, pctx_, cc_,
                    U, Wc, W_comb_att, U_att, Ux, Wcx, U_nl, Ux_nl, conv_Q, conv_Uf,
                    Whg, Umg, W_m_att, U_when_att, context_mask):
        # preact1 = batch_size × 512
        preact1 = U(h_)
        preact1 = preact1 + x_
        preact1 = torch.sigmoid(preact1)

        r1 = self._slice(preact1, 0, self.dim)  # reset gate
        u1 = self._slice(preact1, 1, self.dim)  # update gate

        # preact1 = batch_size × 256
        preactx1 = Ux(h_)
        preactx1 = r1 * preactx1
        preactx1 = preactx1 + xx_

        h1 = torch.tanh(preactx1)

        # h1 = batch_size × 256

        h1 = u1 * h_ + (1. - u1) * h1
        h1 = m_[:, None] * h1 + (1. - m_)[:, None] * h_

        # gm = batch_size × 500 = batch_size × 256 dot 256 × 500
        g_m = Whg(h_)
        g_m = yg + g_m
        g_m = torch.sigmoid(g_m)
        # mt = batch_size × 500 = batch_size × 256 dot 256 × 500
        mt = Umg(h1)
        mt = torch.tanh(mt)
        mt = g_m * mt
        # attention
        # pstate_ = batch_size × 500
        pstate_ = W_comb_att(h1)

        # converage vector
        # batch_size × in_chancel × height × width  过卷积 out_channel × in_channel × height × width
        # batch_size × 1 × seq_x × 1 过卷积 256 × 1 × 121 × 1
        # cover_F =  batch_size × 256 × seq_x × 1

        cover_F = conv_Q(alpha_past_[:, None, :, None])

        cover_F = cover_F.permute(1, 2, 0, 3)  # dim(256) x seq_x x batch_size x 1
        # cover_F = # dim(256) x seq_x x batch_size
        cover_F = cover_F.reshape([cover_F.shape[0], cover_F.shape[1], cover_F.shape[2]])
        assert cover_F.ndim == 3, \
            'Output of conv must be 3-d: #dim x SeqL x batch'
        # cover_F = cover_F[:,pad:-pad,:]
        # cover_F = # seq_x × batch_size × dim(256)
        cover_F = cover_F.permute(1, 2, 0)
        # cover_F must be SeqL x batch x dimctx
        # cover_vector = Seqx x batch x 500
        cover_vector = conv_Uf(cover_F)
        # cover_vector = cover_vector * context_mask[:,:,None]

        # seq_x × batch_size(8) × 500 + 1 × batch_size(8) × 500 + Seqx x batch x 500
        pctx__ = pctx_ + pstate_[None, :, :] + cover_vector
        # pctx__ += xc_
        pctx__ = torch.tanh(pctx__)
        # alpha = seq_x × batch_size × 1
        alpha = U_att(pctx__)
        # compute alpha_when
        # pctx_when = batch_size × 500
        pctx_when = W_m_att(mt)
        # pstate_ = batch_size × 500
        pctx_when = pstate_ + pctx_when
        pctx_when = torch.tanh(pctx_when)
        # alpha_when = batch_size × 1
        alpha_when = U_when_att(pctx_when)  # batch * 1

        # alpha = Seq_x × batch
        alpha = alpha.reshape([alpha.shape[0], alpha.shape[1]])  # Seq_x * batch
        # alpha = Seq_x × batch
        alpha = torch.exp(alpha)
        alpha_when = torch.exp(alpha_when)
        if context_mask is not None:
            alpha = alpha * context_mask
        if context_mask is not None:
            alpha_mean = alpha.sum(0, keepdims=True) / context_mask.sum(0, keepdims=True)
        else:
            # alpha_mean = 1 × batch_size
            alpha_mean = alpha.mean(0, keepdims=True)
        # alpha_when = (1+1)×batch
        alpha_when = torch.cat([alpha_mean, alpha_when.T], axis=0)  # (SeqL+1)*batch
        # alpha = Seq_x × batch
        alpha = alpha / alpha.sum(0, keepdims=True)
        # 2 × batch_size
        alpha_when = alpha_when / alpha_when.sum(0, keepdims=True)
        # beta = batch_size
        beta = alpha_when[-1, :]
        # alpha_past = batch × Seql
        alpha_past = alpha_past_ + alpha.T
        # ctx_ = batch_size(8) × 500 = (seq_x × batch_size(8) × 500 * seq_x ×batch_size × 1).sum(0)
        ctx_ = (cc_ * alpha[:, :, None]).sum(0)  # current context
        # batch_size × 1 * batch_size × 500 + ...
        # ctx_ = batch_size × 500
        ctx_ = beta[:, None] * mt + (1. - beta)[:, None] * ctx_

        # preact2 = batch_size × 512 = batch_size × 256 * 256 × 512
        preact2 = U_nl(h1)
        # preact2 = batch_size × 512 = batch_size × 500 * 500 × 512
        preact2 = preact2 + Wc(ctx_)
        preact2 = torch.sigmoid(preact2)

        r2 = self._slice(preact2, 0, self.dim)
        u2 = self._slice(preact2, 1, self.dim)
        # preactx2 = batch_size × 256 = batch_size × 256 * 256 × 256
        preactx2 = Ux_nl(h1)
        preactx2 = r2 * preactx2
        # preactx2 += batch_size × 256 = batch_size × 500 * 500 × 256
        preactx2 = Wcx(ctx_) + preactx2

        h2 = torch.tanh(preactx2)

        h2 = u2 * h1 + (1. - u2) * h2
        h2 = m_[:, None] * h2 + (1. - m_)[:, None] * h1

        return h2, ctx_, alpha.T, alpha_past, beta  # pstate_, preact, preactx, r, u
