import time
import os
import re
import numpy as np
import random
import torch
from torch import optim, nn
from utils import dataIterator, load_dict, prepare_data, gen_sample, weight_init
from tap_nmt import tap_dataIterator, tap_dataIterator_valid, tap_prepare_data
from encoder_decoder import Encoder_Decoder

print("GPU可用否")
print(torch.cuda.is_available())

# whether use multi-GPUs
multi_gpu_flag = False
# whether init params
init_param_flag = True

# load configurations
# paths
dictionaries = ['./data/dictionary.txt']
valid_output = ['./result/valid_decode_result.txt']
valid_result = ['./result/valid.wer']

tap_datasets = ["./data/online-train.pkl",
                "./data/train_caption.txt",
                "./data/align-online-train.pkl"]
tap_valid_datasets = ['./data/online-test.pkl',
                      './data/test_caption.txt']

saveto = r'./result/TAP_params_assemble1.pkl'

# training settings
if multi_gpu_flag:
    batch_size = 24
    valid_batch_size = 24
else:
    batch_size = 8
    valid_batch_size = 8

maxlen = 200
max_epochs = 5000
start_lr = 3
my_eps = 1e-8
decay_c = 0.0005
clip_c = 1000.
RELOAD = True
# early stop
estop = False
halfLrFlag = 0
bad_counter = 0
patience = 15
finish_after = 10000000

# model architecture
params = {}
params['n'] = 256
params['m'] = 256
params['dim_attention'] = 512
params['D'] = 684
params['K'] = 111
params['growthRate'] = 24
params['reduction'] = 0.5
params['bottleneck'] = True
params['use_dropout'] = True
params['input_channels'] = 1
# TAP model ---encoder
params["tap_gru_U"] = [250, 500]
params["tap_gru_Ux"] = [250, 250]
params["tap_gru_W0"] = [9, 500]
params["tap_gru_Wx0"] = [9, 250]
params["tap_gru_W"] = [500, 500]
params["tap_gru_Wx"] = [500, 250]
params["hidden_size"] = [250, 250, 250, 250]
params["down_sample"] = [0, 0, 1, 1]
params['dim_feature'] = 9
params["ff_state_W"] = [500, 256]
params['num_target'] = 111
params['word_dim'] = 256
params['tap_decoder_Wcx'] = [500, 256]
params['tap_decoder_Wc_att'] = [500, 500]
params['tap_decoder_Wx'] = [256, 256]
params['tap_decoder_W'] = [256, 512]
params['tap_decoder_Wyg'] = [256, 500]
params['tap_decoder_U'] = [256, 512]
params['tap_decoder_Ux'] = [256, 256]
params['tap_decoder_Whg'] = [256, 500]
params['tap_decoder_Umg'] = [256, 500]
params['tap_decoder_W_comb_att'] = [256, 500]
params['tap_decoder_conv_Uf'] = [256, 500]
params['tap_decoder_U_att'] = [500, 1]
params['tap_decoder_W_m_att'] = [500, 500]
params['tap_decoder_U_when_att'] = [500, 1]
params['tap_decoder_U_nl'] = [256, 512]
params['tap_decoder_Wc'] = [500, 512]
params['tap_decoder_Ux_nl'] = [256, 256]

params["ff_logit_lstm"] = [256, 256]
params["ff_logit_prev"] = [256, 256]
params["ff_logit_ctx"] = [500, 256]
params["ff_logit"] = [128, params['num_target']]
params["test_logit"] = [500, params['num_target']]
params['gamma'] = 0.1
params["model_cost_coeff"] = 0.1
# load dictionary
worddicts = load_dict(dictionaries[0])
worddicts_r = [None] * len(worddicts)
for kk, vv in worddicts.items():
    worddicts_r[vv] = kk

# ******************************************************************************************************************
tap_train, tap_num_batch = tap_dataIterator(tap_datasets[0], tap_datasets[1], tap_datasets[2],
                                            worddicts,
                                            batch_size=batch_size, maxlen=maxlen)

tap_valid, tap_valid_uid_list = tap_dataIterator_valid(tap_valid_datasets[0], tap_valid_datasets[1],
                                                       worddicts,
                                                       batch_size=valid_batch_size, maxlen=maxlen)
# ******************************************************************************************************************

# display
uidx = 0  # count batch
loss_s = 0.  # count loss
ud_s = 0  # time for training an epoch
validFreq = -1
saveFreq = -1
sampleFreq = -1
dispFreq = 100
if validFreq == -1:
    validFreq = len(tap_train)
if saveFreq == -1:
    saveFreq = len(tap_train)
if sampleFreq == -1:
    sampleFreq = len(tap_train)

# initialize model
TAP_model = Encoder_Decoder(params)
if RELOAD == True:
    TAP_model.load_state_dict(
        torch.load(saveto, map_location=lambda storage, loc: storage))
else:
    if init_param_flag:
        TAP_model.apply(weight_init)
    if multi_gpu_flag:
        TAP_model = nn.DataParallel(TAP_model, device_ids=[0, 1])
TAP_model.cuda()

tparams_p_u = []
tparams_p_ls2 = []
running_grads2_miu = []
running_up2_miu = []
running_grads2_sigma = []
running_up2_sigma = []
# print model's parameters
model_params = TAP_model.named_parameters()

for k, v in model_params:
    print(k)

len_model_params = 0
for i, param in enumerate(TAP_model.parameters()):
    if param.grad != None:
        running_grads2_miu.append(torch.zeros_like(param.grad.data))
        running_grads2_sigma.append(torch.zeros_like(param.grad.data))
    else:
        running_grads2_miu.append(None)
        running_grads2_sigma.append(None)
    running_up2_sigma.append(torch.zeros_like(param.data))
    running_up2_miu.append(torch.zeros_like(param.data))
    len_model_params += 1

print("参数数量：", len_model_params)
# loss function
criterion = torch.nn.CrossEntropyLoss(reduce=False)


def getTparams():
    with torch.no_grad():
        for i, param in enumerate(TAP_model.parameters()):
            init_sigma = 1.0e-12
            log_sigma_scale = 2048.0
            p_u = torch.ones_like(param.data) * param.data
            p_ls2 = torch.zeros_like(param.data) + np.log(init_sigma) * 2. / log_sigma_scale  # log_(sigma^2)
            tparams_p_u.append(p_u)
            tparams_p_ls2.append(p_ls2)


getTparams()


# 首先该函数将tparams_p_u与tparams_p_ls2使用当前weight填好，并将tparams_p_ls2应用于当前weight
# 其次产生Beta与prior_s2供后续new_grads_miu与new_grads_sigma使用
def f_apply_noise_to_weight():
    with torch.no_grad():
        log_sigma_scale = 2048.0
        #  compute the prior mean and variation
        temp_sum = 0.0
        temp_param_count = 0.0

        for i, param in enumerate(TAP_model.parameters()):
            temp_sum = temp_sum + tparams_p_u[i].sum()
            temp_param_count = temp_param_count + np.prod(np.array(list(tparams_p_u[i].shape)).astype("float32"))

            # param.data += torch.normal(0, 1, param.data.shape).cuda() * torch.exp(
            #     tparams_p_ls2[i] * log_sigma_scale) ** 0.5
            # add noise to weight
        prior_u = float(temp_sum) / temp_param_count
        temp_sum = 0.0
        for i, p_u in enumerate(tparams_p_u):
            p_s2 = torch.exp(tparams_p_ls2[i] * log_sigma_scale)  # sigma^2
            temp_sum = temp_sum + (p_s2).sum() + (((p_u - prior_u) ** 2).sum())

        prior_s2 = float(temp_sum) / temp_param_count
        return prior_u, prior_s2


def f_copy_weight():
    with torch.no_grad():
        # restore weight
        for i, param in enumerate(TAP_model.parameters()):
            param.data = tparams_p_u[i]


def f_update_miu(tparams_miu, grads_miu):
    with torch.no_grad():
        for i, param in enumerate(TAP_model.parameters()):
            if grads_miu[i] != None:
                if running_grads2_miu[i] == None:
                    temp_running_grads2_miu = 0
                else:
                    temp_running_grads2_miu = running_grads2_miu[i]
                running_grads2_miu[i] = temp_running_grads2_miu * 0.95 + 0.05 * grads_miu[i] ** 2
                temp_updir_miu = - (running_up2_miu[i] + my_eps) ** 0.5 / (running_grads2_miu[i] + my_eps) ** 0.5 * \
                                 grads_miu[i]
                running_up2_miu[i] = running_up2_miu[i] * 0.95 + 0.05 * temp_updir_miu ** 2
                # print(temp_updir_miu.shape)
                assert temp_updir_miu.shape == tparams_miu[i].shape
                # print(temp_updir_miu)
                # print(tparams_miu[i])
                tparams_miu[i] += temp_updir_miu
                del temp_updir_miu
        # print(i)
        assert i == len_model_params - 1
        del grads_miu[i]
        torch.cuda.empty_cache()


def f_update_sigma(tparams_sigma, grads_sigma):
    with torch.no_grad():
        for i, param in enumerate(TAP_model.parameters()):
            if grads_sigma[i] != None:
                if running_grads2_sigma[i] == None:
                    temp_running_grads2_sigma = 0
                else:
                    temp_running_grads2_sigma = running_grads2_sigma[i]
                running_grads2_sigma[i] = temp_running_grads2_sigma * 0.95 + 0.05 * grads_sigma[i] ** 2
                temp_updir_sigma = - (running_up2_sigma[i] + my_eps) ** 0.5 / (running_grads2_sigma[i] + my_eps) ** 0.5 * \
                                   grads_sigma[i]
                running_up2_sigma[i] = running_up2_sigma[i] * 0.95 + 0.05 * temp_updir_sigma ** 2
                # print(temp_updir_miu.shape)
                assert temp_updir_sigma.shape == tparams_sigma[i].shape
                tparams_sigma[i] += temp_updir_sigma
                del temp_updir_sigma
        # print(i)
        assert i == len_model_params - 1
        del grads_sigma[i]
        torch.cuda.empty_cache()


# def f_update(tparams, grads, optimizer):
#     for i, param in enumerate(TAP_model.parameters()):
#         param.data = tparams[i]
#         if grads[i] != None:
#             param.grad.data = grads[i]
#     if clip_c > 0.:
#         torch.nn.utils.clip_grad_norm_(TAP_model.parameters(), clip_c)
#
#     for i, param in enumerate(TAP_model.parameters()):
#         tparams[i] = param.data


def produceGrad(prior_u, prior_s2, num_examples):
    with torch.no_grad():
        log_sigma_scale = 2048.0
        model_cost_coefficient = params['model_cost_coeff']
        new_grads_miu = []
        new_grads_sigma = []
        for p_u, p_ls2, param in zip(tparams_p_u, tparams_p_ls2, TAP_model.parameters()):
            if param.grad != None:
                p_s2 = torch.exp(p_ls2 * log_sigma_scale)  # sigma^2
                p_u_grad = (model_cost_coefficient * (p_u - prior_u) /
                            (num_examples * prior_s2) + param.grad.data)
                a = np.float32(model_cost_coefficient *
                               0.5 / num_examples * log_sigma_scale) * (p_s2 / prior_s2 - 1.0)
                b = (0.5 * log_sigma_scale) * p_s2 * (param.grad.data ** 2)
                p_ls2_grad = (a + b)
                # print("p_ls2_grad", param.grad.data)
            else:
                p_u_grad = None
                p_ls2_grad = None
            new_grads_miu.append(p_u_grad)
            new_grads_sigma.append(p_ls2_grad)
        return new_grads_miu, new_grads_sigma


print('Optimization')

# statistics
history_errs = []

for eidx in range(max_epochs):
    n_samples = 0
    ud_epoch = time.time()
    random.shuffle(tap_train)
    lr = start_lr * (0.97 ** eidx)
    if lr < .65:
        lr = .65
    for tap_x, tap_y, tap_a in tap_train:
        TAP_model.train()
        ud_start = time.time()
        n_samples += len(tap_x)
        uidx += 1
        tap_x, tap_x_mask, tap_y, tap_y_mask, tap_a, tap_a_mask = tap_prepare_data(
            params, tap_x, tap_y, tap_a, maxlen=maxlen)
        tap_x = torch.from_numpy(tap_x).cuda()
        tap_x_mask = torch.from_numpy(tap_x_mask).cuda()
        # batch_size × seq_y
        tap_y = torch.from_numpy(tap_y).cuda()
        tap_y_mask = torch.from_numpy(tap_y_mask).cuda()
        tap_a = torch.from_numpy(tap_a).cuda()
        tap_a_mask = torch.from_numpy(tap_a_mask).cuda()

        # 一,tparams_p_u,tparams_p_ls2只因被初始化一次，但prior_u,prior_s2,Beta需要被重新计算
        prior_u, prior_s2 = f_apply_noise_to_weight()
        # forward
        tap_ctx, cost_alphas = TAP_model(params, tap_x, tap_x_mask, tap_a, tap_a_mask, tap_y,
                                         tap_y_mask)

        # tap_y = torch.from_numpy(
        #     params["num_target"] * np.random.rand(tap_ctx.shape[0], tap_ctx.shape[1]).astype('int64')).cuda()
        # tap_y_mask = torch.ones(tap_ctx.shape[0], tap_ctx.shape[1]).cuda()
        tap_ctx = torch.reshape(tap_ctx, [-1, tap_ctx.shape[2]])

        # loss = criterion(scores, y.view(-1))
        loss = criterion(tap_ctx, torch.reshape(tap_y, [-1]))
        # loss:seq_y × batch_size
        loss = torch.reshape(loss, [tap_y.shape[0], tap_y.shape[1]])
        # loss:1 × batch_size
        # loss = ((loss * tap_y_mask).sum(0) + params['gamma'] * cost_alphas.sum(0).sum(1)) / tap_y_mask.sum(0)
        loss = ((loss * tap_y_mask).sum(0) + params['gamma'] * cost_alphas.sum(0).sum(1))
        # loss = (loss * tap_y_mask).sum(0)
        # loss = (loss * tap_y_mask).sum(0) / tap_y_mask.sum(0)
        loss = loss.mean()
        loss_s += loss.item()

        # backward
        TAP_model.zero_grad()

        loss.backward()
        # 二
        new_grads_miu, new_grads_sigma = produceGrad(prior_u, prior_s2, 2 * 8835)

        # apply gradient clipping here
        # if clip_c > 0.:
        #     g2 = 0.
        #     for param in TAP_model.parameters():
        #         g2 += (param.grad.data ** 2).sum()
        #     new_grads = []
        #     for g in TAP_model.parameters():
        #         if g2 > (clip_c ** 2):
        #             new_grads.append(g / g2 ** 0.5 * clip_c)
        #         else:
        #             new_grads.append(g)
        #     grads = new_grads

        # update

        # for nm in TAP_model.parameters():
        #     if nm != None:
        #         print("before", nm.data)
        #         break
        f_update_miu(tparams_p_u, new_grads_miu)

        f_update_sigma(tparams_p_ls2, new_grads_sigma)

        # for nm in TAP_model.parameters():
        #     if nm != None:
        #         print("after", nm.data)
        #         break

        f_copy_weight()

        ud = time.time() - ud_start
        ud_s += ud

        # display
        if np.mod(uidx, dispFreq) == 0:
            ud_s /= 60.
            loss_s /= dispFreq
            print('Epoch ', eidx, 'Update ', uidx, 'Cost ', loss_s, 'UD ', ud_s, 'lrate ', lr, 'eps', my_eps,
                  'bad_counter', bad_counter)
            ud_s = 0
            loss_s = 0.

        # validation
        valid_stop = False
        if np.mod(uidx, sampleFreq) == 0:
            # if True:
            TAP_model.eval()
            with torch.no_grad():
                fpp_sample = open(valid_output[0], 'w')
                valid_count_idx = 0
                for tap_x, tap_y in tap_valid:
                    for tap_xx in tap_x:
                        tap_xx_pad = np.zeros((tap_xx.shape[0] + 1, tap_xx.shape[1]), dtype='float32')
                        tap_xx_pad[:tap_xx.shape[0], :] = tap_xx
                        tap_xx_pad = torch.from_numpy(tap_xx_pad).cuda()
                        stochastic = False
                        # print(tap_xx_pad.shape)
                        sample, score = gen_sample(TAP_model, tap_xx_pad[:, None, :], params, multi_gpu_flag, k=10,
                                                   maxlen=1000,
                                                   stochastic=stochastic,
                                                   argmax=False)

                        if len(score) == 0:
                            print('valid decode error happens')
                            valid_stop = True
                            break
                        score = score / np.array([len(s) for s in sample])
                        ss = sample[score.argmin()]
                        # write decoding results
                        fpp_sample.write(tap_valid_uid_list[valid_count_idx])
                        valid_count_idx = valid_count_idx + 1
                        # symbols (without <eos>)
                        for vv in ss:
                            if vv == 0:  # <eos>
                                break
                            fpp_sample.write(' ' + worddicts_r[vv])
                        fpp_sample.write('\n')
                        # print(str(valid_count_idx) + "预测完毕")
                    if valid_stop:
                        break
            fpp_sample.close()
            print('valid set decode done')
            ud_epoch = (time.time() - ud_epoch) / 60.
            print('epoch cost time ... ', ud_epoch)

        # calculate wer and expRate
        if np.mod(uidx, validFreq) == 0 and valid_stop == False:
            os.system('python compute-wer.py ' + valid_output[0] + ' ' + tap_valid_datasets[
                1] + ' ' + valid_result[0])

            fpp = open(valid_result[0])
            stuff = fpp.readlines()
            fpp.close()
            m = re.search('WER (.*)\n', stuff[0])
            valid_err = 100. * float(m.group(1))
            m = re.search('ExpRate (.*)\n', stuff[1])
            valid_sacc = 100. * float(m.group(1))
            valid_err = 0.6 * valid_err + 0.4 * (100. - valid_sacc)
            history_errs.append(valid_err)

            # the first time validation or better model
            if uidx // validFreq == 0 or valid_err <= np.array(history_errs).min():
                bad_counter = 0
                print('Saving model params ... ')
                if multi_gpu_flag:
                    torch.save(TAP_model.module.state_dict(), saveto)
                else:
                    torch.save(TAP_model.state_dict(), saveto)

            # worse model
            if uidx / validFreq != 0 and valid_err > np.array(history_errs).min():
                bad_counter += 1
                if bad_counter > patience:
                    if halfLrFlag == 2:
                        print('Early Stop!')
                        estop = True
                        break
                    else:
                        print('Lr decay and retrain!')
                        bad_counter = 0
                        my_eps = my_eps / 10.
                        halfLrFlag += 1
            print('Valid WER: %.2f%%, ExpRate: %.2f%%' % (valid_err, valid_sacc))

        # finish after these many updates
        if uidx >= finish_after:
            print('Finishing after %d iterations!' % uidx)
            estop = True
            break

    print('Seen %d samples' % n_samples)

    # early stop
    if estop:
        break
