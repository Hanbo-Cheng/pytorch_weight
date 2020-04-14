'''
使用weight_noise不需要模型带有额外优化器，该算法自带adadelta优化器，即使用该算法，原模型优化器需要移除！
实现了weight_noise所有方法类，共有4个函数需要调用
1.init_weight_noise_class()，即构造函数，在初始化完完整模型后调用，example:
==================================================================
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

weight_noise = weight_noise_class(TAP_model)
==================================================================
2.f_apply_noise_to_weight，在前向反馈前使用，为原weight增加噪声，example:
==================================================================
for tap_x, tap_y, tap_a in tap_train:
    TAP_model.train()
    tap_x, tap_x_mask, tap_y, tap_y_mask, tap_a, tap_a_mask = tap_prepare_data(
        params, tap_x, tap_y, tap_a, maxlen=maxlen)

    # 一,tparams_p_u,tparams_p_ls2只因被初始化一次，但prior_u,prior_s2,Beta需要被重新计算
    prior_u, prior_s2 = weight_noise_class.f_apply_noise_to_weight(TAP_model)
    # forward
    tap_ctx, cost_alphas = TAP_model(params, tap_x, tap_x_mask, tap_a, tap_a_mask, tap_y,
                                     tap_y_mask)
==================================================================
3.produceGrad，在损失回传完毕后调用，产生新的梯度，model_cost_coefficient为新添梯度的权重，num_examples为训练样本数*2，example:
==================================================================
TAP_model.zero_grad()
loss.backward()
new_grads_miu, new_grads_sigma = weight_noise.produceGrad(TAP_model, prior_u, prior_s2, 2 * 8835)
==================================================================
4.f_update_miu & f_update_sigma，紧接produceGrad后调用，之所以分开写是因为在调用之前还可对新梯度做一些操作(例如梯度裁剪)，example:
==================================================================
new_grads_miu, new_grads_sigma = weight_noise.produceGrad(TAP_model, prior_u, prior_s2, 2 * 8835)

weight_noise.f_update_miu(TAP_model, new_grads_miu)
weight_noise.f_update_sigma(TAP_model, new_grads_sigma)
==================================================================
5.f_copy_weight，紧接f_update_miu & f_update_sigma后调用，将梯度更新到原梯度，example:
==================================================================
weight_noise.f_update_miu(TAP_model, new_grads_miu)
weight_noise.f_update_sigma(TAP_model, new_grads_sigma)
weight_noise.f_copy_weight(TAP_model)
==================================================================
'''

import torch
import numpy as np


class weight_noise_class:
    def __init__(self, model):
        self.tparams_p_u = []
        self.tparams_p_ls2 = []
        self.running_grads2_miu = []
        self.running_up2_miu = []
        self.running_grads2_sigma = []
        self.running_up2_sigma = []

        for i, param in enumerate(model.parameters()):
            if param.grad != None:
                self.running_grads2_miu.append(torch.zeros_like(param.grad.data))
                self.running_grads2_sigma.append(torch.zeros_like(param.grad.data))
            else:
                self.running_grads2_miu.append(None)
                self.running_grads2_sigma.append(None)
            self.running_up2_sigma.append(torch.zeros_like(param.data))
            self.running_up2_miu.append(torch.zeros_like(param.data))
        self.getTparams(model)

    def getTparams(self, model):
        with torch.no_grad():
            for i, param in enumerate(model.parameters()):
                init_sigma = 1.0e-12
                log_sigma_scale = 2048.0
                p_u = torch.ones_like(param.data) * param.data
                p_ls2 = torch.zeros_like(param.data) + np.log(init_sigma) * 2. / log_sigma_scale  # log_(sigma^2)
                self.tparams_p_u.append(p_u)
                self.tparams_p_ls2.append(p_ls2)

    # 首先该函数将tparams_p_u与tparams_p_ls2使用当前weight填好，并将tparams_p_ls2应用于当前weight
    # 其次产生Beta与prior_s2供后续new_grads_miu与new_grads_sigma使用
    def f_apply_noise_to_weight(self, model):
        with torch.no_grad():
            log_sigma_scale = 2048.0
            #  compute the prior mean and variation
            temp_sum = 0.0
            temp_param_count = 0.0

            for i, param in enumerate(model.parameters()):
                temp_sum = temp_sum + self.tparams_p_u[i].sum()
                temp_param_count = temp_param_count + np.prod(
                    np.array(list(self.tparams_p_u[i].shape)).astype("float32"))

                # param.data += torch.normal(0, 1, param.data.shape).cuda() * torch.exp(
                #     tparams_p_ls2[i] * log_sigma_scale) ** 0.5
                # add noise to weight
            prior_u = float(temp_sum) / temp_param_count
            temp_sum = 0.0
            for i, p_u in enumerate(self.tparams_p_u):
                p_s2 = torch.exp(self.tparams_p_ls2[i] * log_sigma_scale)  # sigma^2
                temp_sum = temp_sum + (p_s2).sum() + (((p_u - prior_u) ** 2).sum())

            prior_s2 = float(temp_sum) / temp_param_count
            return prior_u, prior_s2

    def produceGrad(self, model, prior_u, prior_s2, num_examples, model_cost_coefficient=0.1):
        with torch.no_grad():
            log_sigma_scale = 2048.0

            new_grads_miu = []
            new_grads_sigma = []
            for p_u, p_ls2, param in zip(self.tparams_p_u, self.tparams_p_ls2, model.parameters()):
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

    def f_update_miu(self, model, grads_miu, my_eps=-1):
        with torch.no_grad():
            for i, param in enumerate(model.parameters()):
                if grads_miu[i] != None:
                    if self.running_grads2_miu[i] == None:
                        temp_running_grads2_miu = 0
                    else:
                        temp_running_grads2_miu = self.running_grads2_miu[i]
                    self.running_grads2_miu[i] = temp_running_grads2_miu * 0.95 + 0.05 * grads_miu[i] ** 2
                    temp_updir_miu = - (self.running_up2_miu[i] + my_eps) ** 0.5 / (
                            self.running_grads2_miu[i] + my_eps) ** 0.5 * \
                                     grads_miu[i]
                    self.running_up2_miu[i] = self.running_up2_miu[i] * 0.95 + 0.05 * temp_updir_miu ** 2
                    assert temp_updir_miu.shape == self.tparams_miu[i].shape
                    self.tparams_miu[i] += temp_updir_miu
                    del temp_updir_miu
            # print(i)
            del grads_miu[i]
            torch.cuda.empty_cache()

    def f_update_sigma(self, model, grads_sigma, my_eps):
        with torch.no_grad():
            for i, param in enumerate(model.parameters()):
                if grads_sigma[i] != None:
                    if self.running_grads2_sigma[i] == None:
                        temp_running_grads2_sigma = 0
                    else:
                        temp_running_grads2_sigma = self.running_grads2_sigma[i]
                    self.running_grads2_sigma[i] = temp_running_grads2_sigma * 0.95 + 0.05 * grads_sigma[i] ** 2
                    temp_updir_sigma = - (self.running_up2_sigma[i] + my_eps) ** 0.5 / (
                            self.running_grads2_sigma[i] + my_eps) ** 0.5 * \
                                       grads_sigma[i]
                    self.running_up2_sigma[i] = self.running_up2_sigma[i] * 0.95 + 0.05 * temp_updir_sigma ** 2

                    assert temp_updir_sigma.shape == self.tparams_sigma[i].shape
                    self.tparams_sigma[i] += temp_updir_sigma
                    del temp_updir_sigma

            del grads_sigma[i]
            torch.cuda.empty_cache()

    def f_copy_weight(self, model):
        with torch.no_grad():
            # restore weight
            for i, param in enumerate(model.parameters()):
                param.data = self.tparams_p_u[i]
