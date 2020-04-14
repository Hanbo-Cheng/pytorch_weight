这份代码是复现的在CROHME上跑的一份代码，数据和基础模型已经准备好了，解压之后，在python3下安装好环境，直接用sbatch crohme.slurm就可以运行，weight_noise代码主要在weight_noise.py

需要参考原作者版本可移步https://github.com/JianshuZhang/TAP


## weight_noise算法类说明 ##

使用weight_noise不需要模型带有额外优化器，该算法自带adadelta优化器，即使用该算法，原模型优化器需要移除！

实现了weight_noise所有方法类，共有4个函数需要调用

+ init_weight_noise_class()，即构造函数，在初始化完完整模型后调用，example:

```
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
```

+ f_apply_noise_to_weight，在前向反馈前使用，为原weight增加噪声，example:

```
for tap_x, tap_y, tap_a in tap_train:
    TAP_model.train()
    tap_x, tap_x_mask, tap_y, tap_y_mask, tap_a, tap_a_mask = tap_prepare_data(
        params, tap_x, tap_y, tap_a, maxlen=maxlen)

    # 一,tparams_p_u,tparams_p_ls2只因被初始化一次，但prior_u,prior_s2,Beta需要被重新计算
    prior_u, prior_s2 = weight_noise_class.f_apply_noise_to_weight(TAP_model)
    # forward
    tap_ctx, cost_alphas = TAP_model(params, tap_x, tap_x_mask, tap_a, tap_a_mask, tap_y,
                                     tap_y_mask)
```


+ produceGrad，在损失回传完毕后调用，产生新的梯度，model_cost_coefficient为新添梯度的权重，num_examples为训练样本数*2，example:

```
TAP_model.zero_grad()
loss.backward()
new_grads_miu, new_grads_sigma = weight_noise.produceGrad(TAP_model, prior_u, prior_s2, 2 * 8835)
```

+ f_update_miu & f_update_sigma，紧接produceGrad后调用，之所以分开写是因为在调用之前还可对新梯度做一些操作(例如梯度裁剪)，example:

```
new_grads_miu, new_grads_sigma = weight_noise.produceGrad(TAP_model, prior_u, prior_s2, 2 * 8835)

weight_noise.f_update_miu(TAP_model, new_grads_miu)
weight_noise.f_update_sigma(TAP_model, new_grads_sigma)
```


+ f_copy_weight，紧接f_update_miu & f_update_sigma后调用，将梯度更新到原梯度，example:

```
weight_noise.f_update_miu(TAP_model, new_grads_miu)
weight_noise.f_update_sigma(TAP_model, new_grads_sigma)

weight_noise.f_copy_weight(TAP_model)
```
        

