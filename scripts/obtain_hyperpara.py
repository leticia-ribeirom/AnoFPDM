import torch
import numpy as np
from sample import sample
from evaluate import evaluate, median_pool
from noise import generate_simplex_noise


# %% This block is for the proposed method
def cal_cos_and_abe_range(mse_flat, mse_null_flat, abe_flat, lab):
    """
    mse_flat: N_val x sample_steps x n_modality
    mse_null_flat: N_val x sample_steps x n_modality
    abe_flat: N_val x sample_steps x n_modality
    lab: N_val

    get the cosine similarity threshold to differentiate healthy and tumour slices
    get the abe diff range for tumour slices to determine the quantile threshold for predicted mask
    """
    mse_0_flat = mse_flat[torch.where(lab == 0)[0]]
    mse_1_flat = mse_flat[torch.where(lab == 1)[0]]
    # print(f"mse_0_flat: {mse_0_flat.shape}")
    mse_0_null_flat = mse_null_flat[torch.where(lab == 0)[0]]
    mse_1_null_flat = mse_null_flat[torch.where(lab == 1)[0]]

    cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
    output_0 = cos(
        torch.mean(mse_0_flat, dim=2), torch.mean(mse_0_null_flat, dim=2)
    )  # N_val
    output_1 = cos(
        torch.mean(mse_1_flat, dim=2), torch.mean(mse_1_null_flat, dim=2)
    )  # N_val

    n_min = 1e6
    for q in np.linspace(0.01, 0.08, 100):
        n = torch.sum(output_1 > torch.quantile(output_0, q)) + torch.sum(
            output_0 < torch.quantile(output_0, q)
        )
        if n < n_min:
            n_min = n
            thr_01 = torch.quantile(output_0, q)

    abe_diff_1 = abe_flat[
        torch.where(lab == 1)[0]
    ]  # N_val_1 x sample_steps x n_modality
    abe_max = torch.max(abe_diff_1, dim=1)[0]  # N_val_1 x n_modality
    print(f"abe_max: {abe_max.shape}")
    return thr_01, abe_max.min(dim=0)[0], abe_max.max(dim=0)[0]


def obtain_hyperpara(data_val, diffusion, model, args, device):
    '''
    return the optimal threshold for cosine similarity for classification
            and the range of abe diff for quantile threshold for predicted mask
    '''
    MSE = []
    MSE_NULL = []
    ABE_DIFF = []
    LAB = []

    # TODO: make it memory efficient!
    for i in range(args.num_batches_val):
        source_val, _, lab_val = next(data_val)
        source_val = source_val.to(device)

        y0 = torch.ones(source_val.shape[0], dtype=torch.long) * torch.arange(
            start=0, end=1
        ).reshape(
            -1, 1
        )  # 0 for healthy
        y0 = y0.reshape(-1, 1).squeeze().to(device)

        model_kwargs_reverse = {"threshold": -1, "clf_free": True, "null": args.null}
        model_kwargs = {"y": y0, "threshold": -1, "clf_free": True}
        minibatch_metrics = diffusion.calc_pred_xstart_loop(
            model,
            source_val,
            args.w,
            modality=args.modality,
            d_reverse=args.d_reverse,
            sample_steps=args.rev_steps,
            model_kwargs=model_kwargs,
            model_kwargs_reverse=model_kwargs_reverse,
        )

        mse = (
            minibatch_metrics["xstart"] - source_val[:, args.modality, ...].unsqueeze(1)
        ) ** 2  # batch_size x sample_steps x n_modality x 128 x 128
        # mse = torch.mean(mse, dim=2, keepdim=True) # batch_size x sample_steps x 1 x 128 x 128

        mse_null = (
            minibatch_metrics["xstart_null"]
            - source_val[:, args.modality, ...].unsqueeze(1)
        ) ** 2
        # mse_null = torch.mean(mse_null, dim=2, keepdim=True)

        abe_diff = torch.abs(
            (mse - mse_null)
        )  # batch_size x sample_steps x n_modality x 128 x 128

        mse_flat = torch.mean(mse, dim=(3, 4))  # batch_size x sample_steps x n_modality
        mse_null_flat = torch.mean(mse_null, dim=(3, 4))
        abe_diff_flat = torch.mean(abe_diff, dim=(3, 4))

        MSE.append(mse_flat)
        MSE_NULL.append(mse_null_flat)
        ABE_DIFF.append(abe_diff_flat)
        LAB.append(lab_val)

    MSE = torch.cat(MSE, dim=0)
    MSE_NULL = torch.cat(MSE_NULL, dim=0)
    ABE_DIFF = torch.cat(ABE_DIFF, dim=0)
    LAB = torch.cat(LAB, dim=0)
    thr_01, abe_min, abe_max = cal_cos_and_abe_range(MSE, MSE_NULL, ABE_DIFF, LAB)
    return thr_01, abe_min, abe_max


def get_mask_batch_FPDM(
    minibatch_metrics,
    source,
    modality,
    thr_01,
    abe_min,
    abe_max,
    shape,
    device,
    thr = None,
    t_e = None,
    median_filter=True,
):
    """
    thr_01: threshold for cosine similarity
    thr: threshold for predicted mask (if not provided, it will be calculated)
    t_e: steps (noise scale) for predicted mask (if not provided, it will be calculated)  
    mse: batch_size x sample_steps x n_modality x 128 x 128 or 256 x 256
    mse_null: batch_size x sample_steps x n_modality x 128 x 128 or 256 x 256
    """
    mse = (
        minibatch_metrics["xstart"] - source[:, modality, ...].unsqueeze(1)
    ) ** 2  # batch_size x sample_steps x n_modality x 128 x 128
    mse_null = (
        minibatch_metrics["xstart_null"] - source[:, modality, ...].unsqueeze(1)
    ) ** 2

    abe_diff = torch.mean(
        torch.abs((mse - mse_null)),
        dim=(3, 4),  # batch_size x sample_steps x n_modality
    )

    mse_flat = torch.mean(mse, dim=(2, 3, 4))  # batch_size x sample_steps
    mse_null_flat = torch.mean(mse_null, dim=(2, 3, 4))

    batch_mask = torch.zeros(mse_flat.shape[0], 1, shape, shape).to(device)
    batch_map = torch.zeros(mse_flat.shape[0], 1, shape, shape).to(device)
    cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)

    quant_range = torch.flip(torch.linspace(0.90, 0.98, 101), dims=(0,)).to(
        device
    )

    pred_lab = []
    for sample_num in range(abe_diff.shape[0]):
        # get the cosine similarity between mse and mse_null
        sim = cos(
            mse_flat[sample_num : sample_num + 1, ...],
            mse_null_flat[sample_num : sample_num + 1, ...],
        )

        # get the quantile threshold for predicted mask 
        abe_diff_i = abe_diff[sample_num, ...]  # sample_steps x n_modality
        
        abe_max_i = abe_diff_i.max(dim=0)[0]  # n_modality
        abe_max_i = torch.clamp((abe_max_i / abe_max), 0, 1)
        abe_max_i = torch.round(abe_max_i, decimals=2) * 100
        index = abe_max_i.to(torch.int64)  # n_modality
        quant = quant_range[index]

        # get the steps for predicted mask
        t_s_i = 0
        t_e_i = torch.argmax(abe_diff_i, dim=0)  if t_e is None else t_e # n_modality

        mapp = torch.zeros(1, 1, shape, shape).to(device)
        
        thr_i = 0 
        for mod in range(mse.shape[2]):
            mask_mod = torch.mean(
                mse[sample_num, t_s_i : t_e_i[mod], [mod], ...], axis=[0, 1], keepdim=True
            )  # 1 x 1 x 128 x 128
            
            thr_i += torch.quantile(mask_mod.reshape(-1), quant[mod])
            mapp += mask_mod

        # collect the predicted mask and map
        mapp /= mse.shape[2]
        mapp = (
            median_pool(mapp, kernel_size=5, stride=1, padding=2)
            if median_filter
            else mapp
        )
        batch_map[sample_num] = mapp

        if sim <= thr_01:
            thr_i = (thr_i / mse.shape[2]) if thr is None else thr
            mask = mapp >= thr_i
            batch_mask[sample_num] = mask.float()
            pred_lab.append(1)
        else:
            pred_lab.append(0)

    return batch_mask, torch.tensor(pred_lab), batch_map





# %% for non-dynamical threshold to obtain pred_mask
def get_mask_for_batch(source, target, threshold, mod, median_filter=True):
    abe_sum = (
        (source[:, mod, ...] - target[:, mod, ...]).abs().mean(dim=1, keepdims=True)
    )  # nx1x128x128

    abe_sum = (
        median_pool(abe_sum, kernel_size=5, stride=1, padding=2)
        if median_filter
        else abe_sum
    )

    abe_mask = abe_sum >= threshold
    abe_mask = abe_mask.float()
    pred_lab = (torch.sum(abe_mask, dim=(1, 2, 3)) > 0).float().cpu()
    return abe_mask, abe_sum, pred_lab


def obtain_optimal_threshold(
    data_val,
    diffusion,
    model,
    args,
    device,
    ddib=True,
    guided=True,
    cond_fn=None,
    noise_fn=None,
    use_ddpm=False,
):

    TARGET = []
    SOURCE = []
    MASK = []
    for i in range(args.num_batches_val):
        source_val, mask_val, _ = next(data_val)
        source_val = source_val.to(device)
        mask_val = mask_val.to(device)

        # Forward process
        # if ddib, image will be encoded by DDIM forward process
        # ddib is from the paper "DUAL DIFFUSION IMPLICIT BRIDGES FOR IMAGE-TO-IMAGE TRANSLATION"
        if ddib:
            noise, _ = sample(
                model,
                diffusion,
                noise=source_val,
                sample_steps=args.sample_steps,
                reverse=True,
                null=True,
                dynamic_clip=args.dynamic_clip,
                normalize_img=False,
                ddpm=False,
            )
        else:
            t = torch.tensor(
                [args.sample_steps - 1] * source_val.shape[0], device=device
            )
            ep = noise_fn(source_val, t) if noise_fn else None
    
            noise = diffusion.q_sample(source_val, t=t, noise=ep)
            
        # if guided, y will be used to guide the sampling process
        if guided:
            y = torch.ones(source_val.shape[0], dtype=torch.long) * torch.arange(
                start=0, end=1
            ).reshape(
                -1, 1
            )  # 0 only for healthy
            y = y.reshape(-1, 1).squeeze().to(device)
        else:
            y = None

        # sampling process
        target, _ = sample(
            model,
            diffusion,
            y=y,
            noise=noise,
            w=args.w,
            noise_fn=noise_fn,
            cond_fn=cond_fn,
            sample_shape=source_val.shape,
            sample_steps=args.sample_steps,
            dynamic_clip=args.dynamic_clip,
            normalize_img=False,
            ddpm=use_ddpm,
        )
        TARGET.append(target)
        SOURCE.append(source_val)
        MASK.append(mask_val)

    TARGET = torch.cat(TARGET, dim=0)
    SOURCE = torch.cat(SOURCE, dim=0)
    MASK = torch.cat(MASK, dim=0)

    dice_max = 0
    thr_opt = 0
    # range of threshold, select the best one
    threshold_range = np.arange(0.01, 0.7, 0.01)
    for thr in threshold_range:
        PRED_MASK, PRED_MAP, _ = get_mask_for_batch(
            SOURCE, TARGET, thr, args.modality, median_filter=True
        )
        eval_metrics = evaluate(MASK, PRED_MASK, SOURCE, PRED_MAP)

        if eval_metrics["dice"] > dice_max:
            dice_max = eval_metrics["dice"]
            thr_opt = thr

    return thr_opt, dice_max
