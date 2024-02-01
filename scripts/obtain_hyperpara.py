import torch
import numpy as np
from sample import sample
from evaluate import evaluate, median_pool 

#%% for the proposed method
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
    MSE = []
    MSE_NULL = []
    ABE_DIFF = []
    LAB = []

    # TODO: make it memory efficient!
    for i in range(args.num_batches_val):
        source_val, _, lab_val = next(data_val)
        source_val = source_val.to(device)

        y0 = torch.ones(source_val.shape[0], dtype=torch.long) *\
                torch.arange(start = 0, end = 1).reshape(-1, 1) # 0 for healthy
        y0 = y0.reshape(-1,1).squeeze().to(device)

        model_kwargs_reverse = {'threshold':-1, 'clf_free':True, 'null':args.null}
        model_kwargs = {'y': y0, 'threshold':-1, 'clf_free':True}
        minibatch_metrics = diffusion.calc_pred_xstart_loop(
                                                            model, source_val, args.w,
                                                            modality=args.modality,
                                                            d_reverse=args.d_reverse,
                                                            sample_steps=args.rev_steps,
                                                            model_kwargs=model_kwargs,
                                                            model_kwargs_reverse=model_kwargs_reverse)

        mse = (minibatch_metrics['xstart'] - source_val[:, args.modality, ...].unsqueeze(1))**2 # batch_size x sample_steps x n_modality x 128 x 128
        # mse = torch.mean(mse, dim=2, keepdim=True) # batch_size x sample_steps x 1 x 128 x 128

        mse_null = (minibatch_metrics['xstart_null'] - source_val[:, args.modality, ...].unsqueeze(1))**2
        # mse_null = torch.mean(mse_null, dim=2, keepdim=True)

        abe_diff = torch.abs((mse - mse_null)) # batch_size x sample_steps x n_modality x 128 x 128

        mse_flat = torch.mean(mse, dim=(3,4)) # batch_size x sample_steps x n_modality
        mse_null_flat = torch.mean(mse_null, dim=(3,4))
        abe_diff_flat = torch.mean(abe_diff, dim=(3,4))

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
#%% for ddib style sampling
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
    # abe_mask = F.avg_pool2d(abe_mask.float(), kernel_size=5, stride=1, padding=2) > 0.70
    abe_mask = abe_mask.float()
    pred_lab = (torch.sum(abe_mask, dim=(1, 2, 3)) > 0).float().cpu()
    return abe_mask, abe_sum, pred_lab

def obtain_optimal_threshold(data_val, diffusion, model, args, device, ddib=True):
    
    TARGET = []
    SOURCE = []
    MASK = []
    for i in range(args.num_batches_val):
        source_val, mask_val, _ = next(data_val)
        source_val = source_val.to(device)
        mask_val = mask_val.to(device)
        
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
            )
        else:
            t = torch.tensor(
                [args.sample_steps - 1] * source_val.shape[0], device=device
            )
            noise = diffusion.q_sample(source_val, t=t)
        
        y = torch.ones(source_val.shape[0], dtype=torch.long) * torch.arange(
            start=0, end=1
        ).reshape(
            -1, 1
        )  # 0 only for healthy
        y = y.reshape(-1, 1).squeeze().to(device)

        target, _ = sample(
            model,
            diffusion,
            y=y,
            noise=noise,
            w=args.w,
            sample_shape=source_val.shape,
            sample_steps=args.sample_steps,
            dynamic_clip=args.dynamic_clip,
            normalize_img=False,
        )
        TARGET.append(target)
        SOURCE.append(source_val)
        MASK.append(mask_val)
    
    TARGET = torch.cat(TARGET, dim=0)
    SOURCE = torch.cat(SOURCE, dim=0)
    MASK = torch.cat(MASK, dim=0)
    
    dice_max = 0
    thr_opt = 0
    threshold_range = np.arange(0.01, 0.5, 0.01)
    for thr in threshold_range:
        PRED_MASK, PRED_MAP, _ = get_mask_for_batch(
            SOURCE, TARGET, thr, args.modality, median_filter=True
        )
        eval_metrics = evaluate(MASK, PRED_MASK, SOURCE, PRED_MAP)
        
        if eval_metrics['dice'] > dice_max:
            dice_max = eval_metrics['dice']
            thr_opt = thr
            
    return thr_opt, dice_max
   