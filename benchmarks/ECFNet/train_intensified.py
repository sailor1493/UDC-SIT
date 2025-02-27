import os
from pathlib import Path
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import functools
import torch.distributed as dist
import torch.multiprocessing as mp

from torch.utils.data import Subset
from torch.utils.data.distributed import DistributedSampler
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    BackwardPrefetch,
    MixedPrecision,
    ShardingStrategy,
    FullStateDictConfig,
    StateDictType,
)
from torch.distributed.fsdp.wrap import (
    size_based_auto_wrap_policy,
)

from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    apply_activation_checkpointing,
)


from torch.utils.data import DataLoader
from pathlib import Path

import tqdm

from networks.intensifiedECFNet import *
from datasets.dataset_pairs_npy import *
from torchmetrics.functional import (
    peak_signal_noise_ratio,
    structural_similarity_index_measure,
)
import numpy as np
from PIL import Image
import wandb
from util import setup_img_save_function


def print_root(rank, msg):
    if rank == 0:
        print(msg)


def _interpolate_0(x):
    return [x]


def _interpolate_1(x):
    return [x, F.interpolate(x, scale_factor=0.5)]


def _interpolate_2(x):
    return [x, F.interpolate(x, scale_factor=0.5), F.interpolate(x, scale_factor=0.25)]


def _interpolate_3(x):
    return [
        x,
        F.interpolate(x, scale_factor=0.5),
        F.interpolate(x, scale_factor=0.25),
        F.interpolate(x, scale_factor=0.125),
    ]


def _interpolate_4(x):
    return [
        x,
        F.interpolate(x, scale_factor=0.5),
        F.interpolate(x, scale_factor=0.25),
        F.interpolate(x, scale_factor=0.125),
        F.interpolate(x, scale_factor=0.0625),
    ]


def _interpolate_5(x):
    return [
        x,
        F.interpolate(x, scale_factor=0.5),
        F.interpolate(x, scale_factor=0.25),
        F.interpolate(x, scale_factor=0.125),
        F.interpolate(x, scale_factor=0.0625),
        F.interpolate(x, scale_factor=0.03125),
    ]


# def interpolate_down(x):
#     x_2 = F.interpolate(x, scale_factor=0.5)  # 1, 4, 128, 128
#     x_4 = F.interpolate(x_2, scale_factor=0.5)  # 1, 4, 64, 64
#     x_8 = F.interpolate(x_4, scale_factor=0.5)  # 1, 4, 32, 32
#     return [x_8, x_4, x_2, x]


def setup(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    # os.environ["MASTER_PORT"] = "12376"

    # initialize the process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)


def cleanup():
    dist.destroy_process_group()


def format_metrics_to_gb(item):
    """quick function to format numbers to gigabyte and round to 4 digit precision"""
    metric_num = item / (2**30)
    metric_num = round(metric_num, ndigits=4)
    return metric_num


def ecfnet_loss(lq, gt, ratio=0.5, eps=1e-12):
    charbonnier_loss = torch.mean(torch.sqrt((lq - gt) ** 2 + eps))
    fft_lq = abs(torch.fft.fftn(lq, dim=(-2, -1)))
    fft_gt = abs(torch.fft.fftn(gt, dim=(-2, -1)))
    fft_abs = F.l1_loss(fft_lq, fft_gt, reduction="mean")
    return charbonnier_loss + ratio * fft_abs


def ecfnet_loss_v2(lq, gt, ratio=0.5, eps=1e-8):
    # Fixed after call: 24/02/15
    charbonnier_loss = torch.sqrt(torch.sum((lq - gt) ** 2) + eps)
    fft_lq = torch.fft.fftn(lq, dim=(-2, -1))
    fft_gt = torch.fft.fftn(gt, dim=(-2, -1))
    fft_abs = F.l1_loss(fft_lq, fft_gt, reduction="sum")
    return charbonnier_loss + ratio * fft_abs


def ecfnet_loss_v3(lq, gt, ratio=0.5, eps=1e-8):
    # Fixed after call: 24/02/15 - changed to mean
    charbonnier_loss = torch.sqrt(torch.mean((lq - gt) ** 2) + eps)
    fft_lq = torch.fft.fftn(lq, dim=(-2, -1))
    fft_gt = torch.fft.fftn(gt, dim=(-2, -1))
    fft_abs = F.l1_loss(fft_lq, fft_gt, reduction="mean")
    return charbonnier_loss + ratio * fft_abs


def ensemble_loss(lqs, gts, ratio=0.5, eps=1e-12, fn=ecfnet_loss):
    loss = 0
    for lq, gt in zip(lqs, gts):
        loss += fn(lq, gt, ratio, eps)
    return loss


def train(
    args,
    model,
    rank,
    world_size,
    train_loader,
    optimizer,
    epoch,
    loss_lambda=0.5,
    sampler=None,
    level=3,
    loss_function="v1",
    wandb_base_epoch=0,
):
    model.train()
    ddp_loss = torch.zeros(2).to(rank)

    if sampler:
        sampler.set_epoch(epoch)
    if level == 1:
        interpolate_down = _interpolate_0
    elif level == 2:
        interpolate_down = _interpolate_1
    elif level == 3:
        interpolate_down = _interpolate_2
    elif level == 4:
        interpolate_down = _interpolate_3
    elif level == 5:
        interpolate_down = _interpolate_4
    elif level == 6:
        interpolate_down = _interpolate_5
    else:
        raise ValueError(f"Level {level} not found")
    save_dir_path = f"experiments/{args.experiment_name}"

    if loss_function == "v1":
        loss_fn = ecfnet_loss
    elif loss_function == "v2":
        loss_fn = ecfnet_loss_v2
    elif loss_function == "v3":
        loss_fn = ecfnet_loss_v3
    else:
        raise ValueError(f"Loss function {loss_function} not found")

    train_bar = tqdm.tqdm(train_loader) if rank == 0 else train_loader
    for data, target in train_bar:
        data = data.to(rank)
        target = target.to(rank)
        optimizer.zero_grad()
        data.requires_grad = True
        output = model(data)
        targets = interpolate_down(target)
        loss = ensemble_loss(output, targets, ratio=loss_lambda, fn=loss_fn)
        loss.backward()
        optimizer.step()
        ddp_loss[0] += loss.item()
        ddp_loss[1] += len(data)

    dist.all_reduce(ddp_loss, op=dist.ReduceOp.SUM)
    if rank == 0:
        avg_loss = ddp_loss[0] / ddp_loss[1]
        msg = "Train Epoch: {} \tLoss: {:.6f}".format(epoch, avg_loss)
        wandb.log({"train loss": avg_loss}, step=epoch + wandb_base_epoch)
        print(msg)
        with open(os.path.join(save_dir_path, f"train.log"), "a") as f:
            f.write(f"Epoch {epoch}: {msg}\n")


def save_image(output, save_path):
    print(f"Saving image to {save_path}")


def crop_and_forward(data, model, w_div=2, h_div=2, pad_size=128):
    """
    Crop the data for 4 by 4 patches and forward it through the model
    All size divisible by 64, and overlap by pad_size pixels
    1792 1280 -> 448 + 64
    """
    B, C, H, W = data.shape
    dev = data.device

    h_step = H // h_div
    w_step = W // w_div

    hs = []
    for i in range(h_div + 1):
        hs.append(i * h_step)
    ws = []
    for j in range(h_div + 1):
        ws.append(j * w_step)

    out_patch = torch.zeros((B, C, H, W), device=dev)
    for i in range(w_div):
        for j in range(h_div):
            left_target = ws[i] - pad_size
            right_target = ws[i + 1] + pad_size
            top_target = hs[j] - pad_size
            bottom_target = hs[j + 1] + pad_size
            left = left_target if left_target > 0 else 0
            left_pad = pad_size if left_target > 0 else 0
            right = right_target if right_target < W else W
            right_pad = pad_size if right_target < W else 0
            top = top_target if top_target > 0 else 0
            top_pad = pad_size if top_target > 0 else 0
            bottom = bottom_target if bottom_target < H else H
            bottom_pad = pad_size if bottom_target < H else 0
            in_patch = data[:, :, top:bottom, left:right]
            # print("i, j, top, bottom, left, right: ", i, j, top, bottom, left, right)
            # print("Paddings: ", top_pad, bottom_pad, left_pad, right_pad)
            patch = model(in_patch)[0][
                :,
                :,
                top_pad : h_step + top_pad,
                left_pad : w_step + left_pad,
            ]
            # print("patch: ", patch.shape)
            # print("data shape: ", data.shape)
            # print("in patch shape: ", in_patch.shape)
            out_patch[:, :, hs[j] : hs[j + 1], ws[i] : ws[i + 1]] = patch
    return out_patch


def test(
    model: nn.Module,
    rank: int,
    test_loader: DataLoader,
    save_image=False,
    save_dir=None,
    epoch=None,
    wandb_base_epoch=0,
    save_fn=None,
    save_img_count=-1,
):
    model.eval()
    ddp_loss = torch.zeros(3).to(rank)
    epoch_dir = os.path.join(save_dir, f"epoch_{epoch:03}")
    if rank == 0 and not os.path.exists(epoch_dir):
        os.makedirs(epoch_dir, exist_ok=True)

    remaining = save_img_count
    save = True
    iter_bar = tqdm.tqdm(test_loader) if rank == 0 else test_loader
    with torch.no_grad():
        for data, target, img_name in iter_bar:
            data, target = data.to(rank), target.to(rank)
            # output = model(data)
            output = crop_and_forward(data, model)
            # if ouptut is list, take only the last one
            if isinstance(output, list):
                output = output[0]

            img = output.clone()
            output = output * 255
            target = target * 255
            output = output.to(torch.uint8).to(torch.float32)
            target = target.to(torch.uint8).to(torch.float32)

            ddp_loss[0] += peak_signal_noise_ratio(output, target)
            ddp_loss[1] += structural_similarity_index_measure(output, target)
            ddp_loss[2] += len(data)
            # if save_image and save_dir is not None:
            if save:
                save_fn(
                    img,
                    os.path.join(epoch_dir, img_name[-1].replace(".npy", ".png")),
                )
                remaining -= 1
                if remaining == 0:
                    save = False

    dist.all_reduce(ddp_loss, op=dist.ReduceOp.SUM)

    if rank == 0 and epoch is not None:
        psnr_avg = ddp_loss[0] / ddp_loss[2]
        ssim_avg = ddp_loss[1] / ddp_loss[2]
        msg = "Test set: Average PSNR: {:.4f}, Average SSIM: {:.4f}".format(
            psnr_avg, ssim_avg
        )
        print(msg)
        with open(os.path.join(save_dir, f"test.log"), "a") as f:
            f.write(f"Epoch {epoch}: {msg}\n")
        wandb.log({"psnr": psnr_avg, "ssim": ssim_avg}, step=epoch + wandb_base_epoch)
    # return average psnr
    return ddp_loss[0] / ddp_loss[2]


def load_states(experiment_name: str):
    dir = Path("experiments") / experiment_name
    if not dir.exists():
        # make one
        dir.mkdir(parents=True, exist_ok=True)
        return None

    # check if model_latest, state_laste, and opt_latest exists
    model_latest = dir / "model_latest.pth"
    sched_latest = dir / "lr_scheduler_latest.pth"
    if not model_latest.exists() or not sched_latest.exists():
        return None

    # load the states
    model_state = torch.load(model_latest)
    sched_state = torch.load(sched_latest)
    return model_state, sched_state


def fsdp_main(rank, world_size, opt):
    setup(rank, world_size)
    print_root(rank, f"Running on rank {rank} of {world_size}")

    train_dataset = my_dataset(
        opt.train_input, opt.train_gt, mapping=opt.mapping, crop_size=opt.patch_size
    )
    print_root(rank, f"Training dataset path: {opt.train_input}")
    if opt.limit_train_batches > 1:
        train_dataset = Subset(
            train_dataset, np.arange(opt.limit_train_batches * world_size)
        )
        print_root(rank, f"Training dataset size: {len(train_dataset)}")
    else:
        print_root(rank, f"Training dataset size: {len(train_dataset)}")

    validation_dataset = my_dataset_eval(opt.val_input, opt.val_gt, mapping=opt.mapping)
    print_root(rank, f"Validation dataset path: {opt.val_input}")
    if opt.limit_val_batches > 1:
        validation_dataset = Subset(
            validation_dataset, np.arange(opt.limit_val_batches * world_size)
        )
        print_root(rank, f"Validation dataset size: {len(validation_dataset)}")
    else:
        print_root(rank, f"Validation dataset size: {len(validation_dataset)}")

    train_sampler = DistributedSampler(
        train_dataset, rank=rank, num_replicas=world_size, shuffle=True
    )
    validation_sampler = DistributedSampler(
        validation_dataset, rank=rank, num_replicas=world_size
    )
    print_root(rank, f"Created Sampler")

    train_kwargs = {"batch_size": opt.batch_size, "sampler": train_sampler}
    test_kwargs = {"batch_size": 1, "sampler": validation_sampler}
    cuda_kwargs = {"num_workers": 2, "pin_memory": True, "shuffle": False}
    train_kwargs.update(cuda_kwargs)
    test_kwargs.update(cuda_kwargs)

    train_loader = torch.utils.data.DataLoader(train_dataset, **train_kwargs)
    print_root(rank, f"Created train loader")
    validation_loader = torch.utils.data.DataLoader(validation_dataset, **test_kwargs)
    print_root(rank, f"Created validation loader")

    my_auto_wrap_policy = functools.partial(
        size_based_auto_wrap_policy, min_num_params=1
    )
    torch.cuda.set_device(rank)

    init_start_event = torch.cuda.Event(enable_timing=True)
    init_end_event = torch.cuda.Event(enable_timing=True)

    model = IntensifiedECFNet(
        in_nc=opt.channels,
        out_nc=opt.channels,
        level=opt.level_ablation,
        num_res=opt.num_res,
        base_channel=opt.base_channel,
    ).to(rank)
    # Activation Checkpoint Wrapper
    if opt.aggressive_checkpointing:
        blocks = [EBlock, DBlock, AFF1, AFF, SCM, FAM, SAM]
        check_fn = lambda m: isinstance(m, tuple(blocks))
        apply_activation_checkpointing(model, check_fn=check_fn)

    precision = MixedPrecision(
        param_dtype=torch.float16,
        reduce_dtype=torch.float16,
        buffer_dtype=torch.float16,
    )
    model = FSDP(
        model,
        device_id=rank,
        sharding_strategy=ShardingStrategy.NO_SHARD,
        backward_prefetch=BackwardPrefetch.BACKWARD_POST,
        forward_prefetch=False,
        # mixed_precision=precision,
        auto_wrap_policy=my_auto_wrap_policy,
    )
    print_root(rank, f"Created model")

    optimizer = optim.AdamW(model.parameters(), lr=opt.lr)
    print_root(rank, f"Created optimizer")
    lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, opt.num_epochs, eta_min=1e-6
    )
    print_root(rank, f"Created lr scheduler")

    dir = Path("experiments") / opt.experiment_name
    if not dir.exists():
        # make one
        dir.mkdir(parents=True, exist_ok=True)
    if rank == 0:
        wandb.init(
            project="ECFNet",
            name=opt.wandb_name,
            id=opt.experiment_name,
            resume="Allowed",
        )
    wandb_base_epoch = 0
    if opt.patch_size == 512:
        wandb_base_epoch = 1000
    elif opt.patch_size == 800:
        wandb_base_epoch = 1300

    save_path = Path(f"experiments/{opt.experiment_name}")
    save_dir_path = save_path.absolute()
    model_state, sched_state = None, None
    model_path = opt.pretrained_model
    if model_path is not None:
        model_path = Path(model_path)
        if model_path.exists():
            model_state = torch.load(model_path)
            print_root(rank, f"Loaded pretrained model from {model_path}")
    start_epoch = 1
    if opt.auto_resume:
        found_state = load_states(opt.experiment_name)
        if found_state is not None:
            model_state, sched_state = found_state
            # start_epoch = model_state["current_epoch"] + 1
            print_root(rank, f"Resuming from previous state, epoch {start_epoch}")
    else:
        found_state = None
        print_root(rank, "Not resuming from previous state")

    if model_state is not None:
        if "params" in model_state:
            model_state = model_state["params"]
        if "current_epoch" in model_state:
            del model_state["current_epoch"]
        model.load_state_dict(model_state)
    if sched_state is not None:
        lr_scheduler.load_state_dict(sched_state)
        print_root(rank, f"Loaded previous state")

    best_value = -1
    save_fn = setup_img_save_function(opt.channels)
    init_start_event.record()
    for epoch in range(start_epoch, opt.num_epochs + 1):
        print_root(rank, f"Training at epoch {epoch}")
        train(
            opt,
            model,
            rank,
            world_size,
            train_loader,
            optimizer,
            epoch,
            loss_lambda=opt.loss_lambda,
            sampler=train_sampler,
            loss_function=opt.loss_version,
            wandb_base_epoch=wandb_base_epoch,
            level=opt.level_ablation,
        )
        if epoch % opt.validation_interval == 0:
            print_root(rank, f"Testing at epoch {epoch}")
            curr_psnr = test(
                model,
                rank,
                validation_loader,
                epoch=epoch,
                save_dir=save_dir_path,
                wandb_base_epoch=wandb_base_epoch,
                save_fn=save_fn,
                save_img_count=opt.save_img_count,
            )

        if epoch % opt.save_interval == 0:
            print_root(rank, f"Saving at epoch {epoch}")
            dist.barrier()
            save_policy = FullStateDictConfig(offload_to_cpu=False, rank0_only=True)
            with FSDP.state_dict_type(
                model, StateDictType.FULL_STATE_DICT, save_policy
            ):
                cpu_state = model.module.state_dict()
            if rank == 0:
                for key in cpu_state:
                    cpu_state[key] = cpu_state[key].cpu()
                cpu_lr_scheduler = lr_scheduler.state_dict()
                torch.save(
                    cpu_state,
                    os.path.join(save_path, f"model_epoch_{epoch:03}.pth"),
                )
                # save latest states
                torch.save(
                    cpu_state,
                    os.path.join(save_path, f"model_latest.pth"),
                )
                torch.save(
                    cpu_lr_scheduler,
                    os.path.join(save_path, f"lr_scheduler_latest.pth"),
                )
                if curr_psnr > best_value:
                    best_value = curr_psnr
                    torch.save(
                        cpu_state,
                        os.path.join(save_path, f"model_best.pth"),
                    )
                    print_root(rank, f"Saved best model at epoch {epoch}")

            dist.barrier()
        lr_scheduler.step()

    init_end_event.record()
    dist.barrier()

    save_policy = FullStateDictConfig(offload_to_cpu=False, rank0_only=True)
    with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, save_policy):
        cpu_state = model.state_dict()
        if rank == 0:
            # save final model
            print(f"Saving final model")
            states = model.module.state_dict()
            for key in states:
                states[key] = states[key].cpu()
                # print(f"Saving {key} with size {states[key].size()}")
            torch.save(
                states,
                os.path.join(save_path, f"model_final.pth"),
            )
    dist.barrier()
    cleanup()


def main():
    parser = argparse.ArgumentParser()
    data_group = parser.add_argument_group("Data")
    data_group.add_argument("--train-input", type=str, required=True)
    data_group.add_argument("--train-gt", type=str, required=True)
    data_group.add_argument("--val-input", type=str, required=True)
    data_group.add_argument("--val-gt", type=str, required=True)
    data_group.add_argument(
        "--mapping", type=str, default="tonemap", choices=["tonemap", "norm"]
    )
    data_group.add_argument("--channels", type=int, default=3)

    train_group = parser.add_argument_group("Training Options")
    train_group.add_argument("--batch-size", type=int, default=1)
    train_group.add_argument("--num-epochs", type=int, default=150)
    train_group.add_argument("--lr", type=float, default=8 * 1e-6)
    train_group.add_argument("--patch-size", type=int, default=800)
    train_group.add_argument("--loss-lambda", type=float, default=0.5)
    train_group.add_argument("--num-res", type=int, default=6)
    train_group.add_argument("--loss-version", type=str, default="v1")
    train_group.add_argument(
        "--level-ablation", type=int, required=True, choices=[1, 2, 3, 4, 5, 6]
    )
    train_group.add_argument("--base-channel", type=int)

    trainer_group = parser.add_argument_group("Trainer Options")
    trainer_group.add_argument("--num-workers", type=int, default=8)
    trainer_group.add_argument("--validation-interval", type=int, default=5)
    trainer_group.add_argument("--limit-train-batches", type=int, default=1.0)
    trainer_group.add_argument("--limit-val-batches", type=int, default=1.0)
    trainer_group.add_argument("--aggressive-checkpointing", action="store_true")
    trainer_group.add_argument("--num-gpu", type=int, default=4)

    saving_group = parser.add_argument_group("Experiment Saving")
    saving_group.add_argument("--experiment-name", type=str, default="ECFNet")
    saving_group.add_argument("--save-interval", type=int, default=15)
    saving_group.add_argument("--save-img-count", type=int, default=-1)

    load_group = parser.add_argument_group("Experiment Loading")
    load_group.add_argument("--auto-resume", action="store_true")
    load_group.add_argument("--pretrained-model", type=str)
    load_group.add_argument("--wandb-name", type=str, required=True)

    args = parser.parse_args()

    msg = f"""Running with the following options:
    Train input: {args.train_input}
    Train gt: {args.train_gt}
    Val input: {args.val_input}
    Val gt: {args.val_gt}
    
    GPU Count: {args.num_gpu}
    Batch size: {args.batch_size}
    Num epochs: {args.num_epochs}
    
    num_res: {args.num_res}
    base_channel: {args.base_channel}
    """
    print(msg)

    torch.manual_seed(42)
    # torch.set_float32_matmul_precision("medium")
    torch.set_float32_matmul_precision("highest")

    WORLD_SIZE = args.num_gpu  # in our case, we have 4 GPUs
    # WORLD_SIZE = torch.cuda.device_count()

    mp.spawn(fsdp_main, args=(WORLD_SIZE, args), nprocs=WORLD_SIZE, join=True)


if __name__ == "__main__":
    main()
