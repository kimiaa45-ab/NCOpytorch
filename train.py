import os
import time
from tqdm import tqdm
import torch
import math

from torch.utils.data import DataLoader
from utils.log_utils import log_values
from utils import move_to
from problem.problem_NCO import NCODataset, NCOProblem
from nets.attention_decor import AttentionModel
from nets.encoder.gnn_encoder import GNNEncoder


def get_inner_model(model):
    """Extract the inner model if using DataParallel."""
    return model.module if isinstance(model, torch.nn.DataParallel) else model


def set_decode_type(model, decode_type):
    """Set the decoding type (e.g., greedy or sampling) for the model."""
    if isinstance(model, torch.nn.DataParallel):
        model = model.module
    model.set_decode_type(decode_type)


def validate(model, dataset, problem, opts):
    """Validate the model on the dataset and compute average cost."""
    print(f"\nValidating on {dataset.size} samples from {dataset.filename}...")
    cost = rollout(model, dataset, opts)

    print(
        "Validation average cost: {:.3f} +- {:.3f}".format(cost.mean(), torch.std(cost))
    )

    return cost.mean()


def rollout(model, dataset, opts):
    """Evaluate the model in greedy mode to compute costs."""
    set_decode_type(model, "greedy")
    model.eval()

    def eval_model_bat(bat):
        with torch.no_grad():
            services_data, nodes_data = bat
            services_data = move_to(services_data, opts.device)
            nodes_data = move_to(nodes_data, opts.device)
            cost, _ = model(services_data, nodes_data)
        return cost.data.cpu()

    return torch.cat(
        [
            eval_model_bat(bat)
            for bat in tqdm(
                DataLoader(
                    dataset,
                    batch_size=opts.batch_size,
                    shuffle=False,
                    num_workers=opts.num_workers,
                ),
                disable=opts.no_progress_bar,
                ascii=True,
            )
        ],
        0,
    )


def clip_grad_norms(param_groups, max_norm=math.inf):
    """Clip gradient norms for all parameter groups and return norms before/after clipping."""
    grad_norms = [
        torch.nn.utils.clip_grad_norm_(
            group["params"], max_norm if max_norm > 0 else math.inf, norm_type=2
        )
        for group in param_groups
    ]
    grad_norms_clipped = (
        [min(g_norm.item(), max_norm) for g_norm in grad_norms]
        if max_norm > 0
        else grad_norms
    )
    return grad_norms, grad_norms_clipped


def train_epoch(
    model,
    optimizer,
    baseline,
    lr_scheduler,
    epoch,
    val_datasets,
    problem,
    tb_logger,
    opts,
):
    """Train the model for one epoch using RL."""
    print(
        "\nStart train epoch {}, lr={} for run {}".format(
            epoch, optimizer.param_groups[0]["lr"], opts.run_name
        )
    )
    step = epoch * (opts.epoch_size // opts.batch_size)
    start_time = time.time()

    if not opts.no_tensorboard:
        tb_logger.log_value("learnrate_pg0", optimizer.param_groups[0]["lr"], step)

    # Generate new training data for each epoch
    train_dataset = baseline.wrap_dataset(
        NCODataset(json_dir=opts.json_dir, num_samples=opts.epoch_size)
    )
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=opts.batch_size,
        shuffle=False,
        num_workers=opts.num_workers,
    )

    # Put model in train mode
    model.train()
    optimizer.zero_grad()
    set_decode_type(model, "sampling")

    for batch_id, batch in enumerate(
        tqdm(train_dataloader, disable=opts.no_progress_bar, ascii=True)
    ):
        train_batch(
            model, optimizer, baseline, epoch, batch_id, step, batch, tb_logger, opts
        )
        step += 1

    lr_scheduler.step(epoch)

    epoch_duration = time.time() - start_time
    print(
        "Finished epoch {}, took {} s".format(
            epoch, time.strftime("%H:%M:%S", time.gmtime(epoch_duration))
        )
    )

    if (
        opts.checkpoint_epochs != 0 and epoch % opts.checkpoint_epochs == 0
    ) or epoch == opts.n_epochs - 1:
        print("Saving model and state...")
        torch.save(
            {
                "model": get_inner_model(model).state_dict(),
                "optimizer": optimizer.state_dict(),
                "rng_state": torch.get_rng_state(),
                "cuda_rng_state": torch.cuda.get_rng_state_all(),
                "baseline": baseline.state_dict() if baseline is not None else None,
            },
            os.path.join(opts.save_dir, "epoch-{}.pt".format(epoch)),
        )

    for val_idx, val_dataset in enumerate(val_datasets):
        avg_reward = validate(model, val_dataset, problem, opts)
        if not opts.no_tensorboard:
            tb_logger.log_value(
                "val{}/avg_reward".format(val_idx + 1), avg_reward, step
            )


def train_batch(
    model, optimizer, baseline, epoch, batch_id, step, batch, tb_logger, opts
):
    """Train one batch using REINFORCE with baseline."""
    # Unwrap baseline
    bat, bl_val = (
        baseline.unwrap_batch(batch) if baseline is not None else (batch, None)
    )

    # Move tensors to device
    services_data, nodes_data = bat
    services_data = move_to(services_data, opts.device)
    nodes_data = move_to(nodes_data, opts.device)
    bl_val = move_to(bl_val, opts.device) if bl_val is not None else None

    # Evaluate model, get costs and log probabilities
    cost, log_likelihood = model(services_data, nodes_data)

    # Evaluate baseline, get baseline loss if any
    bl_val, bl_loss = (
        baseline.eval(services_data, nodes_data, cost)
        if bl_val is None and baseline is not None
        else (bl_val, 0)
    )

    # Calculate loss
    reinforce_loss = (
        ((cost - bl_val) * log_likelihood).mean() if bl_val is not None else cost.mean()
    )
    loss = reinforce_loss + bl_loss

    # Normalize loss for gradient accumulation
    loss = loss / opts.accumulation_steps

    # Perform backward pass
    loss.backward()

    # Clip gradient norms and get (clipped) gradient norms for logging
    grad_norms = clip_grad_norms(optimizer.param_groups, opts.max_grad_norm)

    # Perform optimization step after accumulating gradients
    if step % opts.accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()

    # Logging
    if step % int(opts.log_step) == 0:
        log_values(
            cost,
            grad_norms,
            epoch,
            batch_id,
            step,
            log_likelihood,
            reinforce_loss,
            bl_loss,
            tb_logger,
            opts,
        )
