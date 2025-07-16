def log_values(
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
):
    """Log training metrics for reinforcement learning (RL) to console and TensorBoard.

    Args:
        cost (torch.Tensor): Cost tensor (e.g., execution_time for NCO assignments).
        grad_norms (tuple): Tuple of (grad_norms, grad_norms_clipped) for model parameters.
        epoch (int): Current epoch number.
        batch_id (int): Current batch index.
        step (int): Global step (e.g., epoch * num_batches + batch_id).
        log_likelihood (torch.Tensor): Log-likelihood of selected actions.
        reinforce_loss (torch.Tensor): REINFORCE loss for the actor.
        bl_loss (torch.Tensor): Baseline (e.g., critic) loss.
        tb_logger (SummaryWriter): TensorBoard logger instance.
        opts (argparse.Namespace): Options including no_tensorboard and baseline.
    """
    avg_cost = cost.mean().item()
    grad_norms, grad_norms_clipped = grad_norms

    # Log values to screen
    print(
        "\nepoch: {}, train_batch_id: {}, avg_cost: {}".format(
            epoch, batch_id, avg_cost
        )
    )
    print("grad_norm: {}, clipped: {}".format(grad_norms[0], grad_norms_clipped[0]))

    # Log values to TensorBoard
    if not opts.no_tensorboard:
        tb_logger.log_value("avg_cost", avg_cost, step)
        tb_logger.log_value("actor_loss", reinforce_loss.item(), step)
        tb_logger.log_value("nll", -log_likelihood.mean().item(), step)
        tb_logger.log_value("grad_norm", grad_norms[0], step)
        tb_logger.log_value("grad_norm_clipped", grad_norms_clipped[0], step)
        if opts.baseline == "critic":
            tb_logger.log_value("critic_loss", bl_loss.item(), step)
            tb_logger.log_value("critic_grad_norm", grad_norms[1], step)
            tb_logger.log_value("critic_grad_norm_clipped", grad_norms_clipped[1], step)
