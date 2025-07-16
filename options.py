import os
import time
import argparse
import torch


def get_options(args=None):
    parser = argparse.ArgumentParser(
        description="Arguments and hyperparameters for training and evaluating learning-driven solvers for NCO"
    )

    # Data
    parser.add_argument(
        "--problem", default="nco", help="The problem to solve, default 'nco'"
    )
    parser.add_argument(
        "--min_size", type=int, default=10, help="Minimum number of nodes in NCO"
    )
    parser.add_argument(
        "--max_size", type=int, default=10, help="Maximum number of nodes in NCO"
    )
    parser.add_argument(
        "--neighbors", type=float, default=-1, help="No k-nearest neighbors for NCO"
    )
    parser.add_argument(
        "--knn_strat", type=str, default="none", help="No knn strategy for NCO"
    )
    parser.add_argument(
        "--n_epochs", type=int, default=10, help="The number of epochs to train"
    )
    parser.add_argument(
        "--epoch_size", type=int, default=128, help="Number of instances per epoch"
    )
    parser.add_argument(
        "--batch_size", type=int, default=16, help="Number of instances per batch"
    )
    parser.add_argument(
        "--accumulation_steps",
        type=int,
        default=1,
        help="Gradient accumulation step (effective batch_size = batch_size * accumulation_steps)",
    )
    parser.add_argument(
        "--val_datasets",
        type=str,
        nargs="+",
        default=["data/processed/val.json"],
        help="JSON dataset files for validation",
    )
    parser.add_argument(
        "--val_size", type=int, default=32, help="Number of instances for validation"
    )
    parser.add_argument(
        "--json_dir",
        type=str,
        default="data/processed",
        help="Directory containing JSON data for NCODataset",
    )
    parser.add_argument(
        "--data_distribution",
        type=str,
        default="random",
        help='Data distribution for NCO: "random"',
    )

    # Model/GNN Encoder
    parser.add_argument("--model", default="attention", help="Model: 'attention' only")
    parser.add_argument("--encoder", default="gnn", help="Graph encoder: 'gnn' only")
    parser.add_argument(
        "--embedding_dim", type=int, default=128, help="Dimension of input embedding"
    )
    parser.add_argument(
        "--hidden_dim",
        type=int,
        default=128,
        help="Dimension of hidden layers in Enc/Dec",
    )
    parser.add_argument(
        "--n_encode_layers", type=int, default=3, help="Number of layers in the encoder"
    )
    parser.add_argument(
        "--aggregation",
        default="max",
        help="Neighborhood aggregation function: 'sum'/'mean'/'max'",
    )
    parser.add_argument(
        "--aggregation_graph",
        default="mean",
        help="Graph embedding aggregation: 'sum'/'mean'/'max'",
    )
    parser.add_argument(
        "--normalization",
        default="layer",
        help="Normalization type: 'batch'/'layer'/None",
    )
    parser.add_argument(
        "--learn_norm", action="store_true", help="Enable learnable normalization"
    )
    parser.add_argument(
        "--track_norm", action="store_true", help="Enable tracking batch statistics"
    )
    parser.add_argument("--gated", action="store_true", help="Enable edge gating")
    parser.add_argument(
        "--n_heads", type=int, default=4, help="Number of attention heads"
    )
    parser.add_argument(
        "--tanh_clipping",
        type=float,
        default=10.0,
        help="Clip parameters to within +- this value using tanh. Set to 0 to disable.",
    )

    # Training
    parser.add_argument(
        "--lr_model",
        type=float,
        default=1e-4,
        help="Learning rate for the actor network",
    )
    parser.add_argument(
        "--lr_critic",
        type=float,
        default=1e-4,
        help="Learning rate for the critic network",
    )
    parser.add_argument(
        "--lr_decay", type=float, default=0.99, help="Learning rate decay per epoch"
    )
    parser.add_argument(
        "--max_grad_norm",
        type=float,
        default=1.0,
        help="Maximum L2 norm for gradient clipping",
    )
    parser.add_argument(
        "--exp_beta", type=float, default=0.8, help="Exponential baseline decay"
    )
    parser.add_argument(
        "--baseline",
        default="genetic",
        help="Baseline: 'genetic', 'exponential', 'none'",
    )
    parser.add_argument(
        "--population_size",
        type=int,
        default=50,
        help="Population size for genetic baseline",
    )
    parser.add_argument(
        "--generations",
        type=int,
        default=100,
        help="Number of generations for genetic baseline",
    )
    parser.add_argument(
        "--mutation_rate",
        type=float,
        default=0.1,
        help="Mutation rate for genetic baseline",
    )
    parser.add_argument(
        "--crossover_rate",
        type=float,
        default=0.8,
        help="Crossover rate for genetic baseline",
    )
    parser.add_argument(
        "--checkpoint_encoder",
        action="store_true",
        help="Checkpoint encoder to save memory",
    )
    parser.add_argument(
        "--shrink_size", type=int, default=None, help="Shrink batch size to save memory"
    )
    parser.add_argument(
        "--eval_only", action="store_true", help="Set to only evaluate model"
    )
    parser.add_argument("--seed", type=int, default=1234, help="Random seed to use")

    # Evaluation
    parser.add_argument(
        "--decode_strategy",
        type=str,
        default="greedy",
        help="Decoding strategy: 'greedy', 'sample', 'bs'",
    )
    parser.add_argument(
        "--width", type=int, default=1, help="Width for beam search decoding"
    )
    parser.add_argument(
        "--eval_batch_size", type=int, default=16, help="Batch size for evaluation"
    )

    # Misc
    parser.add_argument(
        "--num_workers", type=int, default=0, help="Number of workers for DataLoaders"
    )
    parser.add_argument(
        "--log_step", type=int, default=100, help="Log info every log_step steps"
    )
    parser.add_argument(
        "--log_dir", default="logs", help="Directory to write TensorBoard information"
    )
    parser.add_argument(
        "--run_name", default="nco_run", help="Name to identify the run"
    )
    parser.add_argument(
        "--output_dir", default="models", help="Directory to write output models"
    )
    parser.add_argument("--epoch_start", type=int, default=0, help="Start at epoch #")
    parser.add_argument(
        "--checkpoint_epochs",
        type=int,
        default=1,
        help="Save checkpoint every n epochs",
    )
    parser.add_argument(
        "--load_path", help="Path to load model parameters and optimizer state"
    )
    parser.add_argument("--resume", help="Resume from previous checkpoint file")
    parser.add_argument(
        "--no_tensorboard",
        action="store_true",
        help="Disable logging TensorBoard files",
    )
    parser.add_argument(
        "--no_progress_bar", action="store_true", help="Disable progress bar"
    )
    parser.add_argument("--no_cuda", action="store_true", help="Disable CUDA")

    opts = parser.parse_args(args)
    opts.use_cuda = torch.cuda.is_available() and not opts.no_cuda
    opts.run_name = f"{opts.run_name}_{time.strftime('%Y%m%dT%H%M%S')}"
    opts.save_dir = os.path.join(opts.output_dir, "nco", opts.run_name)

    return opts
