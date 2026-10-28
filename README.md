# PyTorch

Exercises and notes from learning PyTorch, with a few small building blocks
for multi-agent reinforcement learning.

## Layout

    basics/       tensors, autograd, backpropagation, gradient descent,
                  training pipeline, Dataset/DataLoader, activation functions,
                  a feed-forward network on MNIST
    marl_tools/   five short lessons used for multi-agent RL (MAPPO):
                  softmax, Categorical, action masking, gather, TensorBoard
    scratch/      quick experiments

## Running

    pip install torch torchvision tensorboard numpy matplotlib

Run each script from inside its own folder. Notes:

- `basics/13_feed_forward_nn.py` downloads MNIST into `./data`.
- `basics/09_dataset_dataloader.py` expects `./data/wine/wine.csv`, which is not
  included in this repository.
- `marl_tools/lesson5_tensorboard.py` writes logs to `runs/`; view them with
  `tensorboard --logdir=runs`.
