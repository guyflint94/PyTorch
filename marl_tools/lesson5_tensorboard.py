# Lesson 5 — TensorBoard for MAPPO
"""
Now we'll learn how to monitor whether your reinforcement learning agents are actually learning.
In MARL, the code can run for thousands of episodes without an obvious error, yet the agents might not improve, might stop exploring,
or might learn poor coordination. TensorBoard helps you see these patterns over time.

TensorBoard is a visualization tool. Your PyTorch code writes metrics to log files, and TensorBoard displays them as graphs.

episode/reward:	Whether the agents are achieving better outcomes
loss/policy: How the actor's policy objective changes
loss/value:	How the critic's value predictions are improving
policy/entropy:	How random or concentrated the action distribution is
policy/approx_kl: How much the updated policy differs from the old policy

After running the script, execute this command in the same Python environment (terminal):
tensorboard --logdir=runs
The terminal will display a local address, commonly http://localhost:6006. Open that address in your browser.
"""


import torch
from torch.utils.tensorboard import SummaryWriter

# Create a directory for TensorBoard logs
writer = SummaryWriter("runs/mappo_experiment_1")

for step in range(100):
    # Simulated metrics (not actual RL training yet)
    reward = step * 0.5 + torch.randn(()).item()
    policy_loss = 1.0 / (step + 1)
    entropy = 1.0 / (1 + step * 0.03)

    # Log each metric against the training step
    writer.add_scalar("episode/reward", reward, step)
    writer.add_scalar("loss/policy", policy_loss, step)
    writer.add_scalar("policy/entropy", entropy, step)

writer.close()

print("Finished writing TensorBoard logs.")

"""
    A few important details:
1. SummaryWriter(...) creates a writer that saves your metrics.
2. add_scalar(tag, value, step) logs one numerical value.
3. The tag, such as "episode/reward", becomes the name of a graph.
4. step is the x-axis coordinate — here, the training update number.
5. writer.close() flushes and closes the writer when you're done.

Important: these metrics are simulated. This experiment teaches you how to log data; it does not train a MAPPO agent.
"""

"""
What does entropy mean in reinforcement learning?
Entropy measures how uncertain or spread out a probability distribution is.
In RL, the policy pi(a|s) gives the probability of choosing action \(a\) in state \(s\). Entropy tells us how spread out those action probabilities are.
"""

"""
# TensorBoard & Entropy — Quick Notes

## 1. TensorBoard Metrics

- `writer.add_scalar(tag, value, step)` logs a numeric value for visualization.
- The tag names the metric, the value is the Y-axis, and the step is the X-axis.
- TensorBoard visualizes values; it does not calculate or validate the metrics.
- Simulated metrics are for demonstration, not actual RL training.

```python
writer.add_scalar("episode/reward", episode_reward, episode)
writer.add_scalar("loss/policy", policy_loss.item(), update)
writer.add_scalar("policy/entropy", entropy.item(), update)

writer.close()
```

## 2. Entropy in RL

- Entropy measures how spread out the policy's action probabilities are.
- High entropy: actions have more evenly distributed probabilities; more uncertainty.
- Low entropy: probability is concentrated on fewer actions; more predictable behavior.
- Lower entropy does not necessarily mean better performance.
- Entropy is commonly used as an exploration bonus in PPO/MAPPO.
- RL convention: `log()` means the natural logarithm, equivalent to `ln()`.
- Natural-log entropy is measured in nats.

```python
from torch.distributions import Categorical

dist = Categorical(probs=probs)
entropy = dist.entropy()
```

## 3. TensorBoard X-Axis and Y-Axis

```python
writer.add_scalar("episode/reward", reward, episode)
```

- `"episode/reward"`: metric name.
- `reward`: Y-axis value.
- `episode`: X-axis value, representing progress.

## 4. Choosing the Step Counter

The third argument (`global_step`) can represent different kinds of progress:

```python
writer.add_scalar("episode/reward", episode_reward, episode)
writer.add_scalar("loss/policy", policy_loss, update)
writer.add_scalar("reward/by_env_steps", reward, total_env_steps)
```

- `episode`: reward per episode.
- `update`: loss per training update.
- `total_env_steps`: progress by environment interactions.

## 5. Useful Optional Arguments

```python
writer.add_scalar(
    "episode/reward",
    reward,
    episode,
    walltime=None,
    new_style=False,
    double_precision=False,
)
```

- `walltime`: timestamp in seconds.
- `new_style`: selects an alternative internal event format.
- `double_precision`: writes the scalar with double precision.

## 6. Start TensorBoard

Run in the terminal:

```bash
tensorboard --logdir=runs
```

Open `http://localhost:6006` in your browser.

**Remember:** Real MAPPO metrics must come from the environment and actual policy-training calculations, not invented formulas.
"""