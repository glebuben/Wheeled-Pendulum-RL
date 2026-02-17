import time
import numpy as np
import torch
from torch import optim
from torch.nn.functional import mse_loss
import os
import uuid
from datetime import datetime

# local imports (same directory)
from wheel_pole_system import WheelPoleSystem
from policy_network import PolicyNetwork


def upright_reward(prev_state, action, new_state):
    # theta = new_state[2]
    # return float(np.cos(theta))
    return +1.0


def compute_returns(rewards, gamma):
    returns = []
    G = 0.0
    for r in reversed(rewards):
        G = r + gamma * G
        returns.insert(0, G)
    return returns


def train_reinforce(
    policy_net,
    system,
    batch_size=32,
    max_seq_len=300,
    gamma=0.99,
    lr=3e-4,
    num_updates=100,
    theta_threshold=0.2,
    falling_penalty=-0.0,
    device=None,
):

    average_return_history = []

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    policy_net.to(device)
    optimizer = optim.Adam(policy_net.parameters(), lr=lr)

    for update_idx in range(1, num_updates + 1):
        batch_log_probs = []
        batch_returns = []
        episode_returns = []

        start_time = time.time()
        # Collect trajectories
        for ep in range(batch_size):
            system.reset()
            # small random perturbation to initial pole angle
            init_theta = 0.05 * (np.random.rand() - 0.5)
            system.set_initial_state(phi=0.0, theta=init_theta)

            state = system.get_state()
            log_probs = []
            rewards = []
            values = []

            for t in range(max_seq_len):
                obs = torch.from_numpy(state.astype(np.float32)).unsqueeze(0).to(device)

                mean, log_std, value = policy_net(obs)
                std = torch.exp(log_std)
                dist = torch.distributions.Normal(mean, std)
                action = dist.sample()

                dist.log_prob(action)
                log_prob = dist.log_prob(action).squeeze()

                action_value = float(action.detach().cpu().numpy().squeeze())
                new_state, reward = system.step(
                    action_value, reward_func=upright_reward
                )

                log_probs.append(log_prob)
                rewards.append(float(reward))
                values.append(value)
                if np.abs(new_state[2]) > theta_threshold:
                    rewards[-1] += falling_penalty
                    break

                state = new_state

            returns = compute_returns(rewards, gamma)
            ep_return = sum(rewards)
            episode_returns.append(ep_return)

            # convert to tensors
            returns_t = torch.tensor(returns, dtype=torch.float32, device=device)
            log_probs_t = torch.stack(log_probs).squeeze().to(device)
            values_t = torch.stack(values).squeeze().to(device)

            batch_log_probs.append(log_probs_t)
            batch_returns.append(returns_t)

        loss = torch.tensor(0.0, device=device)

        for log_prob, G in zip(batch_log_probs, batch_returns):
            loss += torch.sum(-log_prob * G) / batch_size

        # for name, param in policy_net.named_parameters():
        #     if param.grad is not None:
        #         grad_norm = param.grad.norm(2)  # Compute the L2 norm
        #         print(f"Gradient norm for layer {name}: {grad_norm.item()}")

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        avg_return = float(np.mean(episode_returns))
        avg_episode_length = float(np.mean([len(lp) for lp in batch_log_probs]))
        elapsed = time.time() - start_time
        print(
            f"Update {update_idx:4d} | Loss {loss.item():.4f} | AvgReturn {avg_return:.3f} | AvgEpisodeLength {avg_episode_length:.2f} | Time {elapsed:.2f}s"
        )

        average_return_history.append(avg_return)

    return average_return_history


if __name__ == "__main__":
    # hyperparameters requested by user
    trajectories_per_update = 256
    max_seq_len = 300
    gamma = 0.99
    lr = 3e-3
    num_updates = 20

    env = WheelPoleSystem(rod_length=1.0, wheel_radius=0.2)
    policy = PolicyNetwork(action_bound=1.0)

    average_return_history = train_reinforce(
        policy,
        env,
        batch_size=trajectories_per_update,
        max_seq_len=max_seq_len,
        gamma=gamma,
        lr=lr,
        num_updates=num_updates,
    )

# Save checkpoint with unique name
os.makedirs("checkpoints", exist_ok=True)
run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
uniq = uuid.uuid4().hex[:8]
ckpt_path = os.path.join("checkpoints", f"checkpoint_{run_id}_{uniq}.pt")
torch.save(
    {
        "policy_state_dict": policy.state_dict(),
        "average_return_history": average_return_history,
        "hyperparams": {
            "trajectories_per_update": trajectories_per_update,
            "max_seq_len": max_seq_len,
            "gamma": gamma,
            "lr": lr,
            "num_updates": num_updates,
            "action_bound": policy.action_bound,
        },
    },
    ckpt_path,
)

print(f"Saved checkpoint to {ckpt_path}")
