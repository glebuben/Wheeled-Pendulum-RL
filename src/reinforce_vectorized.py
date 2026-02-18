import time
import numpy as np
import torch
from torch import optim
import os
import uuid
from datetime import datetime

# local imports (same directory)
from src.systems.numba_wheel_pole_system import NumbaWheelPoleSystem
from src.policy_network import PolicyNetwork


def upright_reward(prev_state, action, new_state):
    """Reward function: cos(theta) where theta=0 is upright."""
    theta = new_state[2]
    return float(np.cos(8 * theta))


def compute_returns(rewards, gamma):
    """Compute discounted returns from a list of rewards."""
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
    save_freq=100,
    checkpoint_dir="checkpoints",
):
    """
    Train policy using REINFORCE with vectorized environments.
    
    The system should be a NumbaWheelPoleSystem with n_envs = batch_size.
    All episodes run in parallel for maximum efficiency.
    
    Parameters:
    -----------
    save_freq : int
        Save checkpoint every N updates (default: 100)
    checkpoint_dir : str
        Directory to save checkpoints
    """
    average_return_history = []
    best_avg_return = -float('inf')
    best_policy_state = None

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    policy_net.to(device)
    optimizer = optim.Adam(policy_net.parameters(), lr=lr)

    # Create checkpoint directory
    os.makedirs(checkpoint_dir, exist_ok=True)
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    uniq = uuid.uuid4().hex[:8]

    try:
        for update_idx in range(1, num_updates + 1):
            batch_log_probs = []
            batch_returns = []
            episode_returns = []

            start_time = time.time()
            
            # Reset all environments at once with random initial perturbations
            init_thetas = 0.2 * (np.random.rand(batch_size).astype(np.float32) - 0.5)
            system.reset(phi=0.0, theta=init_thetas)
            
            # Collect trajectories (all episodes in parallel)
            all_log_probs = [[] for _ in range(batch_size)]
            all_rewards = [[] for _ in range(batch_size)]
            done = np.zeros(batch_size, dtype=bool)
            
            for t in range(max_seq_len):
                # Get current states for all environments
                states = system.get_states()  # (batch_size, 4)
                
                # Forward pass through policy (vectorized)
                obs = torch.from_numpy(states.astype(np.float32)).to(device)
                mean, log_std, value = policy_net(obs)
                std = torch.exp(log_std)
                dist = torch.distributions.Normal(mean, std)
                action = dist.sample()
                log_prob = dist.log_prob(action).squeeze()
                
                # Convert actions to numpy for stepping
                actions = action.detach().cpu().numpy().squeeze()
                if actions.ndim == 0:  # Handle single action case
                    actions = np.array([actions])
                
                # Step all environments
                new_states = system.step(actions)
                
                # Compute rewards for each environment
                rewards = np.array([upright_reward(states[i], actions[i], new_states[i]) 
                                   for i in range(batch_size)], dtype=np.float32)
                
                # Check for falling (per environment)
                newly_fallen = (np.abs(new_states[:, 2]) > theta_threshold) & ~done
                
                # Store transitions for non-done environments
                for i in range(batch_size):
                    if not done[i]:
                        all_log_probs[i].append(log_prob[i])
                        all_rewards[i].append(float(rewards[i]))
                        
                        if newly_fallen[i]:
                            all_rewards[i][-1] += falling_penalty
                            done[i] = True
                
                # Break if all episodes are done
                if done.all():
                    break
            
            # Compute returns for each episode
            for ep in range(batch_size):
                if len(all_rewards[ep]) == 0:
                    # Episode ended immediately, skip
                    continue
                    
                returns = compute_returns(all_rewards[ep], gamma)
                ep_return = sum(all_rewards[ep])
                episode_returns.append(ep_return)
                
                # Convert to tensors
                returns_t = torch.tensor(returns, dtype=torch.float32, device=device)
                if len(all_log_probs[ep]) > 0:
                    log_probs_t = torch.stack(all_log_probs[ep]).to(device)
                    if log_probs_t.dim() > 1:
                        log_probs_t = log_probs_t.squeeze()
                    
                    batch_log_probs.append(log_probs_t)
                    batch_returns.append(returns_t)

            # Compute loss
            loss = torch.tensor(0.0, device=device)
            for log_prob, G in zip(batch_log_probs, batch_returns):
                loss += torch.sum(-log_prob * G) / len(batch_log_probs)

            # Update policy
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Logging
            avg_return = float(np.mean(episode_returns)) if episode_returns else 0.0
            avg_episode_length = float(np.mean([len(lp) for lp in batch_log_probs])) if batch_log_probs else 0.0
            elapsed = time.time() - start_time
            
            # Track best model
            if avg_return > best_avg_return:
                best_avg_return = avg_return
                best_policy_state = policy_net.state_dict().copy()
                print(
                    f"Update {update_idx:4d} | Loss {loss.item():.4f} | "
                    f"AvgReturn {avg_return:.3f} | AvgLen {avg_episode_length:.1f} | "
                    f"Time {elapsed:.2f}s | *** NEW BEST ***"
                )
                
                # Save best checkpoint immediately
                best_ckpt_path = os.path.join(checkpoint_dir, f"best_{run_id}_{uniq}.pt")
                torch.save(
                    {
                        "policy_state_dict": best_policy_state,
                        "average_return_history": average_return_history + [avg_return],
                        "best_avg_return": best_avg_return,
                        "update_idx": update_idx,
                    },
                    best_ckpt_path,
                )
            else:
                print(
                    f"Update {update_idx:4d} | Loss {loss.item():.4f} | "
                    f"AvgReturn {avg_return:.3f} | AvgLen {avg_episode_length:.1f} | "
                    f"Time {elapsed:.2f}s"
                )

            average_return_history.append(avg_return)
            
            # Periodic checkpoint saving
            # if update_idx % save_freq == 0:
            #     periodic_ckpt_path = os.path.join(
            #         checkpoint_dir, 
            #         f"checkpoint_update{update_idx}_{run_id}_{uniq}.pt"
            #     )
            #     torch.save(
            #         {
            #             "policy_state_dict": policy_net.state_dict(),
            #             "optimizer_state_dict": optimizer.state_dict(),
            #             "average_return_history": average_return_history,
            #             "best_avg_return": best_avg_return,
            #             "best_policy_state": best_policy_state,
            #             "update_idx": update_idx,
            #         },
            #         periodic_ckpt_path,
            #     )
            #     print(f"  → Saved periodic checkpoint to {periodic_ckpt_path}")

    except KeyboardInterrupt:
        print("\n" + "="*60)
        print("Training interrupted by user (Ctrl+C)")
        print("="*60)
        
    # Return results (whether completed or interrupted)
    return average_return_history, best_policy_state, best_avg_return, run_id, uniq


if __name__ == "__main__":
    # Hyperparameters
    trajectories_per_update = 256
    max_seq_len = 300
    gamma = 0.99
    lr = 3e-3
    num_updates = 10000

    print("Initializing environment and policy...")
    env = NumbaWheelPoleSystem(
        n_envs=trajectories_per_update, 
        rod_length=1.0, 
        wheel_radius=0.2,
        wheel_mass=1.0,
        pole_mass=0.1,
        dt=0.02
    )
    policy = PolicyNetwork(
        observation_dim=4,
        hidden_dim=128,
        action_dim=1,
        action_bound=3.0
    )

    print(f"Training with {trajectories_per_update} parallel environments...")
    print(f"Device: {'cuda' if torch.cuda.is_available() else 'cpu'}")
    print(f"Checkpoints will be saved every 100 updates and whenever a new best is found")
    print()

    average_return_history, best_policy_state, best_avg_return, run_id, uniq = train_reinforce(
        policy,
        env,
        batch_size=trajectories_per_update,
        max_seq_len=max_seq_len,
        gamma=gamma,
        lr=lr,
        num_updates=num_updates,
        save_freq=100,  # Save every 100 updates
        checkpoint_dir="checkpoints",
    )

    # Save final checkpoint at end
    os.makedirs("checkpoints", exist_ok=True)
    
    final_ckpt_path = os.path.join("checkpoints", f"final_{run_id}_{uniq}.pt")
    torch.save(
        {
            "policy_state_dict": policy.state_dict(),
            "average_return_history": average_return_history,
            "best_avg_return": best_avg_return,
            "best_policy_state": best_policy_state,
            "hyperparams": {
                "trajectories_per_update": trajectories_per_update,
                "max_seq_len": max_seq_len,
                "gamma": gamma,
                "lr": lr,
                "num_updates": num_updates,
            },
        },
        final_ckpt_path,
    )

    print()
    print("="*60)
    print("TRAINING COMPLETE")
    print("="*60)
    print(f"Best average return: {best_avg_return:.3f}")
    print(f"Final average return: {average_return_history[-1]:.3f}")
    print()
    print("Saved checkpoints:")
    print(f"  BEST:  checkpoints/best_{run_id}_{uniq}.pt")
    print(f"  FINAL: {final_ckpt_path}")
    print("="*60)