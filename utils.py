import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

def load_trajectories(expert_path, testing_path, state_dim, action_dim, device):
    """Load and process expert and testing trajectories."""
    try:
        expert_traj = np.load(expert_path, allow_pickle=True)
        testing_traj = np.load(testing_path, allow_pickle=True)
    except:
        raise Exception("Failed to load trajectory files")

    # Process testing trajectory
    testing_trajectory = testing_traj[0]
    for i in range(1, len(testing_traj)):
        testing_trajectory = np.concatenate((testing_trajectory, testing_traj[i]), axis=0)
    
    testing_trajectory = testing_trajectory.reshape(-1, state_dim + action_dim + state_dim)
    testing_trajectory = torch.FloatTensor(testing_trajectory).to(device)
    testing_states, testing_actions, testing_states_next = torch.split(
        testing_trajectory, 
        (state_dim, action_dim, state_dim), 
        dim=1
    )

    # Process expert trajectory
    expert_trajectory = expert_traj[0]
    for i in range(1, len(expert_traj)):
        expert_trajectory = np.concatenate((expert_trajectory, expert_traj[i]), axis=0)
    
    expert_trajectory = expert_trajectory.reshape(-1, state_dim + action_dim + state_dim)
    expert_trajectory = torch.FloatTensor(expert_trajectory).to(device)
    expert_states, expert_actions, expert_states_next = torch.split(
        expert_trajectory, 
        (state_dim, action_dim, state_dim), 
        dim=1
    )

    return {
        'testing': (testing_states, testing_actions, testing_states_next),
        'expert': (expert_states, expert_actions, expert_states_next),
        'raw_testing': testing_traj
    }

def find_best_model(ppo_agent, model_dir, testing_states, testing_actions, start_idx=0, end_idx=1200):
    """Find the best performing model based on MSE."""
    MSE_test = []
    idx = []
    loss = nn.MSELoss()
    
    for item in range(start_idx, end_idx):
        path = os.path.join(model_dir, f"GAIL_{item}.pth")
        
        if not os.path.exists(path):
            continue

        ppo_agent.load(path)
        ppo_agent.set_action_std(0.00000000001)
        ppo_agent.policy_old.eval()

        actions, _, _ = ppo_agent.policy_old.act(testing_states)
        mse = loss(actions, testing_actions).item()
        
        MSE_test.append(mse)
        idx.append(item)

    best_idx = np.argmin(MSE_test)
    return f"{model_dir}/GAIL_{idx[best_idx]}.pth", MSE_test[best_idx]

def calculate_metrics(predicted, actual):
    """Calculate various metrics between predicted and actual actions."""
    mse = nn.MSELoss()(predicted, actual).item()
    mae = torch.mean(torch.abs(predicted - actual)).item()
    
    # Calculate R² score
    ss_tot = torch.sum((actual - torch.mean(actual))**2)
    ss_res = torch.sum((actual - predicted)**2)
    r2 = (1 - ss_res/ss_tot).item()

    # Calculate KLD for each action dimension
    def calculate_kld(p, q):
        # Convert to numpy for histogram calculation
        p = p.cpu().numpy()
        q = q.cpu().numpy()
        
        # Use more bins and ensure both distributions use exactly the same bins
        bins = 100
        # Compute range that covers both distributions
        min_val = min(p.min(), q.min())
        max_val = max(p.max(), q.max())
        
        # Add small padding to range to avoid edge effects
        range_pad = (max_val - min_val) * 0.1
        bin_range = (min_val - range_pad, max_val + range_pad)
        
        # Calculate histograms
        p_hist, bin_edges = np.histogram(p, bins=bins, range=bin_range, density=True)
        q_hist, _ = np.histogram(q, bins=bins, range=bin_range, density=True)
        
        # Smoothing: Add small constant and normalize
        epsilon = 1e-10
        p_hist = p_hist + epsilon
        q_hist = q_hist + epsilon
        
        # Normalize to ensure proper probability distributions
        p_hist = p_hist / np.sum(p_hist)
        q_hist = q_hist / np.sum(q_hist)
        
        # Calculate KLD
        kld = np.sum(p_hist * np.log(p_hist / q_hist))
        return kld

    kld_acc = calculate_kld(actual[:, 0], predicted[:, 0])
    kld_yaw = calculate_kld(actual[:, 1], predicted[:, 1])

    return {
        'mse': mse,
        'mae': mae,
        'r2': r2,
        'kld_acc': kld_acc,
        'kld_yaw': kld_yaw
    }

def plot_trajectory_comparison(actions, testing_actions, title="Trajectory Comparison", save_dir="trajectory_plots"):
    """Plot comparison between predicted and actual trajectories."""
    # Create trajectory-specific directory
    traj_dir = os.path.join(save_dir, title.replace(" ", "_"))
    os.makedirs(traj_dir, exist_ok=True)
    
    plt.figure(dpi=60, figsize=[36, 4])

    # Plot acceleration
    plt.subplot(1, 2, 1)
    plt.plot(actions[:,0], c='b', linewidth=1, label='GAIL')
    plt.plot(testing_actions[:,0], c='r', linewidth=1, label='Human')
    plt.gca().set_title('Acceleration')
    plt.xlabel('Time (0.1s)')
    plt.ylabel('Acc (m/s^2)')
    plt.legend()

    # Plot yaw rate
    plt.subplot(1, 2, 2)
    plt.plot(actions[:,1], c='b', linewidth=1, label='GAIL')
    plt.plot(testing_actions[:,1], c='r', linewidth=1, label='Human')
    plt.gca().set_title('Yaw rate')
    plt.xlabel('Time (0.1s)')
    plt.ylabel('Yaw rate (rad/s)')
    plt.legend()

    plt.suptitle(title)
    plt.savefig(os.path.join(traj_dir, "trajectory_comparison.png"))
    plt.close()

def evaluate_model(ppo_agent, model_path, testing_traj, state_dim, action_dim, device):
    """Evaluate a specific model on all testing trajectories."""
    ppo_agent.load(model_path)
    ppo_agent.set_action_std(0.00000000001)
    ppo_agent.policy_old.eval()
    loss = nn.MSELoss()

    MSE_PATH = []
    save_dir = "trajectory_plots"
    os.makedirs(save_dir, exist_ok=True)
    
    for i in range(len(testing_traj)):
        testing_states, testing_actions, _ = torch.split(
            torch.FloatTensor(testing_traj[i]).to(device), 
            (state_dim, action_dim, state_dim), 
            dim=1
        )

        actions, _, _ = ppo_agent.policy_old.act(testing_states)
        mse = loss(actions, testing_actions).item()
        MSE_PATH.append(mse)
        
        # Plot and save each trajectory separately
        plot_trajectory_comparison(
            actions.cpu().numpy(),
            testing_actions.cpu().numpy(),
            title=f"Trajectory_{i+1}",
            save_dir=save_dir
        )

    return np.mean(MSE_PATH), MSE_PATH

def plot_all_trajectories(actions, testing_actions, save_dir="trajectory_plots"):
    """Plot all trajectories comparison between predicted and actual trajectories."""
    os.makedirs(save_dir, exist_ok=True)
    
    plt.figure(dpi=60, figsize=[36, 4])

    # Plot acceleration
    plt.subplot(1, 2, 1)
    for i in range(len(actions)):
        plt.plot(actions[i][:,0], c='b', linewidth=1, alpha=0.3)
        plt.plot(testing_actions[i][:,0], c='r', linewidth=1, alpha=0.3)
    
    plt.gca().set_title('Acceleration')
    plt.xlabel('Time (0.1s)')
    plt.ylabel('Acc (m/s^2)')
    plt.plot([], [], c='b', label='GAIL')
    plt.plot([], [], c='r', label='Human')
    plt.legend()

    # Plot yaw rate
    plt.subplot(1, 2, 2)
    for i in range(len(actions)):
        plt.plot(actions[i][:,1], c='b', linewidth=1, alpha=0.3)
        plt.plot(testing_actions[i][:,1], c='r', linewidth=1, alpha=0.3)
    
    plt.gca().set_title('Yaw rate')
    plt.xlabel('Time (0.1s)')
    plt.ylabel('Yaw rate (rad/s)')
    plt.plot([], [], c='b', label='GAIL')
    plt.plot([], [], c='r', label='Human')
    plt.legend()

    plt.savefig(os.path.join(save_dir, "all_trajectories.png"))
    plt.close()

def plot_action_distributions(actions, testing_actions, title="Action Distributions", save_dir="trajectory_plots"):
    """Plot distribution comparison between predicted and actual actions with KLD values."""
    # Create trajectory-specific directory
    traj_dir = os.path.join(save_dir, title.replace(" ", "_"))
    os.makedirs(traj_dir, exist_ok=True)
    
    def calculate_kld(p, q):
        # Convert to numpy for histogram calculation
        p = p.cpu().numpy()
        q = q.cpu().numpy()
        
        # Use more bins and ensure both distributions use exactly the same bins
        bins = 100
        # Compute range that covers both distributions
        min_val = min(p.min(), q.min())
        max_val = max(p.max(), q.max())
        
        # Add small padding to range to avoid edge effects
        range_pad = (max_val - min_val) * 0.1
        bin_range = (min_val - range_pad, max_val + range_pad)
        
        # Calculate histograms
        p_hist, bin_edges = np.histogram(p, bins=bins, range=bin_range, density=True)
        q_hist, _ = np.histogram(q, bins=bins, range=bin_range, density=True)
        
        # Smoothing: Add small constant and normalize
        epsilon = 1e-10
        p_hist = p_hist + epsilon
        q_hist = q_hist + epsilon
        
        # Normalize to ensure proper probability distributions
        p_hist = p_hist / np.sum(p_hist)
        q_hist = q_hist / np.sum(q_hist)
        
        # Calculate KLD
        kld = np.sum(p_hist * np.log(p_hist / q_hist))
        return kld

    kld_acc = calculate_kld(testing_actions[:, 0], actions[:, 0])
    kld_yaw = calculate_kld(testing_actions[:, 1], actions[:, 1])
    
    plt.figure(dpi=60, figsize=[36, 4])

    # Plot acceleration distribution
    plt.subplot(1, 2, 1)
    plt.hist(actions[:,0].cpu().numpy(), bins=50, alpha=0.5, density=True, label='GAIL')
    plt.hist(testing_actions[:,0].cpu().numpy(), bins=50, alpha=0.5, density=True, label='Human')
    plt.gca().set_title(f'Acceleration Distribution\nKLD: {kld_acc:.4f}')
    plt.xlabel('Acceleration (m/s^2)')
    plt.ylabel('Density')
    plt.legend()

    # Plot yaw rate distribution
    plt.subplot(1, 2, 2)
    plt.hist(actions[:,1].cpu().numpy(), bins=50, alpha=0.5, density=True, label='GAIL')
    plt.hist(testing_actions[:,1].cpu().numpy(), bins=50, alpha=0.5, density=True, label='Human')
    plt.gca().set_title(f'Yaw Rate Distribution\nKLD: {kld_yaw:.4f}')
    plt.xlabel('Yaw Rate (rad/s)')
    plt.ylabel('Density')
    plt.legend()

    plt.suptitle(title)
    plt.savefig(os.path.join(traj_dir, "action_distributions.png"))
    plt.close()