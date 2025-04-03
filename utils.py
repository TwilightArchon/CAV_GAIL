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

def calculate_metrics(predicted, actual, states=None):
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

    # Calculate safety metrics if states are provided
    safety_metrics = {}
    if states is not None:
        # Extract relevant state information
        # states[:, 0] = E position - B position (longitudinal)
        # states[:, 3] = E speed (longitudinal) - This is ego vehicle speed
        # states[:, 6] = B acc
        # states[:, 8] = B speed - This is back vehicle speed
        
        # Calculate time-to-collision (TTC) for front vehicle
        front_distance = states[:, 0].cpu().numpy()  # Distance to front vehicle
        ego_speed = states[:, 3].cpu().numpy()  # Ego vehicle speed (E speed)
        back_speed = states[:, 8].cpu().numpy()  # Back vehicle speed (B speed)
        
        # Calculate relative speed (ego speed relative to front vehicle)
        relative_speed = ego_speed  # Since front_distance is E position - B position, we only need ego speed
        
        # Calculate TTC (avoid division by zero)
        ttc = np.where(relative_speed > 0, front_distance / relative_speed, float('inf'))
        
        # Calculate safety metrics
        safety_metrics['min_ttc'] = np.min(ttc[ttc != float('inf')])  # Minimum TTC
        safety_metrics['mean_ttc'] = np.mean(ttc[ttc != float('inf')])  # Mean TTC
        safety_metrics['min_distance'] = np.min(front_distance)  # Minimum distance to front vehicle
        safety_metrics['mean_distance'] = np.mean(front_distance)  # Mean distance to front vehicle
        safety_metrics['std_ego_speed'] = np.std(ego_speed)  # Standard deviation of ego vehicle speed
        safety_metrics['std_back_speed'] = np.std(back_speed)  # Standard deviation of back vehicle speed
        safety_metrics['std_relative_speed'] = np.std(relative_speed)  # Standard deviation of relative speed

    metrics = {
        'mse': mse,
        'mae': mae,
        'r2': r2,
        'kld_acc': kld_acc,
        'kld_yaw': kld_yaw
    }
    
    # Add safety metrics if they were calculated
    if safety_metrics:
        metrics.update(safety_metrics)
    
    return metrics

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

def plot_safety_metrics(states, title="Safety Metrics", save_dir="trajectory_plots"):
    """Plot safety metrics (TTC and speed std) for a trajectory."""
    traj_dir = os.path.join(save_dir, title.replace(" ", "_"))
    os.makedirs(traj_dir, exist_ok=True)
    
    # Extract relevant state information
    front_distance = states[:, 0].cpu().numpy()  # Distance to front vehicle
    ego_speed = states[:, 3].cpu().numpy()  # Ego vehicle speed (E speed)
    back_speed = states[:, 8].cpu().numpy()  # Back vehicle speed (B speed)
    
    # Calculate relative speed and TTC
    relative_speed = ego_speed  # Since front_distance is E position - B position
    ttc = np.where(relative_speed > 0, front_distance / relative_speed, float('inf'))
    
    # Calculate rolling window statistics
    window_size = 50  # 5 seconds with 0.1s timestep
    ego_speed_std = np.array([np.std(ego_speed[max(0, i-window_size):i+1]) for i in range(len(ego_speed))])
    
    plt.figure(dpi=60, figsize=[36, 4])

    # Plot TTC
    plt.subplot(1, 2, 1)
    plt.plot(ttc, c='b', linewidth=1, label='TTC')
    plt.gca().set_title('Time to Collision')
    plt.xlabel('Time (0.1s)')
    plt.ylabel('TTC (s)')
    plt.legend()
    
    # Plot speed standard deviation
    plt.subplot(1, 2, 2)
    plt.plot(ego_speed_std, c='b', linewidth=1, label='Speed Std')
    plt.gca().set_title('Ego Vehicle Speed Standard Deviation')
    plt.xlabel('Time (0.1s)')
    plt.ylabel('Standard Deviation (m/s)')
    plt.legend()

    plt.suptitle(f"{title} - Safety Metrics")
    plt.savefig(os.path.join(traj_dir, "safety_metrics.png"))
    plt.close()

def evaluate_model(ppo_agent, model_path, testing_traj, state_dim, action_dim, device):
    """Evaluate a specific model on all testing trajectories."""
    ppo_agent.load(model_path)
    ppo_agent.set_action_std(0.00000000001)
    ppo_agent.policy_old.eval()
    loss = nn.MSELoss()

    MSE_PATH = []
    metrics_list = []
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
        
        # Calculate metrics for this trajectory
        trajectory_metrics = calculate_metrics(actions, testing_actions, testing_states)
        metrics_list.append(trajectory_metrics)
        
        # Plot trajectory comparisons
        plot_trajectory_comparison(
            actions.cpu().numpy(),
            testing_actions.cpu().numpy(),
            title=f"Trajectory_{i+1}",
            save_dir=save_dir
        )
        
        # Plot safety metrics
        plot_safety_metrics(
            testing_states,
            title=f"Trajectory_{i+1}",
            save_dir=save_dir
        )

    return np.mean(MSE_PATH), MSE_PATH, metrics_list

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