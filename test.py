import os
import torch
import numpy as np
from Utils.PPO import PPO
from Utils.NeuralNetwork import ActorCritic
from Utils.Environment_LC import ENVIRONMENT
from utils import *
import matplotlib.pyplot as plt
import matplotlib.animation as animation

def main():
    # Initialize parameters
    state_dim = 11
    action_dim = 2
    sequence_size = 5  # From Environment_LC.py
    lr_actor = 0.0003
    lr_critic = 0.001
    gamma = 0.99
    K_epochs = 80
    eps_clip = 0.2
    has_continuous_action_space = True
    action_std_init = 0.6
    
    # Setup device
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Initialize PPO agent
    ppo_agent = PPO(
        state_dim=state_dim,
        action_dim=action_dim,
        sequence_size=sequence_size,
        lr_actor=lr_actor,
        lr_critic=lr_critic,
        gamma=gamma,
        K_epochs=K_epochs,
        eps_clip=eps_clip,
        has_continuous_action_space=has_continuous_action_space,
        action_std_init=action_std_init
    )

    # Load trajectories
    print("Loading trajectories...")
    trajectories = load_trajectories(
        expert_path="Expert_trajectory/expert_traj.npy",
        testing_path="Expert_trajectory/testing_traj.npy",
        state_dim=state_dim,
        action_dim=action_dim,
        device=device
    )

    # Find best model
    print("Finding best model...")
    best_model_path, best_mse = find_best_model(
        ppo_agent=ppo_agent,
        model_dir="Trained_model",
        testing_states=trajectories['testing'][0],
        testing_actions=trajectories['testing'][1],
        start_idx=0,
        end_idx=1200
    )
    print(f"Best model found at: {best_model_path}")
    print(f"Best MSE: {best_mse}")

    # Create plots directory
    plots_dir = "trajectory_plots"
    os.makedirs(plots_dir, exist_ok=True)

    # Load the best model for evaluation
    ppo_agent.load(best_model_path)
    ppo_agent.set_action_std(0.00000000001)
    ppo_agent.policy_old.eval()
    loss = nn.MSELoss()


    # Step 2: Run simulation ONCE for safety metrics using the same approach as training
    print("\nRunning simulation to calculate safety metrics...")
    
    # Initialize environment parameters
    Env = ENVIRONMENT(para_B="normal", para_A="normal", noise=False)
    veh_len = Env.veh_len
    lane_wid = Env.lane_wid
    
    # Parameters for lane changing
    LC_end_pos = 0.5      # lateral position deviation
    LC_end_yaw = 0.005    # yaw angle deviation 
    
    # Initialize tracking variables
    LC_start = False
    LC_starttime = 0
    LC_endtime = 0
    LC_mid = 0

    path_idx    = 0
    Time_len = 500
    lane_wid = 3.75               
    veh_len  = 5.0
    
    # Initialize arrays to track metrics
    ttc_values = []
    distances = []
    ego_speeds = []

    
    # Run the simulation for the full time loop
    Time_len = Env.Time_len
    
    for t in range(1, 500):
        # Get observation and ground truth data
        s_t, env_t = Env.observe()
        if t != env_t + 1:
            print('warning: time inconsistency!')
            
        Dat = Env.read()
            
        # Record metrics using the data from previous timestep (as in training)
        # Speeds
        b_speed = Dat[t-1, 13]  # B speed
        ego_speeds.append(b_speed)
        
        # Calculate distance and TTC based on vehicle positions
        b_lateral = Dat[t-1, 25]  # B lateral position
        
        # Determine front vehicle based on lateral position
        if b_lateral > lane_wid:  # B is in lane 2, B follows F
            front_pos = Dat[t-1, 9]      # F position
            front_speed = Dat[t-1, 10]   # F speed
            b_pos = Dat[t-1, 12]         # B position
            distance = front_pos - b_pos - veh_len
        else:  # B is in/crossing to lane 1, B follows E
            front_pos = Dat[t-1, 0]      # E position
            front_speed = Dat[t-1, 1]    # E speed
            b_pos = Dat[t-1, 12]         # B position
            distance = front_pos - b_pos - veh_len
        
        # Record positive distances only
        
        # Calculate TTC only when approaching
        relative_speed = b_speed - front_speed
        if relative_speed > 0 and distance > 0:
            ttc = distance / relative_speed
            ttc_values.append(ttc)
        
        # Handle lane change logic as in the training code
        if Dat[t-1, 24] != 0 and LC_start == False and LC_starttime == 0:
            LC_start = True
            LC_starttime = t
        elif abs(Dat[t-1, 25] - 0.5 * lane_wid) <= LC_end_pos and abs(Dat[t-1, 26]) <= LC_end_yaw and LC_start == True and LC_endtime == 0:
            LC_start = False
            LC_endtime = t
        elif (Dat[t-1, 25] <= -lane_wid or Dat[t-1, 25] > 2.0 * lane_wid) and LC_start == True and LC_endtime == 0:
            LC_start = False
            LC_endtime = t
            
        # B cross the line
        if Dat[t-1, 25] <= lane_wid and LC_mid == 0:
            LC_mid = t
            
        # Determine action based on lane change status
        if LC_start == False:
            # Use IDM model based on lane position
            if Dat[t-1, 25] > lane_wid:  # B in lane 2, B follows F
                act_0 = Env.IDM_B(Dat[t-1, 13], Dat[t-1, 13] - Dat[t-1, 10], Dat[t-1, 9] - Dat[t-1, 12] - veh_len)
            else:  # B cross the line, B follow E
                act_0 = Env.IDM_B(Dat[t-1, 13], Dat[t-1, 13] - Dat[t-1, 1], Dat[t-1, 0] - Dat[t-1, 12] - veh_len)
                
            # No lateral movement
            act_1 = 0  # yaw rate
            action = [act_0, act_1]
        else:
            # Use learned policy during lane change
            state, _ = Env.observe()
            action = ppo_agent.select_action(state)
            print(action)
            
        # Run the environment with the action
        Env.run(action)
    
    # Create animation of the simulation
    Dat = Env.read()
    create_animation(Dat, Time_len, lane_wid, os.path.join(plots_dir, "lanechange.gif"))
    
    # Calculate safety metrics from the results
    print(f"Total time steps with valid TTC values: {len(ttc_values)}")

    # Step 1: Evaluate MSE for all trajectories
    individual_mses = []
    r2_scores = []
    mae_scores = []
    all_predictions = []
    all_targets = []

    print("\nEvaluating MSE for all trajectories...")
    for i in range(len(trajectories['raw_testing'])):
        print(f"Evaluating trajectory {i+1}...")
        
        # Extract states and actions from test trajectory
        states, actions, _ = torch.split(
            torch.FloatTensor(trajectories['raw_testing'][i]).to(device),
            (state_dim, action_dim, state_dim),
            dim=1
        )
        
        # Get predicted actions
        pred_actions, _, _ = ppo_agent.policy_old.act(states)
        
        # Calculate metrics
        mse = loss(pred_actions, actions).item()
        mae = torch.mean(torch.abs(pred_actions - actions)).item()
        r2 = (1 - loss(pred_actions, actions) / loss(actions, actions.mean(dim=0).expand_as(actions))).item()
        
        individual_mses.append(mse)
        mae_scores.append(mae)
        r2_scores.append(r2)
        
        # Generate plots
        plot_trajectory_comparison(
            pred_actions.cpu().numpy(),
            actions.cpu().numpy(),
            title=f"Trajectory_{i+1}",
            save_dir=plots_dir
        )
        
        plot_action_distributions(
            pred_actions,
            actions,
            title=f"Trajectory_{i+1}",
            save_dir=plots_dir
        )
        
        # Store predictions and targets for final plot
        all_predictions.append(pred_actions.cpu().numpy())
        all_targets.append(actions.cpu().numpy())
    
    # Calculate overall MSE statistics
    mean_mse = np.mean(individual_mses)
    mean_mae = np.mean(mae_scores)
    mean_r2 = np.mean(r2_scores)
    
    
    # No filtering - preserve original TTC values for a more accurate representation
    safety_metrics = {}
    if distances:
        safety_metrics['min_distance'] = min(distances)
        safety_metrics['mean_distance'] = np.mean(distances)
    else:
        safety_metrics['min_distance'] = float('inf')
        safety_metrics['mean_distance'] = float('inf')
    
    safety_metrics['std_ego_speed'] = np.std(ego_speeds) if ego_speeds else 0
    
    # Add TTC metrics if we have valid TTC values
    if ttc_values:
        safety_metrics['min_ttc'] = min(ttc_values)
        safety_metrics['mean_ttc'] = np.mean(ttc_values)
    else:
        safety_metrics['min_ttc'] = float('inf')
        safety_metrics['mean_ttc'] = float('inf')
    
    # Combine metrics
    detailed_metrics = {
        'mse': mean_mse,
        'mae': mean_mae,
        'r2': mean_r2,
        'min_ttc': safety_metrics['min_ttc'],
        'mean_ttc': safety_metrics['mean_ttc'],
        'min_distance': safety_metrics['min_distance'],
        'mean_distance': safety_metrics['mean_distance'],
        'std_ego_speed': safety_metrics['std_ego_speed']
    }
    
    # Print metrics
    print("\nTrajectory Prediction Metrics:")
    print(f"Mean MSE across all trajectories: {mean_mse:.6f}")
    print(f"Mean MAE across all trajectories: {mean_mae:.6f}")
    print(f"Mean R² Score across all trajectories: {mean_r2:.6f}")
    
    print("\nSafety Metrics (Single Simulation):")
    print(f"Minimum Time-to-Collision (TTC): {safety_metrics['min_ttc']:.2f} seconds")
    print(f"Mean Time-to-Collision (TTC): {safety_metrics['mean_ttc']:.2f} seconds")
    print(f"Minimum Distance to Front Vehicle: {safety_metrics['min_distance']:.2f} meters")
    print(f"Mean Distance to Front Vehicle: {safety_metrics['mean_distance']:.2f} meters")
    print(f"Standard Deviation of Ego Vehicle Speed: {safety_metrics['std_ego_speed']:.2f} m/s")

    # Plot all trajectories overlay
    plot_all_trajectories(
        all_predictions,
        all_targets,
        save_dir=plots_dir
    )

def create_animation(Dat, Time_len, lane_wid, save_path):
    """Create an animation of the lane change simulation using the provided code"""
    lane1 = np.full(Time_len, lane_wid*0.5)
    lane2 = np.full(Time_len, lane_wid*1.5)

    def init():
        ax.set_xlim(-lane_wid, lane_wid*3.5)
        ax.set_ylim(25, 550)
        ax.axvline(x=0, c='black', linestyle='--', linewidth=2)
        ax.axvline(x=lane_wid, c='black', linestyle='--', linewidth=2)
        ax.axvline(x=lane_wid + lane_wid, c='black', linestyle='--', linewidth=2)
        ax.axvline(x=lane_wid + lane_wid + lane_wid, c='black', linestyle='--', linewidth=2)
        ax.set_xlabel('Lateral Position (m)')
        ax.set_ylabel('Longitudinal Position (m)')
        return B_line, A_line, E_line, F_line, C_line, D_line, G_line, H_line

    def animate(num):
        B_line.set_data(Dat[:num, 25], Dat[:num, 12])
        A_line.set_data(lane1[:num], Dat[:num, 3])
        E_line.set_data(lane1[:num], Dat[:num, 0])
        F_line.set_data(lane2[:num], Dat[:num, 9])
        C_line.set_data(lane1[:num], Dat[:num, 6])
        D_line.set_data(lane2[:num], Dat[:num, 15])
        G_line.set_data(lane1[:num], Dat[:num, 18])
        H_line.set_data(lane1[:num], Dat[:num, 21])
        return B_line, A_line, E_line, F_line, C_line, D_line, G_line, H_line

    fig, ax = plt.subplots()
    B_line, = ax.plot([], [], linewidth=2, label='B (ego)', marker='s', markevery=[-1], linestyle='None')
    A_line, = ax.plot([], [], linewidth=2, label='A', marker='s', markevery=[-1], linestyle='None')
    E_line, = ax.plot([], [], linewidth=2, label='E', marker='s', markevery=[-1], linestyle='None')
    F_line, = ax.plot([], [], linewidth=2, label='F', marker='s', markevery=[-1], linestyle='None')
    C_line, = ax.plot([], [], linewidth=2, label='C', marker='s', markevery=[-1], linestyle='None')
    D_line, = ax.plot([], [], linewidth=2, label='D', marker='s', markevery=[-1], linestyle='None')
    G_line, = ax.plot([], [], linewidth=2, label='G', marker='s', markevery=[-1], linestyle='None')
    H_line, = ax.plot([], [], linewidth=2, label='H', marker='s', markevery=[-1], linestyle='None')

    ani = animation.FuncAnimation(
        fig,    
        animate, 
        frames=Time_len, 
        blit=True, 
        interval=20, 
        repeat=False, 
        init_func=init
    )

    plt.gca().set_title('Lane Change Animation')
    plt.legend(loc='upper right')

    ani.save(save_path)
    plt.close()
    print(f"Animation saved to {save_path}")

if __name__ == "__main__":
    main() 