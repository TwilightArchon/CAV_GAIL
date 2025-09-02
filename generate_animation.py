import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from Utils.Environment_LC import ENVIRONMENT
from Utils.PPO import PPO
from utils import find_best_model, load_trajectories

def create_animation(Dat, Time_len, lane_wid, save_path):
    """Create an animation of the lane change simulation"""
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

def run_simulation_with_model(ppo_agent, Env, Time_len=500):
    """Run simulation using trained PPO model for lane changes"""
    
    lane_wid = 3.75
    veh_len = 5.0
    
    # Parameters for lane changing
    LC_end_pos = 0.5      # lateral position deviation
    LC_end_yaw = 0.005    # yaw angle deviation 
    
    # Initialize tracking variables
    LC_start = False
    LC_starttime = 0
    LC_endtime = 0
    LC_mid = 0
    
    # Reset environment
    Env.reset()
    
    # Run simulation
    for t in range(1, Time_len):
        s_t, env_t = Env.observe()
        if t != env_t + 1:
            print('warning: time inconsistency!')
            
        Dat = Env.read()
        
        # Handle lane change logic
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
                
            act_1 = 0  # no yaw rate
            action = [act_0, act_1]
        else:
            # Use learned policy during lane change
            state, _ = Env.observe()
            action = ppo_agent.select_action(state)
            print(f"Lane change at t={t}, action: {action}")
            
        Env.run(action)
    
    return Env.read()

def main():
    """Load trained model and generate animation"""
    
    # Initialize parameters (same as in test.py)
    state_dim = 11
    action_dim = 2
    sequence_size = 5
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
    
    # Option 1: Load a specific model
    model_path = "Trained_model/GAIL_1144.pth"  # You can change this path
    
    # Option 2: Find and load the best model (uncomment if you want to use this)
    # print("Loading trajectories to find best model...")
    # trajectories = load_trajectories(
    #     expert_path="Expert_trajectory/expert_traj.npy",
    #     testing_path="Expert_trajectory/testing_traj.npy",
    #     state_dim=state_dim,
    #     action_dim=action_dim,
    #     device=device
    # )
    # model_path, best_mse = find_best_model(
    #     ppo_agent=ppo_agent,
    #     model_dir="Trained_model",
    #     testing_states=trajectories['testing'][0],
    #     testing_actions=trajectories['testing'][1],
    #     start_idx=0,
    #     end_idx=1200
    # )
    # print(f"Best model found at: {model_path} with MSE: {best_mse}")
    
    # Load the model
    print(f"Loading model from: {model_path}")
    ppo_agent.load(model_path)
    ppo_agent.set_action_std(0.00000000001)  # Set very small std for deterministic behavior
    ppo_agent.policy_old.eval()
    
    # Initialize environment
    Env = ENVIRONMENT(para_B="normal", para_A="normal", noise=False)
    
    # Run simulation with trained model
    print("Running simulation with trained model...")
    Dat = run_simulation_with_model(ppo_agent, Env, Time_len=500)
    
    # Create output directory if it doesn't exist
    output_dir = "animations"
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate animation
    save_path = os.path.join(output_dir, "lanechange_with_trained_model.gif")
    create_animation(Dat, Time_len=500, lane_wid=3.75, save_path=save_path)
    
    print(f"Animation generation complete!")

if __name__ == "__main__":
    main()