import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from Utils.Environment_LC import ENVIRONMENT
from Utils.PPO import PPO

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
        time_text.set_text('')
        return B_line, A_line, E_line, F_line, C_line, D_line, G_line, H_line, time_text

    def animate(num):
        B_line.set_data(Dat[:num, 25], Dat[:num, 12])
        A_line.set_data(lane1[:num], Dat[:num, 3])
        E_line.set_data(lane1[:num], Dat[:num, 0])
        F_line.set_data(lane2[:num], Dat[:num, 9])
        C_line.set_data(lane1[:num], Dat[:num, 6])
        D_line.set_data(lane2[:num], Dat[:num, 15])
        G_line.set_data(lane1[:num], Dat[:num, 18])
        H_line.set_data(lane1[:num], Dat[:num, 21])
        
        # Update time display
        time_sec = num * 0.1  # Each timestep is 0.1 seconds
        time_text.set_text(f'Time: t={num} ({time_sec:.1f}s)\nLat Pos: {Dat[num-1, 25]:.2f}m')
        
        return B_line, A_line, E_line, F_line, C_line, D_line, G_line, H_line, time_text

    fig, ax = plt.subplots()
    B_line, = ax.plot([], [], linewidth=2, label='B (ego)', marker='s', markevery=[-1], linestyle='None')
    A_line, = ax.plot([], [], linewidth=2, label='A', marker='s', markevery=[-1], linestyle='None')
    E_line, = ax.plot([], [], linewidth=2, label='E', marker='s', markevery=[-1], linestyle='None')
    F_line, = ax.plot([], [], linewidth=2, label='F', marker='s', markevery=[-1], linestyle='None')
    C_line, = ax.plot([], [], linewidth=2, label='C', marker='s', markevery=[-1], linestyle='None')
    D_line, = ax.plot([], [], linewidth=2, label='D', marker='s', markevery=[-1], linestyle='None')
    G_line, = ax.plot([], [], linewidth=2, label='G', marker='s', markevery=[-1], linestyle='None')
    H_line, = ax.plot([], [], linewidth=2, label='H', marker='s', markevery=[-1], linestyle='None')
    
    # Add time text display
    time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes, fontsize=12,
                       verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

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

def run_simulation_with_model(ppo_agent, Env, Time_len=500, device='cpu'):
    """Run simulation using trained PPO model - exactly matching GAIL_CAV_Revised notebook"""
    
    # Parameters from training notebook
    LC_end_pos = 0.5      # lateral position deviation (within 0.5m of lane center)
    LC_end_yaw = 0.001    # yaw angle deviation (within 0.005 radians ≈ 0.3 degrees) 
    
    lane_wid = 3.75               
    veh_len  = 5.0
    v_0      = 30   
    
    # Single iteration with normal driver behavior
    A_para = 'normal'
    
    # Environment    
    Env.reset()   
    LC_start = False   
    LC_starttime = 0
    LC_endtime   = 0
    LC_mid       = 0

    for t in range(1, Time_len):  
        s_t, env_t = Env.observe()       #Observation        
        if t != env_t + 1:               # check time consistency between Env and simulation code
            print('warning: time inconsistency!')

        Dat = Env.read()                 # Read ground-truth information
        # print(f"Dat[t-1,24]: {Dat[t-1,24]}")
        # print(f"LC_endangle: {LC_end_yaw}")
        # print(f"Current yaw angle: {Dat[t-1,26]}")
        #Lane change indication
        if Dat[t-1,24]!=0 and LC_start == False and LC_starttime == 0:                 # if LC is true at the end of last time step
            LC_start = True  
            LC_starttime = t
            LC_endtime = 0
            print(f"Lane change started at t={t}")
        # finish lane change - stop in the center of the target lane
        elif abs(Dat[t-1,25] - 0.5*lane_wid) <= LC_end_pos and abs(Dat[t-1,26]) <= LC_end_yaw and LC_start == True and LC_endtime == 0:       
            LC_start = False
            LC_endtime   = t
            print(f"Lane change ended at t={t}, lateral pos={Dat[t-1,25]:.3f}, yaw angle={Dat[t-1,26]:.5f}")
        # out of boundary
        elif (Dat[t-1,25] <= - lane_wid or Dat[t-1,25] > 2.0 * lane_wid) and LC_start == True and LC_endtime == 0:       
            LC_start = False
            LC_endtime   = t
            print(f"Lane change ended (out of boundary) at t={t}, lateral pos={Dat[t-1,25]:.3f}")
        
        # B cross the line    
        if Dat[t-1,25]<=lane_wid and LC_mid==0:
            LC_mid = t         # record the time cross lane-marking
            print(f"Crossed lane marking at t={t}")

        # Low-level task: action              
        if LC_start == False:
            # longitudinal
            if Dat[t-1,25] > lane_wid:    # B in lane 2, B follow F
                act_0 = Env.IDM_B(Dat[t-1,13], Dat[t-1,13] - Dat[t-1,10], Dat[t-1,9] - Dat[t-1,12] - veh_len) #IDM
            elif Dat[t-1,25] <= lane_wid:   # B cross the line, B follow E
                act_0 = Env.IDM_B(Dat[t-1,13], Dat[t-1,13] - Dat[t-1,1], Dat[t-1,0] - Dat[t-1,12] - veh_len) #IDM
            
            # lateral
            act_1 = 0                   # yaw rate
            
            action = [act_0, act_1]
            Env.run(action)
        
        else:
            state, _ = Env.observe()

            action = ppo_agent.select_action(state)
            
            # Check termination conditions and print why it doesn't stop
            target_lat_pos = 0.5 * lane_wid  # 1.875m
            dist_to_target = abs(Dat[t-1,25] - target_lat_pos)
            yaw_angle = abs(Dat[t-1,26])
            
            dist_ok = dist_to_target <= LC_end_pos
            yaw_ok = yaw_angle <= LC_end_yaw
            should_stop = dist_ok and yaw_ok
            
            # Print every 5 timesteps or when conditions change
            if (t - LC_starttime) % 5 == 0 or should_stop:
                print(f"t={t}: lat_pos={Dat[t-1,25]:.3f}m, dist={dist_to_target:.3f}m, yaw={Dat[t-1,26]:.5f}rad")
                print(f"       Dist<={LC_end_pos}m? {dist_ok} | Yaw<={LC_end_yaw}rad? {yaw_ok} | Should STOP? {should_stop}")
                if not should_stop:
                    if not dist_ok:
                        print(f"       ❌ NOT stopping: Distance too large ({dist_to_target:.3f}m > {LC_end_pos}m)")
                    if not yaw_ok:
                        print(f"       ❌ NOT stopping: Yaw angle too large ({yaw_angle:.5f}rad > {LC_end_yaw}rad)")
                else:
                    print(f"       ✅ Both conditions met - will stop at NEXT check!")
            
            Env.run(action) # run human behavior
    
    print(f"Simulation completed. LC_starttime={LC_starttime}, LC_endtime={LC_endtime}, LC_mid={LC_mid}")
    return Env.read()

def main():
    """Load trained model and generate animation - adapted from GAIL_CAV_Revised notebook"""
    
    # Setup device - matching GAIL_CAV_Revised notebook
    device = torch.device("mps" if torch.backends.mps.is_available() else "cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Initialize PPO agent - matching GAIL_CAV_Revised notebook parameters
    state_dim = 11
    action_dim = 2
    action_bound = 2
    lr_actor = 0.001
    lr_critic = 0.001
    gamma = 0.95
    K_epochs = 16
    eps_clip = 0.15
    has_continuous_action_space = True
    action_std_init = 0.4
    
    ppo_agent = PPO(
        state_dim=state_dim,
        action_dim=action_dim,
        action_bound=action_bound,  # Match test.py and notebook approach
        lr_actor=lr_actor,
        lr_critic=lr_critic,
        gamma=gamma,
        K_epochs=K_epochs,
        eps_clip=eps_clip,
        has_continuous_action_space=has_continuous_action_space,
        action_std_init=action_std_init
    )
    
    # Load model - use GAIL_1144 (mentor's working model)
    # model_path = "Trained_model/GAIL_927.pth" pretty good
    # model_path = "Trained_model/GAIL_1110.pth" very good
    # model_path = "Trained_model/GAIL_1160.pth" goes too far left and then too far right
    model_path = "Trained_model/GAIL_113.pth"
    print(f"Loading model from: {model_path}")
    
    ppo_agent.load(model_path)
    ppo_agent.set_action_std(0.00000000001)  # Very small std for deterministic behavior
    ppo_agent.policy_old.eval()
    
    # Initialize environment - MATCH TRAINING PARAMETERS
    # Training uses: para_B='2000', para_A='normal', noise=True (training), noise=False (testing for deterministic results)
    # Note: noise=True randomizes IDM parameters each initialization, causing inconsistent results
    # Use noise=False for deterministic, reproducible testing
    Env = ENVIRONMENT(para_B='2000', para_A='normal', noise=False)
    
    Model_B = Env.IDM_B  # Use IDM_B for longitudinal control
    
    # Run simulation with trained model
    print("Running simulation with trained model...")
    # Note: Model may oscillate around target, out of bounds is expected for undertrained model
    Dat = run_simulation_with_model(ppo_agent, Env, Time_len=500, device=device)
    
    # Create output directory if it doesn't exist
    output_dir = "animations"
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate animation
    save_path = os.path.join(output_dir, "lanechange_with_trained_model.gif")
    create_animation(Dat, Time_len=500, lane_wid=3.75, save_path=save_path)
    
    print(f"Animation generation complete!")

if __name__ == "__main__":
    main()