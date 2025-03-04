import os
import torch
import numpy as np
from Utils.PPO import PPO
from Utils.NeuralNetwork import ActorCritic
from utils import *

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
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
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

    # Evaluate best model on all test trajectories
    print("\nEvaluating best model on all test trajectories...")
    mean_mse, individual_mses = evaluate_model(
        ppo_agent=ppo_agent,
        model_path=best_model_path,
        testing_traj=trajectories['raw_testing'],
        state_dim=state_dim,
        action_dim=action_dim,
        device=device
    )
    print(f"Mean MSE across all trajectories: {mean_mse}")
    print("Individual trajectory MSEs:", individual_mses)

    # Load the best model for detailed analysis
    ppo_agent.load(best_model_path)
    ppo_agent.set_action_std(0.00000000001)
    ppo_agent.policy_old.eval()

    # Get predictions on full test set
    test_states = trajectories['testing'][0]
    test_actions = trajectories['testing'][1]
    predicted_actions, _, _ = ppo_agent.policy_old.act(test_states)

    # Calculate detailed metrics
    metrics = calculate_metrics(predicted_actions, test_actions)
    print("\nDetailed Metrics:")
    print(f"MSE: {metrics['mse']:.6f}")
    print(f"MAE: {metrics['mae']:.6f}")
    print(f"R² Score: {metrics['r2']:.6f}")

    # Create plots directory
    plots_dir = "trajectory_plots"
    os.makedirs(plots_dir, exist_ok=True)

    # Plot individual trajectory comparison
    print("\nPlotting individual trajectory comparison...")
    plot_trajectory_comparison(
        predicted_actions.cpu().numpy(),
        test_actions.cpu().numpy(),
        title="Best_Model_Overall",
        save_dir=plots_dir
    )

    # Plot all trajectories overlay
    print("\nPlotting all trajectories overlay...")
    all_predictions = []
    all_targets = []
    
    for i in range(len(trajectories['raw_testing'])):
        states, actions, _ = torch.split(
            torch.FloatTensor(trajectories['raw_testing'][i]).to(device),
            (state_dim, action_dim, state_dim),
            dim=1
        )
        pred_actions, _, _ = ppo_agent.policy_old.act(states)
        
        # Plot both trajectory comparison and action distributions
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

        all_predictions.append(pred_actions.cpu().numpy())
        all_targets.append(actions.cpu().numpy())

    plot_all_trajectories(
        all_predictions,
        all_targets,
        save_dir=plots_dir
    )

if __name__ == "__main__":
    main() 