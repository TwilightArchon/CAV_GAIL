import os
import torch
import numpy as np
from Utils.PPO import PPO
from utils import load_trajectories

def evaluate_model_mse(model_path, ppo_agent, testing_states, testing_actions):
    """Evaluate a single model and return its MSE"""
    try:
        ppo_agent.load(model_path)
        ppo_agent.policy_old.eval()
        
        with torch.no_grad():
            pred_actions, _, _ = ppo_agent.policy_old.act(testing_states)
            mse = torch.nn.functional.mse_loss(pred_actions, testing_actions).item()
        
        return mse
    except Exception as e:
        print(f"Error evaluating {model_path}: {e}")
        return float('inf')

def main():
    # Setup
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
    
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Initialize PPO agent
    ppo_agent = PPO(
        state_dim=state_dim,
        action_dim=action_dim,
        action_bound=action_bound,
        lr_actor=lr_actor,
        lr_critic=lr_critic,
        gamma=gamma,
        K_epochs=K_epochs,
        eps_clip=eps_clip,
        has_continuous_action_space=has_continuous_action_space,
        action_std_init=action_std_init
    )
    
    # Load testing trajectories
    print("Loading testing trajectories...")
    trajectories = load_trajectories(
        expert_path="Expert_trajectory/expert_traj.npy",
        testing_path="Expert_trajectory/testing_traj.npy",
        state_dim=state_dim,
        action_dim=action_dim,
        device=device
    )
    
    testing_states = trajectories['testing'][0]
    testing_actions = trajectories['testing'][1]
    
    # Get all model files
    model_dir = "Trained_model"
    model_files = [f for f in os.listdir(model_dir) if f.endswith('.pth') and f.startswith('GAIL_')]
    
    # Extract episode numbers and sort
    model_info = []
    for f in model_files:
        try:
            episode = int(f.replace('GAIL_', '').replace('.pth', ''))
            model_info.append((episode, os.path.join(model_dir, f)))
        except:
            continue
    
    model_info.sort()  # Sort by episode number
    
    print(f"\nFound {len(model_info)} models to evaluate")
    print("Evaluating models (this may take a few minutes)...\n")
    
    # Evaluate all models
    results = []
    for i, (episode, model_path) in enumerate(model_info):
        if (i + 1) % 50 == 0:
            print(f"Progress: {i+1}/{len(model_info)} models evaluated...")
        
        mse = evaluate_model_mse(model_path, ppo_agent, testing_states, testing_actions)
        results.append((episode, model_path, mse))
    
    # Sort by MSE (lowest first)
    results.sort(key=lambda x: x[2])
    
    # Print top 10 best models
    print("\n" + "="*70)
    print("🏆 TOP 10 MODELS WITH LOWEST MSE")
    print("="*70)
    print(f"{'Rank':<6} {'Episode':<10} {'MSE':<15} {'Model Path'}")
    print("-"*70)
    
    for rank, (episode, model_path, mse) in enumerate(results[:10], 1):
        print(f"{rank:<6} {episode:<10} {mse:<15.6f} {os.path.basename(model_path)}")
    
    print("="*70)
    
    # Save results to file
    with open('model_evaluation_results.txt', 'w') as f:
        f.write("Model Evaluation Results - Sorted by MSE (Best to Worst)\n")
        f.write("="*70 + "\n")
        f.write(f"{'Rank':<6} {'Episode':<10} {'MSE':<15} {'Model Path'}\n")
        f.write("-"*70 + "\n")
        
        for rank, (episode, model_path, mse) in enumerate(results, 1):
            f.write(f"{rank:<6} {episode:<10} {mse:<15.6f} {os.path.basename(model_path)}\n")
    
    print("\nFull results saved to: model_evaluation_results.txt")
    
    # Print top 10 model paths for easy copy-paste
    print("\n📋 Top 10 Model Paths (for easy copy-paste):")
    print("-"*70)
    for rank, (episode, model_path, mse) in enumerate(results[:10], 1):
        print(f'{rank}. "Trained_model/GAIL_{episode}.pth"  # MSE: {mse:.6f}')

if __name__ == "__main__":
    main()

