# GAIL System Documentation

## Table of Contents
1. [System Overview](#system-overview)
2. [Architecture Components](#architecture-components)
3. [Core Algorithms](#core-algorithms)
4. [Neural Network Architecture](#neural-network-architecture)
5. [Environment and Simulation](#environment-and-simulation)
6. [Training Process](#training-process)
7. [Evaluation and Testing](#evaluation-and-testing)
8. [Usage Guide](#usage-guide)
9. [Performance Metrics](#performance-metrics)
10. [Troubleshooting](#troubleshooting)

## System Overview

The GAIL (Generative Adversarial Imitation Learning) system is designed for autonomous vehicle lane change behavior learning. It combines:

- **PPO (Proximal Policy Optimization)** as the base reinforcement learning algorithm
- **Generative Adversarial Networks** for imitation learning from expert demonstrations
- **Connected Autonomous Vehicle (CAV) Environment** for realistic traffic simulation
- **Neural Network Architecture** with separate actor-critic networks for acceleration and yaw control

### Key Features
- **Dual Action Control**: Separate control of longitudinal acceleration and lateral yaw rate
- **Expert Demonstration Learning**: Uses real traffic data to learn safe lane change behaviors
- **Safety Metrics**: Comprehensive evaluation including Time-to-Collision (TTC) and distance metrics
- **Multi-Device Support**: Compatible with CPU, CUDA, and MPS (Apple Silicon) devices

## Architecture Components

### 1. PPO Implementation (`Utils/PPO.py`)

The PPO algorithm serves as the core reinforcement learning component:

#### Key Classes

**RolloutBuffer**
- Stores experience tuples: states, actions, log probabilities, rewards, state values, and terminal flags
- Provides memory management for the PPO training process

**PPO**
- **Initialization**: Sets up actor-critic networks, optimizers, and hyperparameters
- **Action Selection**: Uses the old policy for action selection during training
- **Policy Update**: Implements the clipped objective PPO algorithm
- **Experience Management**: Handles buffer clearing and policy synchronization

#### PPO Hyperparameters
```python
state_dim = 11          # State space dimension
action_dim = 2          # Action space dimension (acceleration, yaw)
sequence_size = 5       # Sequence length for temporal modeling
lr_actor = 0.0003      # Learning rate for actor networks
lr_critic = 0.001      # Learning rate for critic network
gamma = 0.99           # Discount factor
K_epochs = 80          # Number of policy update epochs
eps_clip = 0.2         # PPO clipping parameter
action_std_init = 0.6  # Initial action standard deviation
```

### 2. Neural Network Architecture (`Utils/NeuralNetwork.py`)

#### ActorCritic Network
The main policy network with separate components:

**Actor Networks**
- **Acceleration Actor**: 3-layer MLP (64→64→1) with Tanh activation and Hardtanh output
- **Yaw Rate Actor**: 3-layer MLP (64→64→1) with Tanh activation and Hardtanh output
- **Output Constraints**: Acceleration limited to [-2.0, 2.0] m/s², Yaw rate to [-2.0, 2.0] rad/s

**Critic Network**
- 3-layer MLP (64→64→1) with Tanh activation
- Estimates state values for advantage calculation

**Action Distribution**
- Uses Multivariate Normal distribution for continuous actions
- Handles MPS device compatibility for Apple Silicon

#### Discriminator Network (GAIL)
- 3-layer MLP (state_dim+action_dim → 64 → 64 → 1)
- Sigmoid output for binary classification (expert vs. learned)
- Trained to distinguish between expert and generated trajectories

#### Additional Networks
- **Value Network**: State value estimation for AIRL
- **Rvalue Network**: State-action value estimation for AIRL

### 3. GAIL Implementation (`Utils/GAIL.py`)

#### DISCRIMINATOR_FUNCTION Class
- **Reward Function**: Provides GAIL rewards based on discriminator output
- **Training Loop**: Updates discriminator to distinguish expert from learned trajectories
- **Expert Data Management**: Processes and stores expert trajectory data

#### Key Methods
```python
def reward(self, state, action, epsilon=1e-40):
    # Returns log probability from discriminator as reward
    
def update(self, agent_net, states, actions):
    # Updates discriminator using expert and generated data
    
def save/load(self, checkpoint_path):
    # Model persistence functionality
```

### 4. Environment (`Utils/Environment_LC.py`)

#### Traffic Models
- **IDM (Intelligent Driver Model)**: Car-following behavior
- **MOBIL**: Lane change decision making
- **EIDM**: Enhanced IDM for CAV scenarios

#### ENVIRONMENT Class
- **Multi-lane Traffic Simulation**: 2-lane highway with multiple vehicles
- **Vehicle Dynamics**: Realistic physics including acceleration, velocity, and position
- **Lane Change Logic**: Automatic lane change initiation and completion
- **Safety Constraints**: Collision avoidance and traffic rule enforcement

#### Environment Parameters
```python
veh_len = 5.0          # Vehicle length (meters)
lane_wid = 3.75        # Lane width (meters)
sequence_length = 5     # Temporal sequence length
Time_len = 500         # Simulation duration (timesteps)
```

## Core Algorithms

### PPO Algorithm Flow
1. **Experience Collection**: Collect trajectories using current policy
2. **Advantage Calculation**: Compute advantages using critic network
3. **Policy Update**: Multiple epochs of policy optimization with clipping
4. **Policy Synchronization**: Copy new policy to old policy
5. **Buffer Clearing**: Reset experience buffer for next iteration

### GAIL Training Process
1. **Discriminator Training**: Update discriminator to distinguish expert from generated data
2. **Reward Generation**: Use discriminator output as reward signal
3. **Policy Update**: Update policy using GAIL rewards
4. **Iterative Refinement**: Repeat until convergence

### Lane Change Decision Making
1. **Lane Change Initiation**: Based on MOBIL model and traffic conditions
2. **Trajectory Execution**: Use learned policy during lane change maneuvers
3. **Completion Detection**: Monitor position and yaw angle for lane change completion
4. **Safety Validation**: Ensure minimum distance and TTC requirements

## Training Process

### Data Preparation
- **Expert Trajectories**: Real traffic data for imitation learning
- **Testing Trajectories**: Separate dataset for evaluation
- **Data Format**: State-action-state tuples with proper dimensionality

### Training Loop
1. **Environment Reset**: Initialize traffic scenario
2. **Action Selection**: Use current policy to select actions
3. **Environment Step**: Execute actions and observe results
4. **Experience Storage**: Store transitions in rollout buffer
5. **Policy Update**: Update policy using collected experience
6. **Discriminator Update**: Update GAIL discriminator
7. **Model Checkpointing**: Save model at regular intervals

### Hyperparameter Tuning
- **Learning Rate Scheduling**: Adjust actor and critic learning rates
- **Action Standard Deviation Decay**: Gradually reduce exploration noise
- **PPO Clipping**: Balance exploration and exploitation
- **GAIL Training**: Balance discriminator and policy training

## Evaluation and Testing

### Performance Metrics
1. **Trajectory Prediction**
   - Mean Squared Error (MSE)
   - Mean Absolute Error (MAE)
   - R² Score
   - Kullback-Leibler Divergence

2. **Safety Metrics**
   - Time-to-Collision (TTC)
   - Minimum Distance to Front Vehicle
   - Speed Standard Deviation
   - Lane Change Success Rate

### Evaluation Process
1. **Model Selection**: Find best performing model based on MSE
2. **Trajectory Analysis**: Compare predicted vs. actual trajectories
3. **Safety Assessment**: Run simulation to calculate safety metrics
4. **Visualization**: Generate plots and animations for analysis

## Usage Guide

### Environment Setup
```bash
# Install dependencies
conda env create -f environment.yml

# Activate environment
conda activate cav_gail

# Verify GPU support
python test_gpu.py
```

### Training
```python
# Main training script
python GAIL_CAV_Revised.ipynb

# Key parameters to adjust
state_dim = 11
action_dim = 2
lr_actor = 0.0003
lr_critic = 0.001
gamma = 0.99
K_epochs = 80
```

### Testing and Evaluation
```python
# Run evaluation
python test.py

# This will:
# 1. Load best trained model
# 2. Evaluate trajectory prediction accuracy
# 3. Run safety simulation
# 4. Generate performance metrics and visualizations
```

## Performance Metrics

### Expected Performance
- **MSE**: < 0.01 for well-trained models
- **R² Score**: > 0.8 indicates good prediction accuracy
- **TTC**: > 2.0 seconds for safe operation
- **Distance**: > 20 meters minimum following distance

### Model Selection
- Models are saved as `GAIL_{iteration}.pth`
- Best model selected based on testing MSE
- Checkpointing every 100-1000 iterations recommended

## Troubleshooting

### Common Issues

1. **Memory Issues**
   - Reduce batch size or sequence length
   - Use gradient accumulation
   - Monitor GPU memory usage

2. **Training Instability**
   - Reduce learning rates
   - Increase PPO clipping parameter
   - Adjust action standard deviation

3. **Poor Convergence**
   - Check expert data quality
   - Verify reward scaling
   - Monitor discriminator loss

4. **Device Compatibility**
   - MPS: Apple Silicon Macs
   - CUDA: NVIDIA GPUs
   - CPU: Fallback option

### Debugging Tips
- Monitor training losses and rewards
- Visualize action distributions
- Check environment state consistency
- Validate data preprocessing

### Performance Optimization
- Use appropriate batch sizes
- Enable mixed precision training
- Optimize data loading pipeline
- Profile computational bottlenecks

## File Structure

```
CAV_GAIL/
├── Utils/
│   ├── PPO.py              # PPO implementation
│   ├── GAIL.py             # GAIL discriminator
│   ├── NeuralNetwork.py    # Neural network architectures
│   └── Environment_LC.py   # Traffic simulation environment
├── Expert_trajectory/      # Expert demonstration data
├── Trained_model/          # Saved model checkpoints
├── trajectory_plots/       # Evaluation visualizations
├── test.py                 # Main evaluation script
├── utils.py                # Utility functions
└── environment.yml         # Conda environment specification
```

## Future Improvements

1. **Algorithm Enhancements**
   - Implement SAC or TD3 for better continuous control
   - Add multi-agent training capabilities
   - Incorporate hierarchical reinforcement learning

2. **Environment Improvements**
   - Add more complex traffic scenarios
   - Implement realistic sensor noise
   - Support for different road geometries

3. **Evaluation Metrics**
   - Add more safety metrics
   - Implement human evaluation
   - Real-world testing framework

4. **Performance Optimization**
   - Distributed training support
   - Better memory management
   - Hardware acceleration

## References

- **PPO Paper**: Schulman et al., "Proximal Policy Optimization Algorithms"
- **GAIL Paper**: Ho & Ermon, "Generative Adversarial Imitation Learning"
- **IDM Model**: Treiber et al., "Congested Traffic Flow"
- **MOBIL Model**: Kesting et al., "General Lane-Changing Model"

## Contact and Support

For questions or issues:
1. Check the troubleshooting section
2. Review the code comments
3. Examine the test outputs
4. Verify environment setup

---

*This documentation covers the complete GAIL system implementation for autonomous vehicle lane change behavior learning. The system combines state-of-the-art reinforcement learning with imitation learning to achieve safe and efficient autonomous driving behaviors.*
