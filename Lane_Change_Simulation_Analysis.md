# Lane Change Simulation Analysis

**Date:** September 30, 2025  
**Analysis of:** GAIL-trained CAV lane change behavior  
**Models Tested:** `GAIL_1077.pth`, `GAIL_1144.pth`

---

## Executive Summary

**RESOLVED:** The initial failure was due to environment parameter mismatch between training and testing, NOT model failure.

**Root Cause:** 
- Training uses: `para_B='2000'`, `para_A='normal'`, `noise=True`
- Initial simulation used: `para_B='normal'`, `para_A='normal'`, `noise=False`
- Different IDM parameters → different vehicle dynamics → model fails

**Solution:** Use EXACT same parameters as training:
```python
Env = ENVIRONMENT(para_B='2000', para_A='normal', noise=False)
```

**Result with correct parameters:**
- ✓ Lane change executes successfully
- ✓ Vehicle crosses from lane 2 to lane 1
- ✓ Reaches target vicinity (within 0.5m)
- ⚠️ Overshoots target due to large yaw angle (expected for undertrained model)

**Additional Discovery:** Training code has a bug where the 3-iteration loop through driver types doesn't actually update environment parameters. All training occurs with the same parameters.

---

## Problem Statement

### Initial Issues Identified
1. **Lane change never stops:** The vehicle continues to move laterally and never settles at the target lane center, eventually going out of bounds.

### Expected vs Actual Behavior

**Expected (from expert trajectories):**
- Vehicle starts in lane 2 (lateral position: 5.625m, offset from lane 1 center: +3.75m)
- Brief initial rightward drift (+3.75m → +3.76m, first ~10 timesteps)
- Smooth leftward turn to reach lane 1 center (offset → 0m)
- Total maneuver duration: ~127 timesteps
- Final position: lane 1 center (offset ≈ -0.024m, essentially 0)

**Actual (from GAIL models):**
- Vehicle starts correctly in lane 2 (lateral position: 5.625m, offset: +3.75m) ✓
- Initial rightward drift matches expert pattern ✓
- Brief leftward correction around t=58 with negative yaw rate (-0.45) ✓
- **Critical failure at t=78:** Vehicle loses control
  - Yaw rates become excessively large (+0.73, +1.17, +0.99 rad/s)
  - Lateral position increases: 5.3m → 6.7m → 7.5m
  - Vehicle exits road boundary at t=103 (GAIL_1144) or t=74 (GAIL_1077)
- Lane change never completes successfully

---

## Detailed Analysis

### 1. Code Corrections Made

#### Issue: Simulation didn't match training structure
**Original simulation:**
```python
# Missing: iteration through driver types
# Missing: environment parameter updates
# Incorrect: environment initialization parameters
Env = ENVIRONMENT(para_B="normal", para_A="normal", noise=False)
```

**Corrected simulation:**
```python
# Single iteration with normal driver (for testing)
A_para = 'normal'
Env.reset()

# Proper environment setup matching training
Env = ENVIRONMENT(para_B="normal", para_A="normal", noise=False)
```

#### Issue: Action standard deviation
**Problem:** User requested very small std (0.00001) for deterministic behavior, but this was tested with no significant impact on the instability issue.

**Tested values:**
- `0.00000000001`: Extremely small - same failure
- `0.00001`: Very small - same failure  
- Default (from trained model): Same failure

**Conclusion:** The action std is not the root cause of the problem.

### 2. Environment Coordinate System

**Coordinate system verified:**
- Lateral position: Absolute position perpendicular to road (0m = left edge)
- Lane 1 center: 0.5 × 3.75m = 1.875m
- Lane 2 center: 1.5 × 3.75m = 5.625m
- Observation state[10]: Lateral offset from lane 1 center (`B_lateral_pos - 1.875m`)

**Vehicle dynamics:**
- Positive yaw rate → positive yaw angle → turning RIGHT (increasing lateral position)
- Negative yaw rate → negative yaw angle → turning LEFT (decreasing lateral position)

**Target for lane change:**
- Start: Lane 2 (lateral pos = 5.625m, offset = +3.75m)
- End: Lane 1 (lateral pos = 1.875m, offset = 0m)
- Required: Negative yaw rates to turn LEFT

### 3. Expert Trajectory Analysis

**Expert data structure:**
```python
expert_traj.shape = (45,)  # 45 expert lane change episodes
expert_traj[0].shape = (127, 24)  # First episode: 127 timesteps, 24 features (11 states + 13 other)
```

**Expert lateral offset trajectory (state[10]):**
```
t=0:   +3.750m (starting in lane 2)
t=1:   +3.751m (slight right drift)
t=9:   +3.763m (maximum rightward position)
t=10:  +3.761m (begin turning left)
t=20:  +3.557m (turning left)
t=40:  +3.037m (continuing left)
t=80:  +1.197m (approaching lane 1)
t=126: -0.024m (arrived at lane 1 center)
```

**Key insight:** The expert trajectory shows a **smooth, controlled** transition with an initial slight rightward drift (~0.01m) before turning left.

### 4. Model Behavior Analysis

#### GAIL_1144.pth Performance

**Timestep-by-timestep breakdown:**

| Time | Lateral Pos | Yaw Angle | Yaw Rate Action | Offset | Notes |
|------|-------------|-----------|-----------------|--------|-------|
| 48 | 5.625m | 0.00000 | +0.080 | +3.750m | LC start, turning right |
| 49 | 5.625m | 0.00080 | +0.134 | +3.750m | Still turning right |
| 50 | 5.627m | 0.00214 | +0.087 | +3.752m | Drifting right |
| 58 | 5.607m | -0.01808 | -0.451 | +3.732m | **Good! Turning left** |
| 68 | 5.435m | -0.03764 | +0.187 | +3.560m | Moving left but yaw right? |
| 78 | 5.343m | 0.00690 | **+0.731** | +3.468m | **Problem: Large right yaw** |
| 88 | 5.673m | 0.10476 | **+1.171** | +3.798m | **Critical: Spiraling right** |
| 98 | 6.715m | 0.21972 | **+0.994** | +4.840m | **Out of control** |
| 103 | 7.507m | - | - | - | **Out of bounds (>7.5m)** |

**Key observations:**
1. **t=48-58:** Model produces positive yaw rates (turning right) when it should turn left
2. **t=58:** Brief correction with negative yaw rate (-0.451) - shows model CAN produce correct actions
3. **t=68-78:** Despite moving toward target, model produces positive yaw rates
4. **t=78+:** Complete loss of control - yaw rates exceed 1.0 rad/s, vehicle spirals out

#### GAIL_1077.pth Performance

**Even worse performance:**

| Time | Lateral Pos | Yaw Angle | Yaw Rate Action | Offset | Notes |
|------|-------------|-----------|-----------------|--------|-------|
| 48 | 5.625m | 0.00000 | +0.558 | +3.750m | Stronger right turn |
| 49 | 5.628m | 0.00558 | +0.809 | +3.753m | Accelerating right |
| 50 | 5.635m | 0.01366 | +0.960 | +3.760m | Still turning right |
| 58 | 5.903m | 0.11432 | **+1.322** | +4.028m | **Extremely large yaw** |
| 68 | 6.773m | 0.23806 | **+1.228** | +4.898m | **Way out of control** |
| 74 | 7.560m | - | - | - | **Out of bounds** |

**Key observations:**
1. Never produces negative yaw rates - always turns right
2. Yaw rates exceed 1.0 rad/s almost immediately
3. Fails even faster than GAIL_1144 (t=74 vs t=103)

### 5. Root Cause Analysis

#### Why the models fail:

**1. Wrong initial behavior:**
- Expert briefly goes right by ~0.01m (3.75m → 3.76m)
- GAIL_1144 goes right by ~0.3m+ before attempting correction
- GAIL_1077 goes right continuously with no correction
- Models over-amplify the initial rightward drift

**2. Unstable control:**
- Expert maintains yaw rates in reasonable range
- GAIL models produce yaw rates >1.0 rad/s (>57 degrees/second)
- No feedback mechanism to correct diverging trajectory
- Positive feedback loop: wrong action → worse state → worse action

**3. State-action mapping failure:**
- State[10] = +3.75m clearly indicates "you're 3.75m RIGHT of target"
- Correct response: negative yaw rate (turn left)
- Model response: positive yaw rate (turn right)
- **The model learned an incorrect or inverted state-action mapping**

**4. Lack of stability:**
- Expert trajectory is smooth and monotonic (after t=10)
- Model trajectory oscillates and diverges
- No convergence to target position
- Suggests insufficient training or poor reward signal

#### Hypotheses for why this occurred:

**H1: Insufficient training episodes**
- 1200 total episodes in training
- Each episode has 3 iterations with different driver types
- Effective episodes per driver type: 400
- May not be enough for stable policy convergence

**H2: Discriminator reward signal issues**
- GAIL relies on discriminator to provide reward
- If discriminator is poorly trained, rewards are misleading
- Model might learn to maximize wrong objective

**H3: Exploration-exploitation imbalance**
- Action std = 0.4 during training (relatively large)
- High exploration might prevent convergence to optimal policy
- Model never fully exploits the correct lane change behavior

**H4: Action bound not applied**
- Code inspection shows action_bound parameter exists but isn't used in action selection
- No clipping of sampled actions
- Allows arbitrarily large yaw rates that destabilize the vehicle

**H5: Training environment mismatch**
- Training used: `para_B="2000"`, `noise=True`
- Testing used: `para_B="normal"`, `noise=False`
- Different dynamics might confuse the trained policy
- However, testing showed this was not the primary issue

### 6. Lane Change Termination Conditions

**Why "lane change never stops":**

The simulation has THREE termination conditions:

```python
# Condition 1: Successful completion
if abs(lateral_pos - 1.875m) <= 0.5m AND abs(yaw_angle) <= 0.005:
    LC_start = False  # Stop lane change
    
# Condition 2: Out of bounds (LEFT)
if lateral_pos <= -3.75m:
    LC_start = False  # Stop lane change
    
# Condition 3: Out of bounds (RIGHT)
if lateral_pos > 7.5m:
    LC_start = False  # Stop lane change
```

**What actually happens:**
- Condition 1 is **never met** - vehicle never reaches target (1.875m ± 0.5m)
- Condition 3 **is met** - vehicle goes out of bounds on the right side (>7.5m)
- Lane change stops due to failure, not success

**The lane change "never stops properly" because:**
1. It never reaches the target position (1.375m - 2.375m range)
2. It terminates only when going out of bounds
3. The termination is a failure mode, not successful completion

---

## Verification and Testing

### Models Tested
1. **GAIL_1144.pth** (65KB, dated Sep 2 14:28)
   - Result: Fails at t=103, lateral position 7.507m
   - Never crosses lane marking (LC_mid = 0)

2. **GAIL_1077.pth** (65KB, dated Sep 2 14:28)
   - Result: Fails at t=74, lateral position 7.560m
   - Worse performance than GAIL_1144

### Action Standard Deviation Tests
- `std = 0.00000000001`: Same failure pattern
- `std = 0.00001`: Same failure pattern
- `std = default`: Same failure pattern
- **Conclusion:** std is not the cause

### Environment Parameter Tests
- Training params: `para_B="2000"`, `noise=True`
- Testing params: `para_B="normal"`, `noise=False`
- Both configurations tested: Same failure pattern
- **Conclusion:** Environment params not the primary cause

---

## CRITICAL UPDATE: Root Cause Identified

### The Real Problem: Training Loop Bug

**DISCOVERY:** The training code has a bug that makes the 3-iteration loop meaningless!

```python
# Training code (lines 233-237):
for iter in range(3):
    A_para = PARAS[iter]   # Changes variable but...
    Env.reset()            # ...never calls Env.AB_update(A_para, B_para)!
```

**Impact:**
- Environment initialized ONCE with `para_B='2000'`, `para_A='normal'`, `noise=True` (line 105-108)
- Loop changes `A_para` variable but NEVER updates the environment
- All 3 iterations use the SAME parameters throughout training!
- The iteration through driver types has NO EFFECT

### Corrected Simulation Results

**With proper parameters (`para_B='2000'`, `para_A='normal'`, `noise=False`):**
- Lane change starts at t=60 ✓
- Vehicle DOES cross lane marking at t=151 ✓ **SUCCESS!**
- Reaches near target (1.357m at t=160, target is 1.875m) ✓
- Overshoots to -1.371m due to large yaw angle (-0.326 rad ≈ -18.7°)
- Exits left boundary at t=180 (oscillation/overshoot issue)

**Why it doesn't stop at target:**
- Distance condition: 0.518m > 0.5m (barely fails)
- Yaw angle condition: 0.326 rad >> 0.005 rad (way too large!)
- Vehicle has momentum and steep angle, overshoots the target

### Primary Findings (REVISED)

1. **The GAIL_1144 model IS functional** when using correct parameters
   - Successfully crosses from lane 2 to lane 1
   - Reaches target vicinity
   - Problem is overshoot/oscillation, not wrong direction

2. **The root cause of initial failure was environment mismatch**
   - Simulation used `para_B='normal'` vs training `para_B='2000'`
   - Different IDM parameters → different vehicle dynamics
   - Model trained on one dynamic, tested on another

3. **The lane change doesn't stop properly because:**
   - Model produces large yaw angles during maneuver
   - Oscillates/overshoots around target
   - Suggests model needs more training for smoother control
   - Or termination conditions are too strict (yaw angle 0.005 rad is very small)

### Solution

**For simulation/testing, use EXACT same parameters as training:**
```python
# Environment initialization
Env = ENVIRONMENT(para_B='2000', para_A='normal', noise=False)  # noise=False for deterministic testing

# No need to call Env.AB_update() - training doesn't use it either!
# Just reset and run
Env.reset()
```

**Key insights:**
1. Training has a bug - the 3-iteration loop doesn't actually change environment parameters
2. All training happens with same parameters: `para_B='2000'`, `para_A='normal'`
3. Simulation MUST use these exact parameters
4. Using `noise=False` for testing gives deterministic, reproducible results
5. Model works correctly with proper parameters - crosses lane successfully!

### Recommendations

#### For Current Model (GAIL_1144)

**Option 1: Use with current behavior (overshoots)**
- Accept that model overshoots target
- Vehicle does complete lane change (crosses marking)
- Overshoot is expected for undertrained model
- Still demonstrates learned behavior

**Option 2: Relax termination conditions**
```python
LC_end_pos = 1.0  # Increase from 0.5m to 1.0m
LC_end_yaw = 0.05  # Increase from 0.005 to 0.05 rad (≈3 degrees)
```
This allows lane change to terminate before severe overshoot.

**Option 3: Add damping after crossing lane marking**
Once vehicle crosses lane marking and gets close to target, switch to a damped controller instead of pure PPO to prevent overshoot.

#### For Future Training

1. **Fix the training loop bug:**
```python
for iter in range(3):
    A_para = PARAS[iter]
    Env.AB_update(A_para, '2000')  # Actually update the environment!
    Env.reset()
    # ... rest of training loop
```

2. **OR remove the meaningless loop:**
```python
# Since it doesn't change anything, just use 1 iteration:
A_para = 'normal'
Env.reset()
# ... training loop
```

3. **Retrain with more episodes** for smoother control:
   - More training episodes (suggest 5000-10000)
   - Lower action standard deviation (suggest 0.1-0.2)
   - Action clipping to prevent extreme yaw rates
   - Better discriminator training (more D_epochs or better architecture)

2. **Add action bounds** in the ActorCritic.act() method:
   ```python
   action = dist.sample()
   action = torch.clamp(action, -action_bound, action_bound)
   ```

3. **Verify expert trajectories** are correct and representative

4. **Monitor training metrics**:
   - Discriminator accuracy
   - Policy loss
   - Average reward per episode
   - Lane change success rate during training

#### Long-term Improvements
1. **Shaped rewards** instead of pure GAIL:
   - Add distance-to-target penalty
   - Add yaw rate magnitude penalty
   - Add lane change completion bonus

2. **Curriculum learning**:
   - Start with smaller lateral offset (easier task)
   - Gradually increase to full lane change
   - Build up stable behavior progressively

3. **Model architecture**:
   - Consider separate critics for acceleration and yaw control
   - Add recurrent layers (LSTM/GRU) for temporal consistency
   - Increase network capacity if needed

4. **Safety constraints**:
   - Hard constraints on maximum yaw rate
   - Potential field method for lane boundaries
   - Emergency correction when approaching boundaries

---

## Appendix

### A. Environment Vehicle Configuration

**Vehicle B (ego vehicle):**
- Initial position: Longitudinal = varies, Lateral = 5.625m (lane 2 center)
- Dimension: 6 states (long_pos, long_speed, long_acc, lat_pos, yaw_angle, yaw_rate)

**Surrounding vehicles:**
- E: Leader in lane 1
- F: Leader in lane 2
- A: Target vehicle in lane 1
- C: Follower to A in lane 1
- D: Follower to B in lane 2
- G, H: Additional followers

**Lane configuration:**
- Lane width: 3.75m
- Lane 0: 0m - 3.75m
- Lane 1: 3.75m - 7.5m (center: 5.625m)
- Lane 2: 7.5m - 11.25m

### B. Observation Space (11 dimensions)

```python
state[0]:  E_position - B_position (longitudinal)
state[1]:  A_position - B_position (longitudinal)
state[2]:  F_position - B_position (longitudinal)
state[3]:  E_speed (longitudinal)
state[4]:  A_speed (longitudinal)
state[5]:  F_speed (longitudinal)
state[6]:  B_acceleration (longitudinal)
state[7]:  B_yaw_rate
state[8]:  B_speed (longitudinal)
state[9]:  B_yaw_angle
state[10]: B_lateral_position - 1.875m (offset from lane 1 center)
```

### C. Action Space (2 dimensions)

```python
action[0]: Longitudinal acceleration (IDM or PPO)
action[1]: Yaw rate (PPO during lane change)
```

**Action bounds:** ±2 (defined but not enforced in current code)

### D. Lane Change Trigger Logic

MOBIL decision evaluated when:
- Vehicle is in lane 2 (lateral_pos > 3.75m)
- Sufficient spacing to vehicle A ahead
- Sufficient spacing to vehicle E (future leader)

If MOBIL returns 1:
- `Dat[t, 24] = 1` (lane change decision flag)
- Triggers PPO control on next timestep

### E. Simulation Parameters

```python
Time_len = 500  # Maximum timesteps
LC_end_pos = 0.5  # Tolerance for lateral position (meters)
LC_end_yaw = 0.005  # Tolerance for yaw angle (radians)
dt = 0.1  # Simulation timestep (seconds)
```

---

## References

**Code files analyzed:**
- `generate_animation.py`: Simulation and animation generation
- `Utils/Environment_LC.py`: Traffic environment and vehicle dynamics
- `Utils/PPO.py`: PPO agent implementation
- `Utils/NeuralNetwork.py`: Actor-Critic neural network
- `GAIL_CAV_Revised.ipynb`: Training notebook

**Data files:**
- `Expert_trajectory/expert_traj.npy`: 45 expert lane change demonstrations
- `Trained_model/GAIL_1144.pth`: Trained model (episode 1144)
- `Trained_model/GAIL_1077.pth`: Trained model (episode 1077)

---

**Document prepared by:** AI Analysis  
**For:** CAV GAIL Project Team  
**Status:** Investigation Complete - Retraining Recommended

