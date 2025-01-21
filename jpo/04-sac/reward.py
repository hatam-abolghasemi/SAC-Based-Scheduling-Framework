def calculate_reward(state, target=80, max_deviation=15):
    """
    Calculate the reward for the given state.

    Parameters:
    - state: numpy array of state values (e.g., [cpu_util, memory_util, ...]).
    - target: Target value for the metrics (default is 80%).
    - max_deviation: Maximum allowed deviation from the target (default is 15%).

    Returns:
    - reward: A scalar reward value based on the state.
    """
    reward = 0
    for value in state:
        if 65 <= value <= 95:
            # Positive reward, closer to target is better
            reward += 1 - (abs(value - target) / max_deviation)
        else:
            # Penalty for values outside the range
            if value < 65:
                penalty = (65 - value) / max_deviation
            else:  # value > 95
                penalty = (value - 95) / max_deviation
            reward -= penalty

    return reward

