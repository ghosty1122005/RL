import numpy as np
import matplotlib.pyplot as plt
import random
from diseases import *
from hmm_entity import HmmEntity
from saving_hmm_complex_game import *

# Q-learning setup
EPISODES = 50_000
LEARNING_RATE = 0.1 # alpha
DISCOUNT = 1.0 # gamma
EPSILON = 0.9
MIN_EPSILON = 0.01 # epsilon_final
EPSILON_DECAY = 0.01 # epsilon_decay
q_table = {}

def choose_action(state_name, available_actions):
    if random.random() < EPSILON:
        return random.choice(available_actions)
    q_vals = {a: q_table.get((state_name, a), 0) for a in available_actions}
    return max(q_vals, key=q_vals.get)

episode_rewards = []
q_value_deltas = []
success_count=0
death_count = 0
TEST_EPISODES = 0
n_tests=0
total_tests=0

# Training loop
for episode in range(EPISODES):
    # Reset state
    state = {a: False for a in actions[:NON_TERMINAL_ACTIONS]}
    state['DIAGNOSED'] = False
    state['DEAD'] = False
    state_history = {'init': state.copy()}
    state_name = 'init'
    actions_available = actions.copy()

    # Initialize entity
    HMM_DISEASE = random.choice(DISEASES)
    HMM_ENTITY = HmmEntity("Hmm1")
    HMM_ENTITY.set_disease(HMM_DISEASE)
    HMM_ENTITY.disease_randomise()

    transitions = create_transition_states(HMM_ENTITY, state_history, state_name, actions)
    mdp = MDP(state_name, state_history, actions, transitions, rewards)

    episode_reward = 0
    total_delta = 0
    step_count = 0

    while actions_available:
        action = choose_action(state_name, actions_available)
        actions_available.remove(action)
        next_state_name = mdp.get_next_state(state_name, action)

        if next_state_name == 'DIAGNOSED':
            reward = mdp.get_reward('DIAGNOSED')
            success_count += 1
            done = True
        elif next_state_name == 'DEAD':
            reward = mdp.get_reward('DEAD')
            death_count += 1
            done = True
        else:
            if next_state_name not in state_history:
                state_history[next_state_name] = state_history[state_name].copy()
            reward = mdp.get_reward(action)
            n_tests+=1
            done = False
        

        future_qs = [q_table.get((next_state_name, a), 0) for a in actions if a in actions_available]
        max_future_q = max(future_qs) if future_qs else 0
        old_q = q_table.get((state_name, action), 0)
        q_table[(state_name, action)] = old_q + LEARNING_RATE * (reward + DISCOUNT * max_future_q - old_q)
        delta = abs(q_table[(state_name, action)] - old_q)
        total_delta += delta
        step_count += 1
        episode_reward += reward       

        if done:
            break

        state_name = next_state_name
        transitions = create_transition_states(HMM_ENTITY, state_history, state_name, actions)
        mdp = MDP(state_name, state_history, actions, transitions, rewards)
    avg_delta = total_delta / step_count if step_count > 0 else 0
    q_value_deltas.append(avg_delta)
    episode_rewards.append(episode_reward)
    EPSILON = max(MIN_EPSILON, EPSILON * EPSILON_DECAY)

print(f"Successes (Diagnosed): {success_count} ({(success_count/EPISODES)*100:.2f}%)")
print(f"Deaths: {death_count} ({(death_count/EPISODES)*100:.2f}%)")
print(f"Average number of tests per episode: {(n_tests/EPISODES):.0f}")
print("Q-learning training complete.")

# Plot rewards
def moving_average(data, window=100):
    return np.convolve(data, np.ones(window)/window, mode='valid')

plt.figure(figsize=(10, 5))
plt.plot(moving_average(episode_rewards), label='Smoothed Reward (avg over 100)', alpha=0.8)
plt.title('Q-learning Performance over Time')
plt.xlabel('Episode')
plt.ylabel('Total Reward')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
