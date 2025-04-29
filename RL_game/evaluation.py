import random
from diseases import *
from hmm_entity import HmmEntity
from saving_hmm_complex_game import *
from qlearn import q_table

# Load trained Q-table if needed
# from joblib import load
# q_table = load('q_table.joblib')

def choose_action(state_name, available_actions):
    q_vals = {a: q_table.get((state_name, a), 0) for a in available_actions}
    return max(q_vals, key=q_vals.get)

# Evaluation (test example)
print("\nTesting learned policy...")

HMM_DISEASE = random.choice(DISEASES)
HMM_ENTITY = HmmEntity("Hmm1")
HMM_ENTITY.set_disease(HMM_DISEASE)
HMM_ENTITY.disease_randomise()

state = {a: False for a in actions[:NON_TERMINAL_ACTIONS]}
state['DIAGNOSED'] = False
state['DEAD'] = False
state_history = {"init": state.copy()}
state_name = "init"
actions_available = actions.copy()
transitions = create_transition_states(HMM_ENTITY, state_history, state_name, actions)
mdp = MDP(state_name, state_history, actions, transitions, rewards)

while True:
    action = choose_action(state_name, actions_available)
    actions_available.remove(action)
    print(f"Agent chooses action: {action}")

    next_state_name = mdp.get_next_state(state_name, action)

    if next_state_name not in state_history:
        state_history[next_state_name] = state_history[state_name].copy()

    if state_history[next_state_name]['DIAGNOSED']:
        print("Diagnosed correctly! \U0001F389")
        break
    elif state_history[next_state_name]['DEAD']:
        print(f"Died! The correct diagnosis was: {HMM_DISEASE.name}")
        break

    state_name = next_state_name
    transitions = create_transition_states(HMM_ENTITY, state_history, state_name, actions)
    mdp = MDP(state_name, state_history, actions, transitions, rewards)

# Batch Evaluation
TEST_EPISODES = 50000
success_count = 0
death_count = 0
total_test_reward = 0

for _ in range(TEST_EPISODES):
    HMM_DISEASE = random.choice(DISEASES)
    HMM_ENTITY = HmmEntity("Hmm1")
    HMM_ENTITY.set_disease(HMM_DISEASE)
    HMM_ENTITY.disease_randomise()

    state = {a: False for a in actions[:NON_TERMINAL_ACTIONS]}
    state['DIAGNOSED'] = False
    state['DEAD'] = False
    state_history = {"init": state.copy()}
    state_name = "init"
    actions_available = actions.copy()
    transitions = create_transition_states(HMM_ENTITY, state_history, state_name, actions)
    mdp = MDP(state_name, state_history, actions, transitions, rewards)

    test_reward = 0

    while actions_available:
        action = choose_action(state_name, actions_available)
        actions_available.remove(action)
        next_state_name = mdp.get_next_state(state_name, action)

        if next_state_name not in state_history:
            state_history[next_state_name] = state_history[state_name].copy()

        if state_history[next_state_name]['DIAGNOSED']:
            success_count += 1
            test_reward += mdp.get_reward('DIAGNOSED')
            break
        elif state_history[next_state_name]['DEAD']:
            death_count += 1
            test_reward += mdp.get_reward('DEAD')
            break
        else:
            test_reward += mdp.get_reward(action)

        state_name = next_state_name
        transitions = create_transition_states(HMM_ENTITY, state_history, state_name, actions)
        mdp = MDP(state_name, state_history, actions, transitions, rewards)

    total_test_reward += test_reward

# Final report
print("\nBatch Evaluation Results:")
print(f"Total Test Episodes: {TEST_EPISODES}")
print(f"Successes (Diagnosed): {success_count} ({(success_count/TEST_EPISODES)*100:.2f}%)")
print(f"Deaths: {death_count} ({(death_count/TEST_EPISODES)*100:.2f}%)")
print(f"Average Test Reward: {total_test_reward / TEST_EPISODES:.2f}")
