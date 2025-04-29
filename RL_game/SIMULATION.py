from diseases import *
from hmm_entity import HmmEntity
from saving_hmm_complex_game import *
import random

print('Greetings, human! Welcome to the TSSIH, also known as the Totally Super Secret Intergalactic Hospital.')
print('This is the best and most popular hospital in the galaxy because the only thing that travels faster than light is secrets.')
print('Hmm is an alien who calmly entered the TSSIH for a regular check-up.')
print('However, the intern quickly noticed that Hmm did not appear purple, as was expected.')
print('Although he is just an intern, it was very clear that Hmm was incredibly sick.')
print('Please help the intern figure out the strange disease and save Hmm!')
print('Thankfully, the intern found the perfect book which holds all possible diseases Hmm suffers from, as well their symptoms.')

for disease in DISEASES:
    print(f'{disease.name}: {disease.symptoms}')

print('Good luck! And may the odds be ever in your favour!\n')

# Setup actions and state
actions_available = actions.copy()
actions_history = []

# Initialize entity, disease, and MDP
HMM_DISEASE = DISEASES[random.randint(0, len(DISEASES) - 1)]
HMM_ENTITY = HmmEntity("Hmm1")
HMM_ENTITY.set_disease(HMM_DISEASE)
HMM_ENTITY.disease_randomise()

state = {
    'goo_density': False,
    'goo_pressure': False,
    'goo_temperature': False,
    'goo_sound': False,
    'goo_pain': False,
    'goo_vibration': False,
    'goo_perspiration': False,
    'goo_communication': False,
    'goo_smell': False,
    'goo_colour': False,
    'goo_transparency': False,
    'DIAGNOSED': False,
    'DEAD': False
}

state_history = {"init": state.copy()}
state_name = "init"
transitions = create_transition_states(HMM_ENTITY, state_history, state_name, actions)
mdp = MDP(state_name, state_history, actions, transitions, rewards)

# Simulation loop
while True:
    if len(actions_available) == 0:
        print("No more actions left.")
        break

    print("\nAvailable actions:")
    for a in actions_available:
        print(f" - {a}")
    action = input("Choose an action from above (type exactly): ").strip()

    if action not in actions_available:
        print("Invalid action. Please choose from the list.")
        continue

    actions_available.remove(action)
    print("You are executing action:", action)

    next_state_name = mdp.get_next_state(state_name, action)

    if next_state_name in ["DIAGNOSED", "DEAD"]:
        reward = mdp.get_reward(next_state_name)
        if next_state_name == "DIAGNOSED":
            print(f"Transition: {action} --> Diagnosed, Reward: {reward}")
            print('Success! You have saved Hmm. He can now continue to overpopulate the galaxy!')
        else:
            print(f"Transition: {action} --> Dead, Reward: {reward}")
            print(f'The correct answer was the {HMM_DISEASE.name}.')
            print('Oh dear! You have killed Hmm. Who will tell his 39845 sons?')
        break

    else:
        next_state = state_history[next_state_name]
        reward = mdp.get_reward(action)
        actions_history.append(action)
        print(f"Transition: {action} ---> {next_state}, Reward: {reward}")
        state_name = next_state_name

        # Update transitions and MDP
        transitions = create_transition_states(HMM_ENTITY, state_history, state_name, actions)
        mdp = MDP(state_name, state_history, actions, transitions, rewards)
