import numpy as np
import random
from diseases import *
from hmm_entity import HmmEntity

class MDP:
    def __init__(self, state_name, state_history, actions, transitions, rewards):
        self.state_name=state_name
        self.state_history=state_history
        self.actions = actions
        self.transitions = transitions
        self.rewards = rewards

    def get_next_state(self, state_name, action):
        next_state_probs = self.transitions.get((state_name, action), {})
        if not next_state_probs:
            print('Error.')  # Or handle the error as needed
        next_states = list(next_state_probs.keys())
        probabilities = list(next_state_probs.values())
        return np.random.choice(next_states, p=probabilities)

    def get_reward(self, next_state):
        return self.rewards.get((next_state), 0)

# Define states, actions, and rewards *outside* the class
NON_TERMINAL_ACTIONS=11
actions = [
    'goo_density',
    'goo_pressure',
    'goo_temperature',
    'goo_sound',
    'goo_pain',
    'goo_vibration',
    'goo_perspiration',
    'goo_communication',
    'goo_smell',
    'goo_colour',
    'goo_transparency',
    'guess'
]

actions_history = []

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

state_history={}

rewards = {(a): -1 for a in actions[:NON_TERMINAL_ACTIONS]}
rewards['DIAGNOSED']=5
rewards['DEAD']=-5

#CREATE ENTITY WITH DISEASE AND VALUES FOR EACH SYMPTOM

HMM_DISEASE = DISEASES[random.randint(0, len(DISEASES) - 1)]
HMM_ENTITY = HmmEntity("Hmm1")
HMM_ENTITY.set_disease(HMM_DISEASE)
HMM_ENTITY.disease_randomise()

WEIGHT_IF_UNKNOWN = 100
WEIGHT_TO_GUESS_PERCENTAGE = 0.20
def calculate_probability_function_for_disease(state_p, disease: BaseDisease):
    cost = 0.0
    MAX_COST = 0.0
    # get max
    for key in disease.symptoms_dict:
        # exclude no symptoms
        if disease.symptoms_dict[key].value:
                MAX_COST += WEIGHT_IF_UNKNOWN
    # parse all symptoms
    for key in disease.symptoms_dict:
        # exclude no symptoms
        if disease.symptoms_dict[key].value:
            if state_p[key]:
                #print(state_p[key], disease.symptoms_dict[key].value)
                cost += abs( (state_p[key] - disease.symptoms_dict[key].value) )
            else:
                cost += WEIGHT_IF_UNKNOWN
    return 1.0 - (cost / MAX_COST)

def calculate_prob_functions(state_p):
    disease_prob=[]
    for disease in DISEASES:
        disease_prob.append(calculate_probability_function_for_disease(state_p, disease))

# Define transition probabilities
def create_transition_states(hmm, state_history, state_name, possible_actions):
    transitions = {}
    for idx in range(len(possible_actions)):
        action=possible_actions[idx]
        next_state_name=state_name+action
        state_history[next_state_name]=state_history[state_name].copy()
        if idx<NON_TERMINAL_ACTIONS:
            action_possible_actions=possible_actions.copy()
            action_possible_actions.remove(action)
            state_history[next_state_name][action]=hmm.disease.symptoms_dict[action].value
            transitions[(state_name, action)] = {
                next_state_name: 1.0,
                'DIAGNOSED': 0.0,
                'DEAD': 0.0,
            }

        else:
            disease_name=hmm.disease
            diag_prob=calculate_probability_function_for_disease(state_history[state_name],disease_name)
            death_prob=1-diag_prob
            transitions[(state_name, action)] = {
                'DIAGNOSED': diag_prob,
                'DEAD': death_prob,
            }
    return transitions

state_history["init"]=state.copy()
transitions=create_transition_states(HMM_ENTITY, state_history, "init", actions)

# Create the MDP *after* defining states, actions, etc.
state_name='init'
mdp = MDP(state_name, state_history, actions, transitions, rewards)
