import numpy as np
from typing import List, Tuple


def step(mu_prev: np.ndarray,
         emission_probs: np.ndarray,
         transition_probs: np.ndarray,
         observed_state: int) -> Tuple[np.ndarray, np.ndarray]:
    """Runs one step of the Viterbi algorithm.

    Args:
        mu_prev: probability distribution with shape (num_hidden),
            the previous mu
        emission_probs: the emission probability matrix (num_hidden,
            num_observed)
        transition_probs: the transition probability matrix, with
            shape (num_hidden, num_hidden)
        observed_state: the observed state at the current step

    Returns:
        - the mu for the next step
        - the maximizing previous state, before the current state,
          as an int array with shape (num_hidden)
    """

    pre_max = mu_prev * transition_probs.T
    max_prev_states = np.argmax(pre_max, axis=1)
    max_vals = pre_max[np.arange(len(max_prev_states)), max_prev_states]
    mu_new = max_vals * emission_probs[:, observed_state]

    return mu_new, max_prev_states


def viterbi(emission_probs: np.ndarray,
            transition_probs: np.ndarray,
            start_probs: np.ndarray,
            observed_states: List[int]) -> Tuple[List[int], float]:
    """Runs the Viterbi algorithm to get the most likely state sequence.

    Args:
        emission_probs: the emission probability matrix (num_hidden,
            num_observed)
        transition_probs: the transition probability matrix, with
            shape (num_hidden, num_hidden)
        start_probs: the initial probabilies for each state, with shape
            (num_hidden)
        observed_states: the observed states at each step

    Returns:
        - the most likely series of states
        - the joint probability of that series of states and the observed

        @article{
            title    = "Coding the Viterbi Algorithm in Numpy",
            journal  = "Ben's Blog",
            author   = "Benjamin Bolte",
            year     = "2020",
            month    = "03",
            url      = "https://ben.bolte.cc/viterbi",
        }
    """
    num_hidden_states = transition_probs.shape[0]
    num_observed_states = emission_probs.shape[1]

    assert transition_probs.shape == (num_hidden_states, num_hidden_states)
    assert transition_probs.sum(1).mean() == 1
    assert emission_probs.shape == (num_hidden_states, num_observed_states)
    assert emission_probs.sum(1).mean()
    assert start_probs.shape == (num_hidden_states,)

    # Runs the forward pass, storing the most likely previous state.
    mu = start_probs * emission_probs[:, observed_states[0]]
    all_prev_states = []
    for observed_state in observed_states[1:]:
        mu, prevs = step(mu, emission_probs, transition_probs, observed_state)
        all_prev_states.append(prevs)

    # Traces backwards to get the maximum likelihood sequence.
    state = np.argmax(mu)
    sequence_prob = mu[state]
    state_sequence = [state]
    for prev_states in all_prev_states[::-1]:
        state = prev_states[state]
        state_sequence.append(state)

    return state_sequence[::-1], sequence_prob


def _test_viterbi():
    num_hidden_states = 3
    num_observed_states = 2
    num_time_steps = 4
    # Initializes the transition probability matrix (nlatent, nlatent).
    transition_probs = np.array([
        [0.1, 0.2, 0.7],
        [0.1, 0.1, 0.8],
        [0.5, 0.4, 0.1],
    ])
    # Initializes the emission probability matrix. (nlatent, nobs)
    emission_probs = np.array([
        [0.1, 0.9],
        [0.3, 0.7],
        [0.5, 0.5],
    ])
    # Initalizes the initial hidden probabilities (nlatent)
    init_hidden_probs = np.array([0.1, 0.3, 0.6])
    # Defines the sequence of observed states (nsteps)
    observed_states = [1, 1, 0, 1]

    s, p = viterbi(emission_probs, transition_probs, init_hidden_probs, observed_states)
    assert s == [2, 0, 2, 0]
    assert p == 0.0212625
