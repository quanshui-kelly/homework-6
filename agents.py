# Include your imports here, if any are used.

student_name = "Xinyuan Quan"


# 1. Value Iteration
class ValueIterationAgent:
    """Implement Value Iteration Agent using Bellman Equations."""

    def __init__(self, game, discount):
        """Store game object and discount value into the agent object,
        initialize values if needed.
        """
        self.game = game
        self.discount = discount
        self.values = {state: 0.0 for state in game.states}

    def get_value(self, state):
        """Return value V*(s) correspond to state.
        State values should be stored directly for quick retrieval.
        """
        return self.values.get(state, 0.0)

    def get_q_value(self, state, action):
        """Return Q*(s,a) correspond to state and action.
        Q-state values should be computed using Bellman equation:
        Q*(s,a) = Σ_s' T(s,a,s') [R(s,a,s') + γ V*(s')]
        """
        total = 0.0
        transitions = self.game.get_transitions(state, action)
        for next_state, probability in transitions.items():
            reward = self.game.get_reward(state, action, next_state)
            total += probability * (
                reward + self.discount * self.get_value(next_state)
            )
        return total
    
    def get_q_value_for_state(self, state):
        return lambda action: self.get_q_value(state, action)

    def get_best_policy(self, state):
        """Return policy π*(s) correspond to state.
        Policy should be extracted from Q-state values using policy extraction:
        π*(s) = argmax_a Q*(s,a)
        """
        actions = self.game.get_actions(state)
        if not actions:
            return None
        return max(actions, key=self.get_q_value_for_state(state))

    def iterate(self):
        """Run single value iteration using Bellman equation:
        V_{k+1}(s) = max_a Q*(s,a)
        Then update values: V*(s) = V_{k+1}(s)
        """
        new_values = {}
        for state in self.game.states:
            actions = self.game.get_actions(state)
            new_values[state] = (
                max(self.get_q_value(state, action) for action in actions)
                if actions else 0.0
            )
        self.values = new_values


# 2. Policy Iteration
class PolicyIterationAgent(ValueIterationAgent):
    """Implement Policy Iteration Agent.

    The only difference between policy iteration and value iteration is at
    their iteration method. However, if you need to implement helper function
    or override ValueIterationAgent's methods, you can add them as well.
    """
    def __init__(self, game, discount):
        super().__init__(game, discount)
        self.policy = {
            state: next(iter(game.get_actions(state)), None)
            for state in game.states
        }

    def iterate(self):
        """Run single policy iteration.
        Fix current policy, iterate state values V(s) until
        |V_{k+1}(s) - V_k(s)| < ε
        """
        epsilon = 1e-6

        while True:
            new_values = {}
            for state in self.game.states:
                action = self.policy[state]
                new_values[state] = (
                    self.get_q_value(state, action)
                    if action is not None else 0.0
                )

            difference = max(
                (abs(new_values[state] - self.values[state])
                 for state in self.game.states),
                default=0.0
            )
            self.values = new_values
            if difference < epsilon:
                break

        for state in self.game.states:
            actions = self.game.get_actions(state)
            if actions:
                self.policy[state] = max(
                    actions, key=self.get_q_value_for_state(state)
                )


# 3. Bridge Crossing Analysis
def question_3():
    discount = 0.9
    noise = 0.0
    return discount, noise


# 4. Policies
def question_4a():
    discount = 0.1
    noise = 0.0
    living_reward = 0.0
    return discount, noise, living_reward
    # If not possible, return 'NOT POSSIBLE'


def question_4b():
    discount = 0.1
    noise = 0.1
    living_reward = 0.0
    return discount, noise, living_reward
    # If not possible, return 'NOT POSSIBLE'


def question_4c():
    discount = 0.9
    noise = 0.0
    living_reward = 0.0
    return discount, noise, living_reward
    # If not possible, return 'NOT POSSIBLE'


def question_4d():
    discount = 0.9
    noise = 0.2
    living_reward = 0.0
    return discount, noise, living_reward
    # If not possible, return 'NOT POSSIBLE'


def question_4e():
    discount = 0.9
    noise = 0.2
    living_reward = 1.0
    return discount, noise, living_reward
    # If not possible, return 'NOT POSSIBLE'


# 5. Feedback
# Just an approximation is fine.
feedback_question_1 = """
4
"""

feedback_question_2 = """
Understanding policy evaluation and the parameter settings
for the Gridworld questions were the most challenging parts.
"""

feedback_question_3 = """
The Gridworld visualization was helpful for understanding
how value iteration and policy iteration behave.
"""
