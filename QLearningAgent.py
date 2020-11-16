import numpy as np
import util


class QLearningAgent:
    """
      Q-Learning Agent

      Functions you should fill in:
        - computeValueFromQValues
        - computeActionFromQValues
        - getQValue
        - getAction
        - update

      Instance variables you have access to
        - self.epsilon (exploration prob)
        - self.alpha (learning rate)
        - self.discount (discount rate)

      Functions you should use
        - self.getLegalActions(state)
          which returns legal actions for a state
    """
    def __init__(self, state_space, action_space, epsilon, alpha, discount):
        "You can initialize Q-values here..."
        self.q_values = util.Counter()  # indexed by (state, action) tuples
        self.state_space = state_space
        self.action_space = action_space #range(0, action_space.n)
        self.epsilon = epsilon
        self.alpha = alpha
        self.discount = discount

    def getQValue(self, state, action):
        """
          Returns Q(state,action)
          Should return 0.0 if we have never seen a state
          or the Q node value otherwise
        """
        return self.q_values[(state, action)]
    
    def updateSS(self, newstate):
        self.state_space = np.append(self.state_space,newstate)
        for a in self.action_space:
            self.q_values[(newstate, a)] = 0.0

    def computeValueFromQValues(self, state):
        """
          Returns max_action Q(state,action)
          where the max is over legal actions.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return a value of 0.0.
        """
        actions = self.action_space
        max_value = None
        for action in actions:
            q_value = self.getQValue(state, action)
            if max_value is None:
                max_value = q_value
            if q_value > max_value:
                max_value = q_value
        if max_value is None:
            return 0.0
        else:
            return max_value

    def computeActionFromQValues(self, state):
        """
          Compute the best action to take in a state.  Note that if there
          are no legal actions, which is the case at the terminal state,
          you should return None.
        """
        actions = self.action_space
        max_value = None
        max_value_action = None
        for action in actions:
            q_value = self.getQValue(state, action)
            if max_value is None:
                max_value = q_value
                max_value_action = action
            if q_value > max_value:
                max_value = q_value
                max_value_action = action
            elif q_value == max_value:
                choices = []
                choices.append(max_value_action)
                choices.append(action)
                max_value_action = np.random.choice(choices)
        if max_value is None:
            return None
        else:
            return max_value_action

    def getAction(self, state, useRandom = True):
        """
          Compute the action to take in the current state.  With
          probability self.epsilon, we should take a random action and
          take the best policy action otherwise.  Note that if there are
          no legal actions, which is the case at the terminal state, you
          should choose None as the action.

          HINT: You might want to use util.flipCoin(prob)
          HINT: To pick randomly from a list, use random.choice(list)
        """
        # Pick Action
        legal_actions = self.action_space
        if state not in self.state_space:
            self.updateSS(state)
        
        if legal_actions is None:
            return None
        best_action = self.computeActionFromQValues(state)
        p = np.random.uniform(0, 1)
        if p <= self.epsilon and useRandom:
            action = np.random.choice(legal_actions)
        else:
            action = best_action
        return action

    def update(self, state, action, nextState, reward):
        """
          The parent class calls this to observe a
          state = action => nextState and reward transition.
          You should do your Q-Value update here

          NOTE: You should never call this function,
          it will be called on your behalf
        """
        self.q_values[(state, action)] = (1 - self.alpha) * self.getQValue(state, action) + \
            self.alpha * (reward + self.discount * self.getValue(nextState))

    def getPolicy(self, state):
        return self.computeActionFromQValues(state)

    def getValue(self, state):
        return self.computeValueFromQValues(state)
