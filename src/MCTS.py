from src.OED import OED, OEDGymConfig
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy  as np

class MCTSConfig():
    def __init__(self):
        self.max_node = 1000
        # self.max_depth = 50
        self.max_depth = 3
        self.num_layers = 10
        self.hidden_size = 100
        self.gamma = 1 #discount factor
        self.prior_score_scale_factor = 1

class MLPNetwork(nn.Module):
    def __init__(self, nx, ny, num_actions, num_layers, hidden_size):
        super(MLPNetwork, self).__init__()
        input_size = nx * ny

        # Build the common network using nn.Sequential.
        layers = []
        # First layer: from input_size to hidden_size.
        layers.append(nn.Linear(input_size, hidden_size))
        layers.append(nn.ReLU())
        # Add additional hidden layers if num_layers > 1.
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_size, hidden_size))
            layers.append(nn.ReLU())

        self.common = nn.Sequential(*layers)

        # Value head: outputs a single scalar.
        self.value_head = nn.Linear(hidden_size, 1)
        # Policy head: outputs logits for each action.
        self.policy_head = nn.Linear(hidden_size, num_actions)

    def forward(self, x):
        # Flatten the grid input.
        x = x.view(x.size(0), -1)
        common_features = self.common(x)

        # Compute value head.
        value = self.value_head(common_features)

        # Compute policy head and apply softmax to obtain probabilities.
        policy_logits = self.policy_head(common_features)
        policy = F.softmax(policy_logits, dim=1)

        return value, policy

class node:
    def __init__(self, state, parent = None):
        #state that the node referencing to
        self.state = copy.deepcopy(state)
        #node can have 1 parent and multiple children
        #root node is the node that has None as parent
        self.parent = parent
        self.children= {}
        #self.children is a dict that point to the childs, with the key is the action that lead to that child (e.g. 8 if action 8)
        #number of time the node has been visited
        self.N = 0
        #state value predicted by the network
        self.V_s = None


class MCTS:
    def __init__(self, seed, pde_system, gym_config: OEDGymConfig, mcts_config: MCTSConfig):
        self.env = OED(pde_system, gym_config)
        self.seed = seed
        #a mlp network to learn V value and policy distribution
        if gym_config.old_action_space:
            self.action_space = gym_config.n_sensor * (pde_system.nx * pde_system.ny - gym_config.n_sensor)
        else:
            self.action_space = 4 * gym_config.n_sensor
        self.network = MLPNetwork(pde_system.nx, pde_system.ny, self.action_space, mcts_config.num_layers, mcts_config.hidden_size)

        # max node to search
        self.max_node = mcts_config.max_node
        # max depth of the tree before exploring another branch from root
        # with max_node = 1000, max_depth = 50, it explores 20 branches from root
        self.max_depth = mcts_config.max_depth
        #discount factor
        self.gamma = mcts_config.gamma
        #scale down prior_score
        self.prior_scale = mcts_config.prior_score_scale_factor

        #keep track of node explored
        self.node_explored = 0
        #current root node
        self.root = None
        #current branch tracking for backpropagation
        self.current_branch = []
        #keep track of env step

    def train(self,total_timestep= 50000):
        # total_timestep is the time that this function call env.step(), this does not include iterations in tree
        env_state, info = self.env.reset(seed=self.seed)
        learning_step = 0
        while learning_step < total_timestep:
            # choose the best next action
            self.network.eval() # turn network to eval mode
            best_action, action_probs = self.best_action(env_state)
            #todo: some data structure to store env_state and action_probs and update reward till the end of episode

            # step the env according to best action
            env_state, reward, done, truncated, info = self.env.step(best_action)
            learning_step += 1

            if done:
                env_state, info = self.env.reset(seed=self.seed + learning_step)
                # each time the self.best_action is called, we have to wait till the end of the episode to see all the cumulative reward of that action
                # the network will learn to match that env_state to action_probs and cumulative rewards
                self.network.train() #turn network to train mode to learn here
                #todo: training code for neural network
        #save the trained model and the optimizer statedict

    @torch.no_grad()
    def expand(self, parent_node, parent_depth):
        """This function accepts a parent node, and expand to best child according to policy"""
        # add to current branch
        self.current_branch += [parent_node]
        #increase node explored count
        self.node_explored += 1
        #update N visit of parent_node
        parent_node.N += 1
        #call the network to get the value of the parent, and the prior of the childs
        with torch.no_grad():
            state_value, next_state_prior = self.network(torch.tensor(parent_node.state, dtype= torch.float32).unsqueeze(0))
        parent_node.V_s = float(state_value)
        current_depth = parent_depth + 1
        #call the network, get the value of the parent node, and get next state
        if current_depth < self.max_depth:
            #get all the next state
            next_states, rewards = zip(*[self.env.update_state_and_reward(parent_node.state, a) for a in range(self.action_space)])
            #calculate the UCB scores for all childs
            next_state_tensor = torch.tensor(np.stack(next_states), dtype = torch.float32)
            with torch.no_grad():
                next_state_V, _ = self.network(next_state_tensor)
            #get the child visits
            child_N = np.array([parent_node.children.get(a).N if a in parent_node.children.keys() else 0 for a in range(self.action_space)])
            # Q_score = instanteous reward + gamma * V_next_state
            Q_score = np.array(rewards) + self.gamma * np.array(next_state_V).flatten()
            # prior_score = child_prior * sqrt(parent_visit) / (child_visit + 1)
            prior_score = np.array(next_state_prior).flatten() + np.sqrt(parent_node.N) / (child_N + 1)
            #UCB scores =  Q_score + prior_score
            UCB_scores = Q_score + self.prior_scale * prior_score
            #pick the action a = argmax(UCB_scores)
            best_action = np.argmax(UCB_scores)
            #add child to parent children tracking dict
            if best_action in parent_node.children.keys():
                #if node is explored before, reference it
                best_child = parent_node.children[best_action]
            else:
                #create and store
                best_child = node(next_states[best_action])
                parent_node.children[best_action] = best_child
            # recursively expend the child node
            self.expand(best_child, current_depth)


    def backpropagation(self):
        pass

    def best_action(self, env_state):
        #set root node
        self.root = node(env_state)
        self.node_explored = 0
        while self.node_explored < self.max_node:
            #expand the root node, until max depth is reached
            self.expand(self.root, 0)
            assert len(self.current_branch) == self.max_depth
            self.backpropagation()
            #after backprop, current_branch should be empty and ready for the next expansion
            assert len(self.current_branch) == 0

        #choose best action based on frequency visit at root node
        return 1, 2