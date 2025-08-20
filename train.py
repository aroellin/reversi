import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.distributions import Categorical
import numpy as np
import os
import glob
import random
import math

from reversi_game import Reversi

# --- Hyperparameters ---
LEARNING_RATE = 2.5e-4
MCTS_SIMULATIONS_TRAIN = 100
NUM_GAMES_PER_ITERATION = 128
EPOCHS_PER_ITERATION = 4
MINI_BATCH_SIZE = 256
MCTS_TEMPERATURE = 2.0
NUM_ITERATIONS = 10000

# --- Architectural Parameters ---
NUM_FILTERS = 128
NUM_RESIDUAL_BLOCKS = 8

# --- Device Setup ---
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
print(f"Using device: {device}")


# --- MCTS Classes ---
class MCTSNode:
    def __init__(self, parent=None, prior_p=1.0):
        self.parent = parent
        self.children = {}
        self.n_visits = 0
        self.q_value = 0
        self.u_value = 0
        self.p_value = prior_p

    def expand(self, action_priors):
        for action, prior in action_priors.items():
            if action not in self.children:
                self.children[action] = MCTSNode(parent=self, prior_p=prior)

    def select(self, c_puct):
        return max(self.children.items(), key=lambda act_node: act_node[1].get_value(c_puct))

    def update(self, leaf_value):
        self.n_visits += 1
        self.q_value += (leaf_value - self.q_value) / self.n_visits

    def update_recursive(self, leaf_value):
        if self.parent:
            self.parent.update_recursive(-leaf_value)
        self.update(leaf_value)

    def get_value(self, c_puct):
        if self.parent is None:
            return self.q_value
        self.u_value = c_puct * self.p_value * math.sqrt(self.parent.n_visits) / (1 + self.n_visits)
        return self.q_value + self.u_value

    def is_leaf(self):
        return self.children == {}

class MCTS:
    def __init__(self, model, c_puct=1.0, n_simulations=MCTS_SIMULATIONS_TRAIN):
        self.model = model
        self.c_puct = c_puct
        self.n_simulations = n_simulations

    def _get_action_priors(self, game_state):
        state_np = game_state.get_input_planes(game_state.current_player)
        state_tensor = torch.from_numpy(state_np).float().unsqueeze(0).to(device)
        with torch.no_grad():
            logits, value = self.model(state_tensor)
        legal_moves = game_state.get_legal_moves(game_state.current_player)
        if not legal_moves:
            return {}, value.item()
        legal_move_indices = [r * 8 + c for r, c in legal_moves]
        legal_logits = logits[0, legal_move_indices]
        probabilities = F.softmax(legal_logits, dim=0).cpu().numpy()
        return {legal_moves[i]: probabilities[i] for i in range(len(legal_moves))}, value.item()

    def run_simulation(self, root, game_state):
        node = root
        sim_game = Reversi()
        sim_game.board = np.copy(game_state.board)
        sim_game.current_player = game_state.current_player
        while not node.is_leaf():
            action_tuple, node = node.select(self.c_puct)
            sim_game.make_move(action_tuple[0], action_tuple[1], sim_game.current_player)
        action_priors, leaf_value = self._get_action_priors(sim_game)
        if not sim_game.is_game_over():
            node.expand(action_priors)
        else:
            winner = sim_game.get_winner()
            if winner == 0:
                leaf_value = 0.0
            else:
                leaf_value = 1.0 if winner == sim_game.current_player else -1.0
        node.update_recursive(-leaf_value)

    def get_move_policy(self, game_state, temperature=MCTS_TEMPERATURE):
        root = MCTSNode()
        for _ in range(self.n_simulations):
            self.run_simulation(root, game_state)
        legal_moves = game_state.get_legal_moves(game_state.current_player)
        visit_counts = np.array([root.children.get(move, MCTSNode()).n_visits for move in legal_moves])
        if np.sum(visit_counts) == 0:
            return None, None
        move_probs = visit_counts**(1 / temperature)
        move_probs /= np.sum(move_probs)
        policy_target = np.zeros(64)
        for i, move in enumerate(legal_moves):
            policy_target[move[0] * 8 + move[1]] = move_probs[i]
        chosen_index = np.random.choice(len(legal_moves), p=move_probs)
        action = legal_moves[chosen_index]
        return action, policy_target

# --- Model and Helper Functions ---
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += identity
        out = self.relu(out)
        return out

class PolicyValueNet(nn.Module):
    def __init__(self, num_residual_blocks, num_filters):
        super(PolicyValueNet, self).__init__()
        self.conv_stem = nn.Sequential(
            nn.Conv2d(6, num_filters, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_filters),
            nn.ReLU(inplace=True)
        )
        res_blocks = [ResidualBlock(num_filters, num_filters) for _ in range(num_residual_blocks)]
        self.residual_blocks = nn.Sequential(*res_blocks)
        self.policy_head = nn.Sequential(
            nn.Conv2d(num_filters, num_filters, kernel_size=1),
            nn.BatchNorm2d(num_filters),
            nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(num_filters * 8 * 8, 2 * num_filters),
            nn.ReLU(inplace=True),
            nn.Linear(2 * num_filters, 64)
        )
        self.value_head = nn.Sequential(
            nn.Conv2d(num_filters, num_filters, kernel_size=1),
            nn.BatchNorm2d(num_filters),
            nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(num_filters * 8 * 8, 2 * num_filters),
            nn.ReLU(inplace=True),
            nn.Linear(2 * num_filters, 1),
            nn.Tanh()
        )
    def forward(self, x):
        base_output = self.conv_stem(x)
        base_output = self.residual_blocks(base_output)
        policy_logits = self.policy_head(base_output)
        value = self.value_head(base_output).squeeze(-1)
        return policy_logits, value

def get_latest_model_path():
    list_of_files = glob.glob('models/reversi_model_iter_*.pt')
    if not list_of_files:
        return None, 0
    latest_file = max(list_of_files, key=lambda f: int(f.split('_')[-1].split('.')[0]))
    latest_iter = int(latest_file.split('_')[-1].split('.')[0])
    return latest_file, latest_iter

def get_policy_move(model, game_state):
    legal_moves = game_state.get_legal_moves(game_state.current_player)
    if not legal_moves:
        return None
    state_np = game_state.get_input_planes(game_state.current_player)
    state_tensor = torch.from_numpy(state_np).float().unsqueeze(0).to(device)
    with torch.no_grad():
        logits, _ = model(state_tensor)
    legal_move_indices = [r * 8 + c for r, c in legal_moves]
    legal_logits = logits[0, legal_move_indices]
    dist = Categorical(logits=legal_logits)
    action_in_legal = dist.sample()
    return legal_moves[action_in_legal.item()]

def generate_self_play_data(model):
    training_data = []
    mcts = MCTS(model)
    opponent_pool = []
    all_model_files = glob.glob('models/reversi_model_iter_*.pt')
    if all_model_files:
        num_opponents_to_load = min(len(all_model_files), 4)
        opponent_files = random.sample(all_model_files, k=num_opponents_to_load)
        print("--- Loading historical opponent pool: ---")
        for f in opponent_files:
            print(f"  - {os.path.basename(f)}")
            opp = PolicyValueNet(num_residual_blocks=NUM_RESIDUAL_BLOCKS, num_filters=NUM_FILTERS).to(device)
            opp.load_state_dict(torch.load(f, map_location=device))
            opp.eval()
            opponent_pool.append(opp)
    if not opponent_pool:
        num_mcts_self_play = NUM_GAMES_PER_ITERATION
        num_historical_play = 0
    else:
        num_mcts_self_play = int(NUM_GAMES_PER_ITERATION * 0.75)
        num_historical_play = NUM_GAMES_PER_ITERATION - num_mcts_self_play
    print(f"--- Collecting Trajectories: {num_mcts_self_play} MCTS self-play, {num_historical_play} vs historical policies ---")
    for i in range(num_mcts_self_play):
        print(f"  Playing MCTS self-play game {i + 1}/{num_mcts_self_play}...", end='\r')
        game_history = []
        game = Reversi()
        game.new_game(start_type='random')
        while not game.is_game_over():
            player = game.current_player
            action, mcts_policy_target = mcts.get_move_policy(game)
            if action is None:
                game.current_player *= -1
                continue
            game_history.append((game.get_input_planes(player), mcts_policy_target, player))
            game.make_move(action[0], action[1], player)
        winner = game.get_winner()
        for state, mcts_policy, move_player in game_history:
            winner_value = 1 if move_player == winner else -1 if winner != 0 else 0
            training_data.append((state, mcts_policy, winner_value))
    if opponent_pool and num_historical_play > 0:
        for i in range(num_historical_play):
            print(f"  Playing historical policy game {i + 1}/{num_historical_play}...", end='\r')
            opponent_model = random.choice(opponent_pool)
            game_history = []
            game = Reversi()
            game.new_game(start_type='random')
            current_model_player = random.choice([-1, 1])
            while not game.is_game_over():
                player = game.current_player
                if player == current_model_player:
                    action, mcts_policy_target = mcts.get_move_policy(game)
                    if action:
                        game_history.append((game.get_input_planes(player), mcts_policy_target, player))
                else:
                    action = get_policy_move(opponent_model, game)
                if action is None:
                    game.current_player *= -1
                    continue
                game.make_move(action[0], action[1], player)
            winner = game.get_winner()
            for state, mcts_policy, move_player in game_history:
                winner_value = 1 if move_player == winner else -1 if winner != 0 else 0
                training_data.append((state, mcts_policy, winner_value))
    print(f"\nCollection finished. Total samples: {len(training_data)}")
    return training_data

def rotate_policy_target(policy, k):
    if k == 0: return policy
    policy_grid = policy.reshape(8, 8)
    rotated_grid = np.rot90(policy_grid, k)
    return rotated_grid.flatten()

def flip_policy_target(policy):
    policy_grid = policy.reshape(8, 8)
    flipped_grid = np.fliplr(policy_grid)
    return flipped_grid.flatten()

def train_step(model, optimizer, training_data):
    model.train()
    states, policy_targets, value_targets = zip(*training_data)
    dataset = TensorDataset(
        torch.from_numpy(np.array(states)).float(),
        torch.from_numpy(np.array(policy_targets)).float(),
        torch.from_numpy(np.array(value_targets)).float()
    )
    dataloader = DataLoader(dataset, batch_size=MINI_BATCH_SIZE, shuffle=True)
    total_policy_loss = 0
    total_value_loss = 0
    for _ in range(EPOCHS_PER_ITERATION):
        for b_states, b_policies, b_values in dataloader:
            aug_states, aug_policies = b_states.clone(), b_policies.clone()
            for j in range(b_states.shape[0]):
                state_np, policy_np = aug_states[j].numpy(), aug_policies[j].numpy()
                if np.random.rand() > 0.5:
                    state_np = np.flip(state_np, axis=2).copy()
                    policy_np = flip_policy_target(policy_np)
                k = np.random.randint(0, 4)
                if k > 0:
                    state_np = np.rot90(state_np, k, axes=(1, 2)).copy()
                    policy_np = rotate_policy_target(policy_np, k)
                aug_states[j] = torch.from_numpy(state_np)
                aug_policies[j] = torch.from_numpy(policy_np)
            aug_states = aug_states.to(device)
            aug_policies = aug_policies.to(device)
            b_values = b_values.to(device)
            policy_logits, value_pred = model(aug_states)
            value_loss = F.mse_loss(value_pred, b_values)
            policy_log_probs = F.log_softmax(policy_logits, dim=1)
            policy_loss = -torch.sum(aug_policies * policy_log_probs, dim=1).mean()
            loss = policy_loss + value_loss
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_policy_loss += policy_loss.item()
            total_value_loss += value_loss.item()
    avg_p_loss = total_policy_loss / (len(dataloader) * EPOCHS_PER_ITERATION)
    avg_v_loss = total_value_loss / (len(dataloader) * EPOCHS_PER_ITERATION)
    print(f"  Update -> Policy Loss: {avg_p_loss:.4f}, Value Loss: {avg_v_loss:.4f}")

if __name__ == "__main__":
    model = PolicyValueNet(num_residual_blocks=NUM_RESIDUAL_BLOCKS, num_filters=NUM_FILTERS).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, eps=1e-5)
    
    if not os.path.exists('models'):
        os.makedirs('models')
        
    latest_model_path, start_iter = get_latest_model_path()
    
    if latest_model_path:
        print(f"Resuming training from {latest_model_path}")
        model.load_state_dict(torch.load(latest_model_path, map_location=device))
    else:
        print("Starting training from scratch.")
        
    for iteration in range(start_iter, NUM_ITERATIONS):
        print(f"\n===== AlphaZero-Style Iteration {iteration + 1}/{NUM_ITERATIONS} =====")
        model.eval()
        training_data = generate_self_play_data(model)
        if not training_data:
            print("No data collected, skipping update.")
            continue
        train_step(model, optimizer, training_data)
        save_path = f'models/reversi_model_iter_{iteration + 1}.pt'
        torch.save(model.state_dict(), save_path)
        print(f"Model saved to {save_path}")
