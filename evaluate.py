# FILE: evaluate.py
import torch
import torch.nn.functional as F
import numpy as np
import glob
import os
import argparse
import random
import math
import time
from collections import defaultdict
import itertools

# Import necessary classes from existing project files
from reversi_game import Reversi
from train import PolicyValueNet, NUM_FILTERS, NUM_RESIDUAL_BLOCKS

# --- ELO Calculation Class ---
class EloCalculator:
    """Calculates and stores ELO ratings for a set of players."""
    def __init__(self, k_factor=32, default_elo=1000):
        self.k_factor = k_factor
        self.default_elo = default_elo
        self.ratings = defaultdict(lambda: self.default_elo)
        self.history = []

    def record_game(self, player1_name, player2_name, winner):
        """
        Records a single game result.
        winner: 1 if player1 wins, 2 if player2 wins, 0 for a draw.
        """
        if winner == 1: score1, score2 = 1.0, 0.0
        elif winner == 2: score1, score2 = 0.0, 1.0
        else: score1, score2 = 0.5, 0.5
        self.history.append((player1_name, player2_name, score1, score2))
        _ = self.ratings[player1_name]; _ = self.ratings[player2_name]

    def _update_ratings(self, p1_name, p2_name, s1, s2):
        r1, r2 = self.ratings[p1_name], self.ratings[p2_name]
        e1 = 1 / (1 + 10**((r2 - r1) / 400)); e2 = 1 - e1
        self.ratings[p1_name] += self.k_factor * (s1 - e1)
        self.ratings[p2_name] += self.k_factor * (s2 - e2)

    def calculate_elos(self, iterations=100):
        print("\nCalculating ELO ratings...")
        for player in self.ratings: self.ratings[player] = self.default_elo
        history_copy = list(self.history)
        for i in range(iterations):
            random.shuffle(history_copy)
            for p1, p2, s1, s2 in history_copy: self._update_ratings(p1, p2, s1, s2)
            print(f"  ELO calculation iteration {i+1}/{iterations}", end='\r')
        print("\nELO calculation finished.")

    def display_ratings(self):
        print("\n" + "*" * 25 + " ELO LEADERBOARD " + "*" * 25)
        sorted_players = sorted(self.ratings.items(), key=lambda item: item[1], reverse=True)
        print(f"{'Rank':<5} {'Player':<50} {'ELO Rating':<10}")
        print("-" * 70)
        for i, (player, rating) in enumerate(sorted_players):
            print(f"{i+1:<5} {player:<50} {int(rating):<10}")
        print("*" * 70 + "\n")


# --- MCTS Classes ---
class MCTSNode:
    def __init__(self, parent=None, prior_p=1.0):
        self.parent = parent; self.children = {}; self.n_visits = 0; self.q_value = 0; self.u_value = 0; self.p_value = prior_p
    def expand(self, action_priors):
        for action, prior in action_priors.items():
            if action not in self.children: self.children[action] = MCTSNode(parent=self, prior_p=prior)
    def select(self, c_puct):
        return max(self.children.items(), key=lambda act_node: act_node[1].get_value(c_puct))
    def update(self, leaf_value):
        self.n_visits += 1; self.q_value += (leaf_value - self.q_value) / self.n_visits
    def update_recursive(self, leaf_value):
        if self.parent: self.parent.update_recursive(-leaf_value)
        self.update(leaf_value)
    def get_value(self, c_puct):
        if self.parent is None: return self.q_value
        self.u_value = c_puct * self.p_value * math.sqrt(self.parent.n_visits) / (1 + self.n_visits)
        return self.q_value + self.u_value
    def is_leaf(self): return self.children == {}

class MCTS:
    def __init__(self, model, device, c_puct=1.0):
        self.model = model; self.device = device; self.c_puct = c_puct; self.root = MCTSNode()
    def _get_priors_and_value(self, game_state):
        state_np = game_state.get_input_planes(game_state.current_player); state_tensor = torch.from_numpy(state_np).float().unsqueeze(0).to(self.device)
        with torch.no_grad(): logits, value = self.model(state_tensor)
        legal_moves = game_state.get_legal_moves(game_state.current_player)
        if not legal_moves: return {}, value.item()
        legal_move_indices = [r * 8 + c for r, c in legal_moves]; legal_logits = logits[0, legal_move_indices]
        probabilities = F.softmax(legal_logits, dim=0).cpu().numpy()
        return {legal_moves[i]: probabilities[i] for i in range(len(legal_moves))}, value.item()
    def _run_simulation(self, game_state):
        node = self.root; sim_game = Reversi(); sim_game.board = np.copy(game_state.board); sim_game.current_player = game_state.current_player
        while not node.is_leaf():
            action_tuple, node = node.select(self.c_puct)
            sim_game.make_move(action_tuple[0], action_tuple[1], sim_game.current_player)
        action_priors, leaf_value = self._get_priors_and_value(sim_game)
        if not sim_game.is_game_over(): node.expand(action_priors)
        else:
            winner = sim_game.get_winner()
            leaf_value = 0.0 if winner == 0 else 1.0 if winner == sim_game.current_player else -1.0
        node.update_recursive(-leaf_value)
    def search(self, game_state, n_sims):
        self.root = MCTSNode()
        for _ in range(n_sims): self._run_simulation(game_state)
    def get_best_move(self):
        if not self.root.children: return None
        return max(self.root.children.items(), key=lambda item: item[1].n_visits)[0]
    def get_stochastic_move(self, temperature=1.0):
        if not self.root.children: return None
        moves, nodes = zip(*self.root.children.items())
        visit_counts = np.array([node.n_visits for node in nodes])
        if np.sum(visit_counts) == 0: return random.choice(moves)
        move_probs = visit_counts**(1 / temperature)
        move_probs /= np.sum(move_probs)
        return moves[np.random.choice(len(moves), p=move_probs)]


# --- Agent and Game Logic ---
class Agent:
    def __init__(self, model, device, play_style, mcts_sims=None):
        self.model = model; self.device = device; self.play_style = play_style; self.mcts_sims = mcts_sims
        if play_style.startswith('mcts'):
            if mcts_sims is None: raise ValueError("MCTS play style requires mcts_sims.")
            self.mcts = MCTS(model, device, c_puct=1.0)

    def get_name(self):
        name = self.play_style.replace('_', ' ').title()
        if self.mcts_sims: name += f" ({self.mcts_sims} sims)"
        return name

    def get_move(self, game):
        legal_moves = game.get_legal_moves(game.current_player)
        if not legal_moves: return None
        if self.play_style in ['best_policy', 'random_policy']:
            state_np = game.get_input_planes(game.current_player)
            state_tensor = torch.from_numpy(state_np).float().unsqueeze(0).to(self.device)
            with torch.no_grad(): logits, _ = self.model(state_tensor)
            legal_move_indices = [r * 8 + c for r, c in legal_moves]
            legal_logits = logits[0, legal_move_indices]
            if self.play_style == 'best_policy': return legal_moves[torch.argmax(legal_logits).item()]
            else:
                probabilities = F.softmax(legal_logits, dim=0).cpu().numpy()
                return legal_moves[np.random.choice(len(legal_moves), p=probabilities)]
        elif self.play_style.startswith('mcts'):
            self.mcts.search(game, self.mcts_sims)
            return self.mcts.get_best_move() if self.play_style == 'mcts_best' else self.mcts.get_stochastic_move()
        else: raise ValueError(f"Unknown play style: {self.play_style}")

def play_game(agent1, agent2, start_type):
    game = Reversi(); game.new_game(start_type=start_type)
    players = {-1: agent1, 1: agent2};
    if random.random() < 0.5: players = {-1: agent2, 1: agent1}
    while not game.is_game_over():
        move = players[game.current_player].get_move(game)
        if move: game.make_move(move[0], move[1], game.current_player)
        else: game.current_player *= -1
    winner = game.get_winner()
    if winner == 0: return 0
    return 1 if players[winner] is agent1 else 2

def run_matchup(agent1, agent2, num_games, elo_calculator):
    agent1_name, agent2_name = agent1.get_name(), agent2.get_name()
    print("-" * 80); print(f"MATCHUP: {agent1_name} vs {agent2_name}"); print(f"GAMES: {num_games}")
    scores = defaultdict(int); start_time = time.time()
    for i in range(num_games):
        result = play_game(agent1, agent2, 'standard' if i % 2 == 0 else 'adjacent')
        scores[result] += 1
        elo_calculator.record_game(agent1_name, agent2_name, result)
        print(f"  Game {i+1}/{num_games} finished... Score: {agent1_name[:15]}.. {scores[1]}-{scores[2]} {agent2_name[:15]}..", end='\r')
    duration = time.time() - start_time; win_rate = (scores[1] / num_games) * 100 if num_games > 0 else 0
    print("\n" + "="*25 + " RESULTS " + "="*25)
    print(f"  {agent1_name} Wins: {scores[1]}\n  {agent2_name} Wins: {scores[2]}\n  Draws: {scores[0]}")
    print(f"  {agent1_name} Win Rate: {win_rate:.1f}%\n  Total time: {duration:.2f}s ({duration/num_games:.2f}s/game)\n")

# --- Model Loading and Main Execution ---
def get_model_list(device):
    model_files = glob.glob('models/reversi_model_iter_*.pt')
    if not model_files: return []
    models = []
    for f in model_files:
        try:
            iteration = int(f.split('_')[-1].split('.')[0])
            model = PolicyValueNet(num_residual_blocks=NUM_RESIDUAL_BLOCKS, num_filters=NUM_FILTERS).to(device)
            model.load_state_dict(torch.load(f, map_location=device)); model.eval()
            models.append({'path': f, 'iter': iteration, 'model': model, 'device': device})
        except (ValueError, IndexError): continue
    return sorted(models, key=lambda x: x['iter'])

def main(args):
    if torch.backends.mps.is_available(): device = torch.device("mps")
    elif torch.cuda.is_available(): device = torch.device("cuda")
    else: device = torch.device("cpu")
    print(f"Using device: {device}")
    
    elo_calculator = EloCalculator(k_factor=args.elo_k_factor)
    all_models = get_model_list(device)
    if not all_models:
        print("No models found in 'models/'. Please run train.py first."); return

    latest_model_info = all_models[-1]
    print(f"\n--- Found {len(all_models)} models. Latest is Iteration {latest_model_info['iter']} ---\n")

    play_configs = [
        {'play_style': 'best_policy', 'mcts_sims': None},
        {'play_style': 'random_policy', 'mcts_sims': None}
    ]
    for sims in [25, 50, 100]:
        play_configs.append({'play_style': 'mcts_best', 'mcts_sims': sims})
        play_configs.append({'play_style': 'mcts_random', 'mcts_sims': sims})
    
    print("*" * 80); print("PART 1: SELF-PLAY TOURNAMENT (LATEST MODEL, MIXED STYLES)"); print("*" * 80)

    # *** THIS IS THE CORRECTED LINE ***
    # Use combinations() to get pairs of *distinct* styles.
    for config1, config2 in itertools.combinations(play_configs, 2):
        agent1 = Agent(latest_model_info['model'], device, **config1)
        agent1_base_name = agent1.get_name(); agent1.get_name = lambda: f"Iter {latest_model_info['iter']} ({agent1_base_name})"
        agent2 = Agent(latest_model_info['model'], device, **config2)
        agent2_base_name = agent2.get_name(); agent2.get_name = lambda: f"Iter {latest_model_info['iter']} ({agent2_base_name})"
        run_matchup(agent1, agent2, args.num_games, elo_calculator)
    
    historical_models = all_models[:-1]
    if historical_models:
        print("\n" + "*" * 80); print("PART 2: GAUNTLET (LATEST MODEL VS. HISTORICAL MODELS)"); print("*" * 80)
        n_hist = len(historical_models)
        indices_to_test = {0, round(0.25 * (n_hist-1)) if n_hist>1 else 0, round(0.50 * (n_hist-1)) if n_hist>1 else 0, round(0.75 * (n_hist-1)) if n_hist>1 else 0}
        opponents_to_test = [historical_models[i] for i in sorted(list(indices_to_test))]
        print("Selected historical opponents:"); [print(f"  - Iteration {opp['iter']}") for opp in opponents_to_test]; print("")
        for opponent_info in opponents_to_test:
            print("\n" + "="*20 + f" GAUNTLET ROUND: LATEST vs ITERATION {opponent_info['iter']} " + "="*20 + "\n")
            for config_latest in play_configs:
                for config_opponent in play_configs:
                    agent_latest = Agent(latest_model_info['model'], device, **config_latest)
                    latest_base_name = agent_latest.get_name(); agent_latest.get_name = lambda: f"Iter {latest_model_info['iter']} ({latest_base_name})"
                    agent_opponent = Agent(opponent_info['model'], device, **config_opponent)
                    opponent_base_name = agent_opponent.get_name(); agent_opponent.get_name = lambda: f"Iter {opponent_info['iter']} ({opponent_base_name})"
                    run_matchup(agent_latest, agent_opponent, args.num_games, elo_calculator)

    elo_calculator.calculate_elos(iterations=args.elo_iterations)
    elo_calculator.display_ratings()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Reversi models and calculate ELO ratings.", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--num_games', type=int, default=10, help="Games per matchup.")
    parser.add_argument('--elo_k_factor', type=int, default=32, help="K-factor for ELO calculation.")
    parser.add_argument('--elo_iterations', type=int, default=100, help="Passes over game history for ELO calculation.")
    args = parser.parse_args()
    main(args)