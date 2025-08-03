# FILE: play.py
import torch
import torch.nn.functional as F
import numpy as np
import pygame
import sys
import glob
import os
import math
import copy

# Import from your existing project files
from reversi_game import Reversi
from train import PolicyValueNet, NUM_FILTERS, NUM_RESIDUAL_BLOCKS

# --- Pygame & UI Setup ---
pygame.init()
SQUARE_SIZE = 75
BOARD_SIZE = 8
BOARD_WIDTH = BOARD_SIZE * SQUARE_SIZE
CONTROL_PANEL_WIDTH = 350
WINDOW_WIDTH = BOARD_WIDTH + CONTROL_PANEL_WIDTH
WINDOW_HEIGHT = BOARD_WIDTH
RADIUS = int(SQUARE_SIZE / 2 - 5)

# Colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
GREEN = (0, 128, 0)
HINT_COLOR = (200, 200, 220, 150)
LEGAL_MOVE_COLOR = (50, 50, 50)
UI_BG_COLOR = (40, 40, 40)
UI_BORDER_COLOR = (80, 80, 80)
UI_FONT_COLOR = (220, 220, 220)
BUTTON_COLOR = (70, 70, 70)
BUTTON_HOVER_COLOR = (100, 100, 100)

# Fonts
TITLE_FONT = pygame.font.SysFont("Arial", 18, bold=True)
LABEL_FONT = pygame.font.SysFont("Arial", 14, bold=True)
OPTION_FONT = pygame.font.SysFont("Arial", 13)
STATUS_FONT = pygame.font.SysFont("Arial", 16)

# --- Device Setup ---
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
print(f"Using device: {device}")


# --- Simple Pygame GUI Classes ---
class Dropdown:
    def __init__(self, x, y, w, h, font, options, default_key):
        self.rect = pygame.Rect(x, y, w, h)
        self.options = options
        self.selected_key = default_key
        self.font = font
        self.is_open = False

    def handle_event(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN:
            if self.rect.collidepoint(event.pos):
                self.is_open = not self.is_open
                return True
            elif self.is_open:
                for i, key in enumerate(self.options.keys()):
                    option_rect = pygame.Rect(self.rect.left, self.rect.bottom + i * self.rect.height, self.rect.width, self.rect.height)
                    if option_rect.collidepoint(event.pos):
                        self.selected_key = key
                        self.is_open = False
                        return True
                self.is_open = False
        return False

    def draw(self, screen):
        pygame.draw.rect(screen, BUTTON_COLOR, self.rect)
        pygame.draw.rect(screen, UI_BORDER_COLOR, self.rect, 2)
        text_surf = self.font.render(self.options[self.selected_key], True, UI_FONT_COLOR)
        screen.blit(text_surf, text_surf.get_rect(center=self.rect.center))

    def draw_options(self, screen):
        if self.is_open:
            for i, key in enumerate(self.options.keys()):
                option_rect = pygame.Rect(self.rect.left, self.rect.bottom + i * self.rect.height, self.rect.width, self.rect.height)
                pygame.draw.rect(screen, BUTTON_COLOR, option_rect)
                pygame.draw.rect(screen, UI_BORDER_COLOR, option_rect, 2)
                text_surf = self.font.render(self.options[key], True, UI_FONT_COLOR)
                screen.blit(text_surf, text_surf.get_rect(center=option_rect.center))
    
    def get_value(self):
        return self.selected_key

class Button:
    def __init__(self, x, y, w, h, font, text, callback):
        self.rect = pygame.Rect(x, y, w, h)
        self.font = font
        self.text = text
        self.callback = callback

    def handle_event(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN:
            if self.rect.collidepoint(event.pos):
                if self.callback:
                    self.callback()

    def draw(self, screen):
        color = BUTTON_HOVER_COLOR if self.rect.collidepoint(pygame.mouse.get_pos()) else BUTTON_COLOR
        pygame.draw.rect(screen, color, self.rect)
        pygame.draw.rect(screen, UI_BORDER_COLOR, self.rect, 2)
        text_surf = self.font.render(self.text, True, UI_FONT_COLOR)
        screen.blit(text_surf, text_surf.get_rect(center=self.rect.center))

# --- MCTS Classes (No changes) ---
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
    def __init__(self, model, c_puct=1.0):
        self.model = model; self.c_puct = c_puct; self.root = MCTSNode()
    def _get_priors_and_value(self, game_state):
        state_np = game_state.get_input_planes(game_state.current_player); state_tensor = torch.from_numpy(state_np).float().unsqueeze(0).to(device)
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
    def get_policy_distribution(self):
        if not self.root.children: return {}
        total_visits = sum(child.n_visits for child in self.root.children.values())
        if total_visits == 0: return {}
        return {move: node.n_visits / total_visits for move, node in self.root.children.items()}

# --- AI and Model Helper Functions ---
def get_latest_model_path():
    list_of_files = glob.glob('models/reversi_model_iter_*.pt')
    if not list_of_files: return None
    return max(list_of_files, key=lambda f: int(f.split('_')[-1].split('.')[0]))

def load_model():
    model = PolicyValueNet(num_residual_blocks=NUM_RESIDUAL_BLOCKS, num_filters=NUM_FILTERS).to(device)
    model_path = get_latest_model_path()
    if model_path:
        print(f"Loading model from {model_path}"); model.load_state_dict(torch.load(model_path, map_location=device)); model.eval(); return model
    else:
        print("No trained model found. Please run train.py first."); sys.exit()

def get_policy_move(model, game):
    legal_moves = game.get_legal_moves(game.current_player)
    if not legal_moves: return None
    state_np = game.get_input_planes(game.current_player); state_tensor = torch.from_numpy(state_np).float().unsqueeze(0).to(device)
    with torch.no_grad(): logits, _ = model(state_tensor)
    legal_move_indices = [r * 8 + c for r, c in legal_moves]
    legal_logits = logits[0, legal_move_indices]
    best_move_index = torch.argmax(legal_logits).item()
    return legal_moves[best_move_index]

# --- Drawing Functions ---
def draw_board_and_pieces(screen, game, human_player, show_legal_moves, hints):
    board_area = pygame.Rect(0, 0, BOARD_WIDTH, BOARD_WIDTH)
    screen.fill(GREEN, board_area)
    for x in range(BOARD_SIZE + 1):
        pygame.draw.line(screen, BLACK, (x * SQUARE_SIZE, 0), (x * SQUARE_SIZE, BOARD_WIDTH), 2)
        pygame.draw.line(screen, BLACK, (0, x * SQUARE_SIZE), (BOARD_WIDTH, x * SQUARE_SIZE), 2)
    
    # Draw professional board markings
    dot_radius = int(SQUARE_SIZE / 20)
    dot_locations = [(2, 2), (2, 6), (6, 2), (6, 6)]
    for r_idx, c_idx in dot_locations:
        pygame.draw.circle(screen, BLACK, (c_idx * SQUARE_SIZE, r_idx * SQUARE_SIZE), dot_radius)

    if hints and game.current_player == human_player:
        max_hint = max(hints.values()) if hints else 0
        if max_hint > 0:
            for move, prob in hints.items():
                r, c = move
                radius = int(RADIUS * math.sqrt(prob / max_hint))
                if radius > 1:
                    hint_surface = pygame.Surface((radius * 2, radius * 2), pygame.SRCALPHA)
                    pygame.draw.circle(hint_surface, HINT_COLOR, (radius, radius), radius)
                    screen.blit(hint_surface, (c * SQUARE_SIZE + SQUARE_SIZE/2 - radius, r * SQUARE_SIZE + SQUARE_SIZE/2 - radius))
    
    legal_moves = game.get_legal_moves(human_player) if show_legal_moves else []
    for r in range(BOARD_SIZE):
        for c in range(BOARD_SIZE):
            if game.board[r, c] == 1:
                pygame.draw.circle(screen, WHITE, (int(c*SQUARE_SIZE + SQUARE_SIZE/2), int(r*SQUARE_SIZE + SQUARE_SIZE/2)), RADIUS)
            elif game.board[r, c] == -1:
                pygame.draw.circle(screen, BLACK, (int(c*SQUARE_SIZE + SQUARE_SIZE/2), int(r*SQUARE_SIZE + SQUARE_SIZE/2)), RADIUS)
            elif (r, c) in legal_moves and game.current_player == human_player:
                pygame.draw.circle(screen, LEGAL_MOVE_COLOR, (int(c*SQUARE_SIZE + SQUARE_SIZE/2), int(r*SQUARE_SIZE + SQUARE_SIZE/2)), RADIUS / 4)

def draw_control_panel(screen, ui_elements_with_labels, status_text, scores, edit_mode):
    panel_rect = pygame.Rect(BOARD_WIDTH, 0, CONTROL_PANEL_WIDTH, WINDOW_HEIGHT)
    pygame.draw.rect(screen, UI_BG_COLOR, panel_rect)

    title_surf = TITLE_FONT.render("Reversi AI Analysis Tool", True, WHITE)
    screen.blit(title_surf, (BOARD_WIDTH + (CONTROL_PANEL_WIDTH - title_surf.get_width()) / 2, 20))

    status_surf = STATUS_FONT.render(status_text, True, UI_FONT_COLOR)
    screen.blit(status_surf, status_surf.get_rect(center=(BOARD_WIDTH + CONTROL_PANEL_WIDTH / 2, 55)))
    
    score_b_surf = OPTION_FONT.render(f"Black (B): {scores['black']}", True, UI_FONT_COLOR)
    score_w_surf = OPTION_FONT.render(f"White (W): {scores['white']}", True, UI_FONT_COLOR)
    screen.blit(score_b_surf, (BOARD_WIDTH + 20, 80))
    screen.blit(score_w_surf, (BOARD_WIDTH + CONTROL_PANEL_WIDTH - score_w_surf.get_width() - 20, 80))
    
    y_offset = 120
    for label_text, element_or_group, is_edit_button in ui_elements_with_labels:
        # Determine if the element should be drawn
        should_draw = True
        if is_edit_button == 'edit_only' and not edit_mode: should_draw = False
        if is_edit_button == 'play_only' and edit_mode: should_draw = False
        if not should_draw: continue

        if label_text:
            label_surf = LABEL_FONT.render(label_text, True, UI_FONT_COLOR)
            screen.blit(label_surf, (BOARD_WIDTH + 20, y_offset))
            y_offset += 25
        
        if isinstance(element_or_group, tuple):
            group_elements = element_or_group
            for element in group_elements: element.rect.top = y_offset
            group_elements[0].draw(screen); group_elements[1].draw(screen)
            y_offset += group_elements[0].rect.height + 15
        else:
            element = element_or_group
            element.rect.top = y_offset; element.draw(screen)
            y_offset += element.rect.height + 15

# --- Main Game Loop ---
def main():
    screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
    pygame.display.set_caption("Reversi AI Analysis Tool")
    
    game = Reversi()
    board_history = []
    player_history = []
    human_player = -1
    game_over = False
    edit_mode = False
    status_text = "Loading Model..."
    hints = {}
    
    def new_game():
        nonlocal game, board_history, player_history, game_over, human_player, hints, edit_mode
        game.new_game()
        board_history = [np.copy(game.board)]
        player_history = [game.current_player]
        game_over = False
        edit_mode = False
        hints = {}
        pygame.display.set_caption(f"Reversi - {'You (B)' if human_player == -1 else 'AI (B)'} vs {'AI (W)' if human_player == 1 else 'You (W)'}")

    def undo_move():
        nonlocal game, board_history, player_history, game_over, hints
        if len(board_history) > 1 and not edit_mode:
            board_history.pop(); player_history.pop()
            if player_history and player_history[-1] == -human_player and len(board_history) > 1:
                 board_history.pop(); player_history.pop()
            game.board = np.copy(board_history[-1]); game.current_player = player_history[-1]
            game_over = False; hints = {}
            
    def handle_new_game_click():
        nonlocal edit_mode
        if edit_mode:
            game.new_game() # Reset board but stay in edit mode
        else:
            new_game() # Full reset

    def change_player_color(new_color):
        nonlocal human_player, hints
        human_player = new_color
        hints = {} # Reset hints on any color change

    def enter_edit_mode():
        nonlocal edit_mode, hints
        edit_mode = True
        hints = {}
    
    def exit_edit_mode(next_player):
        nonlocal edit_mode, game_over, board_history, player_history, hints
        edit_mode = False
        game.current_player = next_player
        board_history = [np.copy(game.board)]
        player_history = [game.current_player]
        game_over = False
        hints = {}

    # --- GUI Element Initialization ---
    dd_height, btn_height = 30, 40
    player_color_dd = Dropdown(BOARD_WIDTH + 20, 0, CONTROL_PANEL_WIDTH - 40, dd_height, OPTION_FONT, {-1: "Play as Black", 1: "Play as White"}, -1)
    ai_mode_dd = Dropdown(BOARD_WIDTH + 20, 0, CONTROL_PANEL_WIDTH - 40, dd_height, OPTION_FONT, {"mcts": "MCTS Search", "policy": "Simple Policy"}, "mcts")
    hint_mode_dd = Dropdown(BOARD_WIDTH + 20, 0, CONTROL_PANEL_WIDTH - 40, dd_height, OPTION_FONT, {"none": "No Hints", "mcts": "MCTS Hints", "policy": "Policy Hints"}, "none")
    sims_dd = Dropdown(BOARD_WIDTH + 20, 0, CONTROL_PANEL_WIDTH - 40, dd_height, OPTION_FONT, {50: "50 Sims", 100: "100 Sims", 200: "200 Sims", 400: "400 Sims"}, 200)
    legal_moves_dd = Dropdown(BOARD_WIDTH + 20, 0, CONTROL_PANEL_WIDTH - 40, dd_height, OPTION_FONT, {True: "Show Legal Moves", False: "Hide Legal Moves"}, True)
    
    new_game_btn = Button(BOARD_WIDTH + 20, 0, (CONTROL_PANEL_WIDTH - 50) / 2, btn_height, OPTION_FONT, "New Game", handle_new_game_click)
    undo_btn = Button(BOARD_WIDTH + 30 + (CONTROL_PANEL_WIDTH - 50) / 2, 0, (CONTROL_PANEL_WIDTH - 50) / 2, btn_height, OPTION_FONT, "Undo", undo_move)
    edit_btn = Button(BOARD_WIDTH + 20, 0, CONTROL_PANEL_WIDTH - 40, btn_height, OPTION_FONT, "Edit Board", enter_edit_mode)
    resume_white_btn = Button(BOARD_WIDTH + 20, 0, (CONTROL_PANEL_WIDTH - 50) / 2, btn_height, OPTION_FONT, "White Next", lambda: exit_edit_mode(1))
    resume_black_btn = Button(BOARD_WIDTH + 30 + (CONTROL_PANEL_WIDTH - 50) / 2, 0, (CONTROL_PANEL_WIDTH - 50) / 2, btn_height, OPTION_FONT, "Black Next", lambda: exit_edit_mode(-1))
    
    # Label, Element, Edit/Play Mode flag
    ui_elements_with_labels = [
        ("Play As:", player_color_dd, 'all'), 
        ("AI Play Style:", ai_mode_dd, 'all'), 
        ("Show Player Hints Using:", hint_mode_dd, 'all'), 
        ("MCTS Simulations:", sims_dd, 'all'), 
        ("Legal Moves:", legal_moves_dd, 'all'), 
        (None, (new_game_btn, undo_btn), 'all'),
        (None, edit_btn, 'play_only'),
        (None, (resume_white_btn, resume_black_btn), 'edit_only')
    ]
    ui_elements = [el for _, group, _ in ui_elements_with_labels for el in (group if isinstance(group, tuple) else (group,))]

    model = load_model()
    mcts = MCTS(model, c_puct=1.0)
    new_game() # Perform initial full setup
    
    running = True
    while running:
        mouse_pos = pygame.mouse.get_pos()
        # --- Handle Events ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT: running = False

            # Handle dropdowns and non-destructive color change
            old_color = player_color_dd.get_value()
            dropdown_changed = any(el.handle_event(event) for el in ui_elements if isinstance(el, Dropdown))
            if dropdown_changed:
                new_color = player_color_dd.get_value()
                if old_color != new_color:
                    change_player_color(new_color)
                continue # Skip other processing for this event

            # Handle buttons
            if not edit_mode:
                new_game_btn.handle_event(event)
                undo_btn.handle_event(event)
                edit_btn.handle_event(event)
            else: # In edit mode
                new_game_btn.handle_event(event) # New game button has special logic
                resume_white_btn.handle_event(event)
                resume_black_btn.handle_event(event)

            # Handle board clicks (Human Move or Board Edit)
            if event.type == pygame.MOUSEBUTTONDOWN and mouse_pos[0] < BOARD_WIDTH:
                c, r = mouse_pos[0] // SQUARE_SIZE, mouse_pos[1] // SQUARE_SIZE
                if edit_mode:
                    # Cycle piece: 0->1, 1->-1, -1->0
                    current_piece = game.board[r, c]
                    if current_piece == 0: game.board[r, c] = 1
                    elif current_piece == 1: game.board[r, c] = -1
                    else: game.board[r, c] = 0
                elif not game_over and game.current_player == human_player:
                    if (r, c) in game.get_legal_moves(human_player):
                        game.make_move(r, c, human_player); board_history.append(np.copy(game.board)); player_history.append(game.current_player); hints = {}
                        status_text = "AI is thinking..."
                        scores = {"black": np.sum(game.board == -1), "white": np.sum(game.board == 1)}
                        draw_board_and_pieces(screen, game, human_player, legal_moves_dd.get_value(), hints)
                        draw_control_panel(screen, ui_elements_with_labels, status_text, scores, edit_mode)
                        for el in ui_elements:
                            if isinstance(el, Dropdown): el.draw_options(screen)
                        pygame.display.flip()
                        pygame.time.wait(200)

        # --- Game Logic (only runs if not in edit mode) ---
        if not edit_mode:
            if not game_over:
                if not game.get_legal_moves(game.current_player):
                    if game.get_legal_moves(-game.current_player):
                        game.current_player *= -1; player_history.append(game.current_player); board_history.append(np.copy(game.board))
                    else: game_over = True
                
                if not game_over and game.current_player == -human_player:
                    ai_mode, n_sims = ai_mode_dd.get_value(), sims_dd.get_value()
                    ai_move = get_policy_move(model, game) if ai_mode == 'policy' else (mcts.search(game, n_sims) or mcts.get_best_move())
                    
                    if ai_move:
                        game.make_move(ai_move[0], ai_move[1], -human_player); board_history.append(np.copy(game.board)); player_history.append(game.current_player)
                
                elif not game_over and game.current_player == human_player:
                    hint_mode = hint_mode_dd.get_value()
                    if hint_mode != 'none' and not hints:
                        status_text = f"Calculating hints..."
                        draw_control_panel(screen, ui_elements_with_labels, status_text, {"black": np.sum(game.board == -1), "white": np.sum(game.board == 1)}, edit_mode)
                        pygame.display.update(pygame.Rect(BOARD_WIDTH, 0, CONTROL_PANEL_WIDTH, WINDOW_HEIGHT))

                        if hint_mode == 'policy': hints, _ = mcts._get_priors_and_value(game)
                        elif hint_mode == 'mcts':
                            n_sims = sims_dd.get_value(); mcts.search(game, n_sims); hints = mcts.get_policy_distribution()
        
        # --- Update Status Text ---
        if edit_mode:
            status_text = "Board Edit Mode"
        elif game_over:
            winner = game.get_winner()
            status_text = "You Won!" if winner == human_player else "AI Won!" if winner == -human_player else "It's a Draw!"
        elif game.current_player == human_player:
            status_text = "Your Turn"
        else:
            status_text = "AI is thinking..."
        
        # --- Final Draw for the frame ---
        scores = {"black": np.sum(game.board == -1), "white": np.sum(game.board == 1)}
        draw_board_and_pieces(screen, game, human_player, legal_moves_dd.get_value(), hints)
        draw_control_panel(screen, ui_elements_with_labels, status_text, scores, edit_mode)
        for el in ui_elements:
            if isinstance(el, Dropdown): el.draw_options(screen)

        pygame.display.flip()

    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main()