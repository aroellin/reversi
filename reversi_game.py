import numpy as np

class Reversi:
    def __init__(self):
        self.new_game(start_type='standard')

    def new_game(self, start_type='standard'):
        self.board = np.zeros((8, 8), dtype=int)
        
        if start_type == 'random':
            start_type = np.random.choice(['standard', 'adjacent'])

        color1 = np.random.choice([-1, 1])
        color2 = -color1

        if start_type == 'adjacent':
            self.board[3, 3] = self.board[3, 4] = color1
            self.board[4, 3] = self.board[4, 4] = color2
        else: # standard
            self.board[3, 3] = self.board[4, 4] = color1
            self.board[3, 4] = self.board[4, 3] = color2

        self.current_player = -1

    def print_board(self):
        """
        CORRECTED: This method has been added back.
        Prints a human-readable representation of the board to the console.
        """
        print("\n  0 1 2 3 4 5 6 7")
        print(" +-----------------+")
        for r in range(8):
            row_str = f"{r}| "
            for c in range(8):
                if self.board[r, c] == -1:
                    row_str += "B " # Black
                elif self.board[r, c] == 1:
                    row_str += "W " # White
                else:
                    row_str += ". " # Empty
            row_str += "|"
            print(row_str)
        print(" +-----------------+")


    def get_legal_moves(self, player):
        legal_moves = []
        for r in range(8):
            for c in range(8):
                if self.board[r, c] == 0 and self._is_valid_move(r, c, player):
                    legal_moves.append((r, c))
        return legal_moves

    def _is_valid_move(self, r, c, player):
        for dr in [-1, 0, 1]:
            for dc in [-1, 0, 1]:
                if dr == 0 and dc == 0:
                    continue
                if self._flips_in_direction(r, c, dr, dc, player):
                    return True
        return False

    def _flips_in_direction(self, r, c, dr, dc, player):
        line = []
        r_curr, c_curr = r + dr, c + dc
        while 0 <= r_curr < 8 and 0 <= c_curr < 8:
            if self.board[r_curr, c_curr] == -player:
                line.append((r_curr, c_curr))
            elif self.board[r_curr, c_curr] == player:
                return len(line) > 0
            else:
                break
            r_curr += dr
            c_curr += dc
        return False

    def make_move(self, r, c, player):
        if not self._is_valid_move(r, c, player):
            return False
        self.board[r, c] = player
        for dr in [-1, 0, 1]:
            for dc in [-1, 0, 1]:
                if dr == 0 and dc == 0:
                    continue
                if self._flips_in_direction(r, c, dr, dc, player):
                    r_curr, c_curr = r + dr, c + dc
                    while self.board[r_curr, c_curr] == -player:
                        self.board[r_curr, c_curr] = player
                        r_curr += dr
                        c_curr += dc
        self.current_player *= -1
        return True

    def get_winner(self):
        score = np.sum(self.board)
        if score > 0:
            return 1
        elif score < 0:
            return -1
        else:
            return 0

    def is_game_over(self):
        return len(self.get_legal_moves(1)) == 0 and len(self.get_legal_moves(-1)) == 0

    def get_input_planes(self, player):
        own_pieces = np.where(self.board == player, 1, 0)
        opponent_pieces = np.where(self.board == -player, 1, 0)
        empty_slots = np.where(self.board == 0, 1, 0)
        all_ones = np.ones((8, 8), dtype=int)
        all_zeros = np.zeros((8, 8), dtype=int)
        legal_moves_mask = np.zeros((8, 8), dtype=int)
        for r, c in self.get_legal_moves(player):
            legal_moves_mask[r, c] = 1
        return np.stack([
            own_pieces,
            opponent_pieces,
            empty_slots,
            all_ones,
            all_zeros,
            legal_moves_mask
        ])