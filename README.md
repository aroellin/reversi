# Reversi AI with Self-Play and MCTS

Play here my own trained model [here](https://aroellin.github.io/apps/reversi/)!

> **Note**: This entire project, from the core game logic to the training pipeline and web interface, was created through an iterative process of "vibe coding" with Google's Gemini Pro 2.5. The development involved describing high-level goals, generating code, and progressively refining it based on the model's outputs.

This project implements a sophisticated AI for the game of Reversi (Othello), built on principles from DeepMind's AlphaZero. The AI learns the game's strategy from scratch through a process of self-play, using a deep neural network and Monte Carlo Tree Search (MCTS) to continuously improve its performance.

The project provides a complete ecosystem for training, analyzing, and playing with the AI:
*   **Training Pipeline (`train.py`):** An AlphaZero-style loop that generates game data and trains the neural network.
*   **Pygame-based GUI (`play.py`):** A feature-rich desktop application to play against the AI, visualize its thinking process with real-time hints, and analyze its move probabilities.
*   **Web Interface (`index.html`, `reversi.js`):** A fully client-side web application that runs the AI in the browser using the ONNX runtime, making it easily shareable and accessible.
*   **Evaluation Suite (`evaluate.py`):** A powerful tool to benchmark different models and strategies against each other in a tournament format, calculating ELO ratings to track progress.
*   **Model Exporter (`pt2onnx.py`):** A utility to convert trained PyTorch models into the ONNX format for deployment on the web.

## Installation

1.  **Clone the repository:**
    ```bash
    git clone <repository-url>
    cd <repository-directory>
    ```

2.  **Set Up a Virtual Environment (Recommended)**
    Using a virtual environment is highly recommended to isolate project dependencies and avoid conflicts with other Python projects. Choose one of the following options.

    **Option A: Using `venv` (Python's built-in tool)**
    This is the standard tool available in Python 3.

    ```bash
    # Create the virtual environment in a folder named 'venv'
    python -m venv venv

    # Activate the environment
    # On macOS and Linux:
    source venv/bin/activate
    # On Windows (Command Prompt):
    venv\Scripts\activate
    ```

    **Option B: Using `conda` (for Anaconda/Miniconda users)**
    If you use Anaconda or Miniconda, you can create an environment with `conda`.

    ```bash
    # Create a new environment named 'reversi_ai' with Python 3
    conda create -n reversi_ai python=3

    # Activate the environment
    conda activate reversi_ai
    ```

3.  **Install the required Python packages:**
    Run the following command in your terminal to install all the necessary libraries from the `requirements.txt` file.
    ```bash
    pip install -r requirements.txt
    ```

## Usage

### 1. Train a New Model

To start the training process from scratch or resume from the latest checkpoint, run:

```bash
python train.py
```
*   A `models/` directory will be created to store model checkpoints (`.pt` files).
*   The script initiates an AlphaZero-style loop:
    1.  **Self-Play Data Generation**: The current best model plays games against itself using MCTS. To ensure robustness, it also plays games against a pool of its own past versions.
    2.  **Network Training**: The network is trained on the generated data (board states, MCTS move probabilities, and game outcomes).
    3.  **Checkpointing**: The newly trained model is saved as `models/reversi_model_iter_X.pt`.

I suggest to start training with only a few MCTS steps at the beginning (`MCTS_SIMULATIONS_TRAIN=10` in `train.py`). This speeds up the initial training phase, so that you can play against some moderately well playing game quickly. After iteration 30, you can switch to `MCTS_SIMULATIONS_TRAIN=100`.

### 2. Evaluate Model Performance

To benchmark models against each other and track their strength, use the evaluation script:

```bash
python evaluate.py --num_games 20
```
*   The script runs a comprehensive tournament to assess model performance.
*   **Part 1 (Self-Play Tournament):** The latest model plays against itself using various strategies (e.g., MCTS with 50 sims vs. MCTS with 100 sims, Best Policy vs. MCTS).
*   **Part 2 (Gauntlet):** The latest model is pitted against a selection of historical models from different training stages.
*   After all games are played, it calculates and displays **ELO ratings** for every agent, providing a clear leaderboard of their relative strengths.
*   You can control the number of games per matchup with the `--num_games` argument.

### 3. Play Against the AI (Python GUI)

The Pygame interface is a powerful tool for playing against and analyzing the AI.

```bash
python play.py
```
This launches a window with the Reversi board and a detailed control panel where you can:
*   **Choose Player Color**: Play as Black (first player) or White.
*   **Select AI Play Style**:
    *   `Simple Policy`: The AI moves instantly based on raw policy output (fast but weak).
    *   `MCTS Search`: The AI uses Monte Carlo Tree Search (stronger but slower).
*   **Enable Hints**: Visualize the AI's preferences using either raw policy probabilities or the refined MCTS search distribution.
*   **Undo Move**: Go back one or two turns (to undo your move and the AI's response).

### 4. Play Against the AI (Web Interface)

You can play against the same AI in any modern web browser.

**Step 1: Export the Model to ONNX**
Convert your latest PyTorch model into the web-compatible ONNX format.

```bash
python pt2onnx.py
```
This finds the latest model in `models/` and creates a `reversi_model.onnx` file.

**Step 2: Start a Local Web Server**
Due to browser security policies, you must serve the project files from a local server.

```bash
# For Python 3
python -m http.server
```

**Step 3: Play in the Browser**
Open your web browser and navigate to:

**http://localhost:8000**

The web interface will load, download the `.onnx` model, and run the AI entirely in your browser using `onnxruntime-web`.

## AI Architecture: Policy-Value Network

The brain of the AI is a single, deep `PolicyValueNet`. The architectural constants are defined as `NUM_FILTERS = 64` and `NUM_RESIDUAL_BLOCKS = 4`. The total number of learnable parameters is approximately **1.37 million**.

*Parameters for a `Conv2d` layer are calculated as `(kernel_h * kernel_w * in_channels) * out_channels + out_channels_bias`. A `Linear` layer is `in_features * out_features + out_features_bias`. A `BatchNorm2d` layer has `2 * num_features` learnable parameters (gamma and beta).*

---

**1. Input Layer**
*   **Shape:** `[B, 6, 8, 8]`
*   **Description:** 6 planes representing the board state (current player, opponent, empty, etc.).

---

**2. Convolutional Stem**
*   **Output Shape:** `[B, 64, 8, 8]`
*   **Layers:**
    *   `Conv2d(in=6, out=64, kernel=3, padding=1)`
    *   `BatchNorm2d(64)` & `ReLU`
*   **Parameters:**
    *   `Conv2d`: `(3*3*6)*64 + 64 = 3,520`
    *   `BatchNorm2d`: `2 * 64 = 128`
    *   **Total for Stem: 3,648**

---

**3. Residual Tower (4 identical blocks)**
*   **Output Shape:** `[B, 64, 8, 8]` (maintained through all blocks)
*   **Layers per Block:**
    1.  `Conv2d(64, 64, kernel=3, padding=1)` + `BatchNorm2d(64)` + `ReLU`
    2.  `Conv2d(64, 64, kernel=3, padding=1)` + `BatchNorm2d(64)`
    3.  Skip Connection + `ReLU`
*   **Parameters (per block):**
    *   `Conv2d_1`: `(3*3*64)*64 + 64 = 36,928`
    *   `BatchNorm2d_1`: `2 * 64 = 128`
    *   `Conv2d_2`: `(3*3*64)*64 + 64 = 36,928`
    *   `BatchNorm2d_2`: `2 * 64 = 128`
    *   **Total per block: 74,112**
*   **Total for Tower:** `74,112 * 4 blocks = 296,448`

---

**4. Policy Head**
*   **Output Shape:** `[B, 64]`
*   **Layers:**
    1.  `Conv2d(64, 64, kernel=1)` & `BatchNorm2d(64)` & `ReLU`
    2.  `Flatten()` -> `[B, 64*8*8]` = `[B, 4096]`
    3.  `Linear(in=4096, out=128)` & `ReLU`
    4.  `Linear(in=128, out=64)`
*   **Parameters:**
    *   `Conv2d`: `(1*1*64)*64 + 64 = 4,160`
    *   `BatchNorm2d`: `2 * 64 = 128`
    *   `Linear_1`: `4096 * 128 + 128 = 524,416`
    *   `Linear_2`: `128 * 64 + 64 = 8,256`
    *   **Total for Policy Head: 536,960**

---

**5. Value Head**
*   **Output Shape:** `[B, 1]`
*   **Layers:**
    1.  `Conv2d(64, 64, kernel=1)` & `BatchNorm2d(64)` & `ReLU`
    2.  `Flatten()` -> `[B, 4096]`
    3.  `Linear(in=4096, out=128)` & `ReLU`
    4.  `Linear(in=128, out=1)` & `Tanh`
*   **Parameters:**
    *   `Conv2d`: `(1*1*64)*64 + 64 = 4,160`
    *   `BatchNorm2d`: `2 * 64 = 128`
    *   `Linear_1`: `4096 * 128 + 128 = 524,416`
    *   `Linear_2`: `128 * 1 + 1 = 129`
    *   **Total for Value Head: 528,833**

---

**Total Network Parameters**
*   **Stem:** `3,648`
*   **Residual Tower:** `296,448`
*   **Policy Head:** `536,960`
*   **Value Head:** `528,833`
*   **GRAND TOTAL: 1,365,889**
