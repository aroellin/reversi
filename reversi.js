// --- Global DOM elements and Constants ---
const canvas = document.getElementById('board');
const ctx = canvas.getContext('2d');
const scoresEl = document.getElementById('scores');
const statusEl = document.getElementById('status');
const restartBtn = document.getElementById('restart-btn');
const undoBtn = document.getElementById('undo-btn');
const editBtn = document.getElementById('edit-btn');
const resumeWhiteBtn = document.getElementById('resume-white-btn');
const resumeBlackBtn = document.getElementById('resume-black-btn');

const BOARD_SIZE = 8;
const EMPTY = 0;

// --- Game State ---
let aiModel = null;
let boardHistory = [];
let playerHistory = [];
let gameOver = false;
let humanPlayer = -1;
let editMode = false;
let aiThinking = false;

// --- Options ---
let options = {
    showLegalMoves: true,
    aiMode: 'mcts',
    mctsSims: 400,
    hintMode: 'none'
};

// --- MCTS Classes (self-contained) ---
class MCTSNode {
    constructor(parent = null, prior_p = 1.0) { this.parent = parent; this.children = {}; this.n_visits = 0; this.q_value = 0; this.u_value = 0; this.p_value = prior_p; }
    expand(action_priors) { for (const [actionStr, prior] of Object.entries(action_priors)) { if (!this.children[actionStr]) { this.children[actionStr] = new MCTSNode(this, prior); } } }
    select(c_puct) { return Object.entries(this.children).reduce((best, [actionStr, node]) => { const value = node.get_value(c_puct); return value > best.value ? { value: value, action: actionStr, node: node } : best; }, { value: -Infinity, action: null, node: null }); }
    update(leaf_value) { this.n_visits += 1; this.q_value += (leaf_value - this.q_value) / this.n_visits; }
    update_recursive(leaf_value) { if (this.parent) { this.parent.update_recursive(-leaf_value); } this.update(leaf_value); }
    get_value(c_puct) { if (this.parent === null) return this.q_value; this.u_value = c_puct * this.p_value * Math.sqrt(this.parent.n_visits) / (1 + this.n_visits); return this.q_value + this.u_value; }
    is_leaf() { return Object.keys(this.children).length === 0; }
}
class MCTS {
    constructor(model, c_puct = 1.0, n_simulations = 200) { this.model = model; this.c_puct = c_puct; this.n_simulations = n_simulations; }
    async _get_priors_and_value(board, player) {
        const inputPlanes = createInputPlanes(board, player); const inputTensor = new ort.Tensor('float32', inputPlanes, [1, 6, 8, 8]);
        const feeds = { 'input': inputTensor }; const results = await this.model.run(feeds);
        const logits = results.policy.data; const value = results.value.data[0];
        const legalMoves = getLegalMoves(board, player);
        if (legalMoves.length === 0) return [{}, value];
        const legalIndices = legalMoves.map(m => m.r * 8 + m.c); const legalLogits = legalIndices.map(i => logits[i]);
        const legalProbs = softmax(legalLogits); const action_priors = {};
        legalMoves.forEach((move, i) => { action_priors[JSON.stringify(move)] = legalProbs[i]; });
        return [action_priors, value];
    }
    async _run_simulation(root, board, player) {
        let node = root; let simBoard = board.map(row => row.slice()); let simPlayer = player;
        while (!node.is_leaf()) {
            const { action, node: nextNode } = node.select(this.c_puct); const move = JSON.parse(action);
            simBoard = makeMoveAndGetNewBoard(simBoard, move.r, move.c, simPlayer); simPlayer = -simPlayer; node = nextNode;
        }
        const [action_priors, leaf_value] = await this._get_priors_and_value(simBoard, simPlayer);
        if (getLegalMoves(simBoard, simPlayer).length > 0) { node.expand(action_priors); }
        node.update_recursive(-leaf_value);
    }
    async get_move_and_policy(board, player) {
        const root = new MCTSNode();
        for (let i = 0; i < this.n_simulations; i++) {
            await this._run_simulation(root, board, player);
        }
        const best = Object.entries(root.children).reduce((best, [action, node]) => {
            return node.n_visits > best.visits ? { visits: node.n_visits, action: action } : best;
        }, { visits: -1, action: null });
        return { move: best.action ? JSON.parse(best.action) : null, policy: root.children };
    }
}
function softmax(arr) { const maxLogit = Math.max(...arr); const exps = arr.map(x => Math.exp(x - maxLogit)); const sumExps = exps.reduce((a, b) => a + b); return exps.map(e => e / sumExps); }

// --- Core AI Logic ---
async function loadModel() {
    try {
        const session = await ort.InferenceSession.create('./reversi_model.onnx', { executionProviders: ['wasm'] });
        aiModel = session;
        statusEl.textContent = "AI Model Loaded. Press 'New Game' to start.";
    } catch (e) {
        statusEl.textContent = `Error loading AI model: ${e.message}. Please check console.`;
        console.error(e);
    }
}
async function getAiMove(currentBoard, player) {
    const legalMoves = getLegalMoves(currentBoard, player);
    if (legalMoves.length === 0) return null;
    let bestMove = null;
    if (options.aiMode === 'mcts') {
        const mcts = new MCTS(aiModel, 1.0, options.mctsSims);
        const { move } = await mcts.get_move_and_policy(currentBoard, player);
        bestMove = move;
    } else {
        const inputPlanes = createInputPlanes(currentBoard, player);
        const inputTensor = new ort.Tensor('float32', inputPlanes, [1, 6, 8, 8]);
        const feeds = { 'input': inputTensor };
        const results = await aiModel.run(feeds);
        const policyOutput = results.policy.data;
        let maxLogit = -Infinity;
        for (const move of legalMoves) {
            const index = move.r * 8 + move.c;
            if (policyOutput[index] > maxLogit) { maxLogit = policyOutput[index]; bestMove = move; }
        }
    }
    return bestMove;
}

// --- Game Logic ---
function createBoard() { return Array(BOARD_SIZE).fill(0).map(() => Array(BOARD_SIZE).fill(EMPTY)); }
function initializeGame() {
    const newBoard = createBoard();
    newBoard[3][3] = 1; newBoard[4][4] = 1;
    newBoard[3][4] = -1; newBoard[4][3] = -1;
    boardHistory = [newBoard];
    playerHistory = [-1];
    gameOver = false;
    editMode = false;
    aiThinking = false;

    // Ensure button visibility is reset to the default state
    editBtn.style.display = 'inline-block';
    resumeWhiteBtn.style.display = 'none';
    resumeBlackBtn.style.display = 'none';

    gameLoop();
}
function getCurrentBoard() { return boardHistory[boardHistory.length - 1]; }
function getCurrentPlayer() { return playerHistory[playerHistory.length - 1]; }
function getLegalMoves(board, player) {
    const legalMoves = [];
    for (let r = 0; r < 8; r++) { for (let c = 0; c < 8; c++) { if (board[r][c] === EMPTY && isValidMove(board, r, c, player)) { legalMoves.push({ r, c }); } } }
    return legalMoves;
}
function isValidMove(board, r, c, player) {
    for (let dr = -1; dr <= 1; dr++) { for (let dc = -1; dc <= 1; dc++) { if (dr === 0 && dc === 0) continue; if (checkLine(board, r, c, dr, dc, player).length > 0) return true; } }
    return false;
}
function checkLine(board, r, c, dr, dc, player) {
    const opponent = -player; let line = [];
    let currR = r + dr, currC = c + dc;
    while (currR >= 0 && currR < 8 && currC >= 0 && currC < 8) {
        if (board[currR][currC] === opponent) line.push({ r: currR, c: currC });
        else if (board[currR][currC] === player) return line;
        else break;
        currR += dr; currC += dc;
    }
    return [];
}
function makeMoveAndGetNewBoard(board, r, c, player) {
    const newBoard = board.map(row => row.slice());
    newBoard[r][c] = player;
    for (let dr = -1; dr <= 1; dr++) {
        for (let dc = -1; dc <= 1; dc++) {
            if (dr === 0 && dc === 0) continue;
            const toFlip = checkLine(newBoard, r, c, dr, dc, player);
            toFlip.forEach(pos => { newBoard[pos.r][pos.c] = player; });
        }
    }
    return newBoard;
}

function handleBoardEdit(r, c) {
    if (r < 0 || r >= BOARD_SIZE || c < 0 || c >= BOARD_SIZE) return;
    const board = getCurrentBoard();

    // Cycle piece: Empty -> White -> Black -> Empty
    if (board[r][c] === EMPTY) board[r][c] = 1;      // Empty -> White
    else if (board[r][c] === 1) board[r][c] = -1;    // White -> Black
    else board[r][c] = 0;                            // Black -> Empty
    
    drawBoard(null); // Redraw without hints
    updateInfoPanel(); // Update scores
}

function handleHumanMove(r, c) {
    if (gameOver || aiThinking || getCurrentPlayer() !== humanPlayer) return;
    const legalMoves = getLegalMoves(getCurrentBoard(), getCurrentPlayer());
    if (!legalMoves.some(m => m.r === r && m.c === c)) return;
    const newBoard = makeMoveAndGetNewBoard(getCurrentBoard(), r, c, getCurrentPlayer());
    boardHistory.push(newBoard);
    playerHistory.push(-getCurrentPlayer());
    gameLoop();
}
function undoMove() {
    if (boardHistory.length <= 1 || aiThinking) return;
    boardHistory.pop(); playerHistory.pop();
    if (getCurrentPlayer() !== humanPlayer && boardHistory.length > 1) {
        boardHistory.pop(); playerHistory.pop();
    }
    gameOver = false;
    gameLoop();
}

// REFACTORED: The main game loop with robust state handling
async function gameLoop() {
    if (editMode) return; // Halt game logic while in edit mode

    updateDisplay(false); // Update the basic UI (board, scores, status)
    
    const board = getCurrentBoard();
    let player = getCurrentPlayer();
    
    if (getLegalMoves(board, player).length === 0 && getLegalMoves(board, -player).length === 0) {
        gameOver = true;
        updateDisplay(false);
        return;
    }
    
    if (getLegalMoves(board, player).length === 0) {
        player = -player;
        playerHistory.push(player);
        updateDisplay(false);
        await new Promise(resolve => setTimeout(resolve, 500));
        gameLoop();
        return;
    }

    if (player !== humanPlayer) {
        aiThinking = true;
        updateDisplay(false); // Show "AI is thinking..."
        await new Promise(resolve => setTimeout(resolve, 20)); // Force repaint

        const move = await getAiMove(board, player);
        aiThinking = false;
        
        if (move) {
            const newBoard = makeMoveAndGetNewBoard(board, move.r, move.c, player);
            boardHistory.push(newBoard);
            playerHistory.push(-player);
        }
        gameLoop();
    } else {
        // Human's turn. Update UI and now calculate hints.
        const hints = await getHints();
        drawBoard(hints); // Redraw board with hints on top
    }
}

// --- UI & Drawing (REFACTORED) ---
function updateDisplay(withHints) {
    if (withHints) {
        updateDisplayWithHints();
    } else {
        drawBoard(null);
        updateInfoPanel();
    }
}
async function updateDisplayWithHints() {
    drawBoard(null);
    updateInfoPanel();
    const hints = await getHints();
    drawBoard(hints);
}
function drawBoard(hints) {
    const squareSize = canvas.clientWidth / BOARD_SIZE;
    const pieceRadius = squareSize / 2 - 5;
    canvas.width = canvas.clientWidth;
    canvas.height = canvas.clientHeight;
    
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.fillStyle = 'green';
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    for (let i = 0; i < BOARD_SIZE + 1; i++) {
        ctx.beginPath();
        ctx.moveTo(i * squareSize, 0); ctx.lineTo(i * squareSize, canvas.height);
        ctx.moveTo(0, i * squareSize); ctx.lineTo(canvas.width, i * squareSize);
        ctx.strokeStyle = 'black'; ctx.lineWidth = 1; ctx.stroke();
    }

    // --- ADDED: Draw professional board markings ---
    const dotRadius = squareSize / 20;
    // The dots are at the intersections of lines 2 & 6
    const dotLocations = [{r: 2, c: 2}, {r: 2, c: 6}, {r: 6, c: 2}, {r: 6, c: 6}];
    ctx.fillStyle = 'black'; // Use the same color as the grid lines
    for (const loc of dotLocations) {
        const centerX = loc.c * squareSize;
        const centerY = loc.r * squareSize;
        ctx.beginPath();
        ctx.arc(centerX, centerY, dotRadius, 0, 2 * Math.PI);
        ctx.fill();
    }
    // --- END ADDED CODE ---

    const board = getCurrentBoard();
    const legalMoves = (options.showLegalMoves && getCurrentPlayer() === humanPlayer) ? getLegalMoves(board, getCurrentPlayer()) : [];
    
    if (hints) {
        const maxHint = Math.max(...Object.values(hints));
        if (maxHint > 0) {
            for (const [actionStr, value] of Object.entries(hints)) {
                const move = JSON.parse(actionStr);
                const radius = pieceRadius * (value / maxHint);
                if (radius > 2) {
                    ctx.beginPath();
                    ctx.arc(move.c * squareSize + squareSize / 2, move.r * squareSize + squareSize / 2, radius, 0, 2 * Math.PI);
                    ctx.fillStyle = 'rgba(200, 200, 220, 0.5)';
                    ctx.fill();
                }
            }
        }
    }
    for (let r = 0; r < 8; r++) {
        for (let c = 0; c < 8; c++) {
            const piece = board[r][c];
            if (piece !== EMPTY) {
                ctx.beginPath();
                ctx.arc(c * squareSize + squareSize / 2, r * squareSize + squareSize / 2, pieceRadius, 0, 2 * Math.PI);
                ctx.fillStyle = piece === 1 ? 'white' : 'black';
                ctx.fill();
            } else if (legalMoves.some(m => m.r === r && m.c === c)) {
                ctx.beginPath();
                ctx.arc(c * squareSize + squareSize / 2, r * squareSize + squareSize / 2, pieceRadius / 4, 0, 2 * Math.PI);
                ctx.fillStyle = 'rgba(0, 0, 0, 0.3)';
                ctx.fill();
            }
        }
    }
}
function updateInfoPanel() {
    const board = getCurrentBoard();
    let blackScore = 0, whiteScore = 0;
    board.flat().forEach(p => { if (p === -1) blackScore++; if (p === 1) whiteScore++; });
    const blackLabel = humanPlayer === -1 ? "Human (B)" : "AI (B)";
    const whiteLabel = humanPlayer === 1 ? "Human (W)" : "AI (W)";
    scoresEl.textContent = `${blackLabel}: ${blackScore}, ${whiteLabel}: ${whiteScore}`;
    
    if (editMode) {
        statusEl.textContent = 'Board Edit Mode. Click squares to change pieces.';
        return;
    }
    if (gameOver) {
        let message = `Game Over! `;
        if (blackScore > whiteScore) message += `${blackLabel} Wins!`;
        else if (whiteScore > blackScore) message += `${whiteLabel} Wins!`;
        else message += "It's a Draw!";
        statusEl.textContent = message;
    } else if (aiThinking) {
        statusEl.textContent = `AI is thinking... (${options.aiMode})`;
    } else {
        statusEl.textContent = getCurrentPlayer() === humanPlayer ? "Your turn" : "AI's turn";
    }
}
async function getHints() {
    if (options.hintMode === 'none' || aiThinking || getCurrentPlayer() !== humanPlayer) {
        return null;
    }
    statusEl.textContent = `Calculating hints... (${options.hintMode})`;
    await new Promise(resolve => setTimeout(resolve, 20));
    const board = getCurrentBoard();
    const player = getCurrentPlayer();
    let hints = {};
    if (options.hintMode === 'policy') {
        const inputPlanes = createInputPlanes(board, player);
        const inputTensor = new ort.Tensor('float32', inputPlanes, [1, 6, 8, 8]);
        const feeds = { 'input': inputTensor };
        const results = await aiModel.run(feeds);
        const logits = results.policy.data;
        const legalMoves = getLegalMoves(board, player);
        const legalIndices = legalMoves.map(m => m.r * 8 + m.c);
        const legalLogits = legalIndices.map(i => logits[i]);
        const legalProbs = softmax(legalLogits);
        legalMoves.forEach((move, i) => { hints[JSON.stringify(move)] = legalProbs[i]; });
    } else if (options.hintMode === 'mcts') {
        const mcts = new MCTS(aiModel, 1.0, options.mctsSims);
        const { policy } = await mcts.get_move_and_policy(board, player);
        let totalVisits = 0;
        Object.values(policy).forEach(node => totalVisits += node.n_visits);
        if (totalVisits > 0) {
            for (const [actionStr, node] of Object.entries(policy)) {
                hints[actionStr] = node.n_visits / totalVisits;
            }
        }
    }
    if (!aiThinking) {
        statusEl.textContent = "Your turn";
    }
    return hints;
}
function createInputPlanes(board, player) {
    const planes = new Float32Array(6 * 8 * 8).fill(0);
    const legalMoves = getLegalMoves(board, player);
    for (let r = 0; r < 8; r++) { for (let c = 0; c < 8; c++) { const idx = r * 8 + c; if (board[r][c] === player) planes[idx] = 1; if (board[r][c] === -player) planes[64 + idx] = 1; if (board[r][c] === 0) planes[128 + idx] = 1; } }
    for (let i = 192; i < 256; i++) planes[i] = 1;
    for (const move of legalMoves) { planes[320 + move.r * 8 + move.c] = 1; }
    return planes;
}

// --- NEW/MODIFIED Event Handler Functions ---

function changePlayerColor(newColor) {
    humanPlayer = newColor;
    aiThinking = false; // Stop any ongoing thinking

    if (editMode) {
        // If in edit mode, just update the UI text and do nothing else.
        // The board and edit-mode state are preserved.
        updateInfoPanel();
        return;
    }
    
    // If in a regular game, don't reset the board. Just re-evaluate the turn.
    // The game state (board, current player) is preserved.
    gameLoop();
}

function handleNewGameClick() {
    if (editMode) {
        // If in edit mode, just reset the board to the start position but REMAIN in edit mode.
        const newBoard = createBoard();
        newBoard[3][3] = 1; newBoard[4][4] = 1;
        newBoard[3][4] = -1; newBoard[4][3] = -1;
        // Replace the current board state being edited
        boardHistory[boardHistory.length - 1] = newBoard;
        drawBoard(null);
        updateInfoPanel();
    } else {
        // Standard behavior: a full, hard reset of the game.
        initializeGame();
    }
}

function enterEditMode() {
    editMode = true;
    aiThinking = false; // Stop any ongoing AI or hint calculations

    // Toggle button visibility
    editBtn.style.display = 'none';
    resumeWhiteBtn.style.display = 'inline-block';
    resumeBlackBtn.style.display = 'inline-block';

    // Update UI and halt game
    updateInfoPanel();
    drawBoard(null); // Redraw to remove any existing hints
}

function exitEditMode(nextPlayer) {
    editMode = false;

    // Toggle button visibility back to normal
    editBtn.style.display = 'inline-block';
    resumeWhiteBtn.style.display = 'none';
    resumeBlackBtn.style.display = 'none';

    // Reset game state with the newly edited board
    const editedBoard = getCurrentBoard();
    // CRITICAL: Reset history, creating a deep copy of the edited board as the new starting point
    boardHistory = [editedBoard.map(row => row.slice())];
    playerHistory = [nextPlayer];
    gameOver = false;
    aiThinking = false;

    // Resume the game from the new state
    gameLoop();
}


// --- Event Listeners ---
function setupEventListeners() {
    // MODIFIED: 'player-color' listener now calls the new, non-destructive function.
    document.getElementById('player-color').addEventListener('change', (e) => changePlayerColor(parseInt(e.target.value)));
    document.getElementById('show-legal').addEventListener('change', (e) => { options.showLegalMoves = e.target.value === 'true'; gameLoop(); });
    document.getElementById('ai-mode').addEventListener('change', (e) => { options.aiMode = e.target.value; });
    document.getElementById('mcts-sims').addEventListener('change', (e) => { options.mctsSims = parseInt(e.target.value) || 200; });
    document.getElementById('hint-mode').addEventListener('change', (e) => { options.hintMode = e.target.value; gameLoop(); });
    // MODIFIED: 'restart-btn' listener now calls a context-aware handler.
    restartBtn.addEventListener('click', handleNewGameClick);
    undoBtn.addEventListener('click', undoMove);
    editBtn.addEventListener('click', enterEditMode);
    resumeWhiteBtn.addEventListener('click', () => exitEditMode(1)); // 1 for White
    resumeBlackBtn.addEventListener('click', () => exitEditMode(-1)); // -1 for Black
    const handleCanvasInput = (event) => {
        event.preventDefault();
        const rect = canvas.getBoundingClientRect();
        const squareSize = rect.width / BOARD_SIZE;
        let x, y;
        if (event.type === 'touchstart') {
            x = event.touches[0].clientX - rect.left;
            y = event.touches[0].clientY - rect.top;
        } else {
            x = event.clientX - rect.left;
            y = event.clientY - rect.top;
        }
        const c = Math.floor(x / squareSize);
        const r = Math.floor(y / squareSize);
        if (editMode) {
            handleBoardEdit(r, c);
        } else {
            handleHumanMove(r, c);
        }
    };
    canvas.addEventListener('click', handleCanvasInput);
    canvas.addEventListener('touchstart', handleCanvasInput);
    window.addEventListener('resize', () => { gameLoop(); });
}

// --- Start Application ---
loadModel().then(() => {
    setupEventListeners();
    initializeGame();
});