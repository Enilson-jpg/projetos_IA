"""
Jogo da Velha com Expectiminimax
=================================
Variação do MINIMAX onde, antes de cada jogada de MAX,
um dado é lançado para determinar quais posições estão disponíveis:
  - Dado 1-3 → posições 0-4 disponíveis (lado esquerdo)
  - Dado 4-6 → posições 4-8 disponíveis (lado direito)

Probabilidade uniforme: P(dado 1-3) = P(dado 4-6) = 0.5

Tipos de nó na árvore:
  - Nó MAX       : escolhe a jogada com maior valor esperado
  - Nó CHANCE    : calcula E[v] = 0.5 * v(esq) + 0.5 * v(dir)
  - Nó MIN       : escolhe a jogada com menor valor (minimax normal)
  - Nó TERMINAL  : retorna +1 (MAX vence), -1 (MIN vence) ou 0 (empate)
"""

import random

POSITIONS_LEFT  = {0, 1, 2, 3, 4}
POSITIONS_RIGHT = {4, 5, 6, 7, 8}

SIMBOLO = {1: "X", -1: "O", 0: "."}

def create_board():
    return [0] * 9

def print_board(board):
    print()
    for row in range(3):
        cells = []
        for col in range(3):
            idx = row * 3 + col
            cells.append(SIMBOLO[board[idx]])
        print(f"  {cells[0]} | {cells[1]} | {cells[2]}   ({row*3} | {row*3+1} | {row*3+2})")
        if row < 2:
            print("  ---------")
    print()

def get_empty(board):
    return [i for i, v in enumerate(board) if v == 0]

LINHAS_VITORIA = [
    (0, 1, 2), (3, 4, 5), (6, 7, 8),  # linhas
    (0, 3, 6), (1, 4, 7), (2, 5, 8),  # colunas
    (0, 4, 8), (2, 4, 6),              # diagonais
]

def check_winner(board):
    """Retorna 1 se MAX ganhou, -1 se MIN ganhou, 0 caso contrário."""
    for a, b, c in LINHAS_VITORIA:
        if board[a] == board[b] == board[c] != 0:
            return board[a]
    return 0

def is_terminal(board):
    return check_winner(board) != 0 or len(get_empty(board)) == 0

def evaluate(board):
    return check_winner(board)

def expectiminimax(board, is_max_turn):
    """Retorna o valor esperado do estado atual."""
    if is_terminal(board):
        return evaluate(board)

    if is_max_turn:
        return chance_node(board)
    else:
        return min_node(board)

def chance_node(board):
    """Calcula a esperanca dos dois resultados possiveis do dado."""
    empty = set(get_empty(board))

    available_left  = list(empty & POSITIONS_LEFT)
    available_right = list(empty & POSITIONS_RIGHT)

    value_left  = max_node(board, available_left)
    value_right = max_node(board, available_right)

    return 0.5 * value_left + 0.5 * value_right

def max_node(board, available):
    """Escolhe a jogada disponivel que maximiza o valor esperado."""
    if not available:
        return min_node(board)

    best = float("-inf")
    for pos in available:
        board[pos] = 1
        val = expectiminimax(board, is_max_turn=False)
        board[pos] = 0
        if val > best:
            best = val
    return best

def min_node(board):
    """Escolhe a jogada de menor valor para MAX."""
    if is_terminal(board):
        return evaluate(board)

    best = float("inf")
    for pos in get_empty(board):
        board[pos] = -1
        val = expectiminimax(board, is_max_turn=True)
        board[pos] = 0
        if val < best:
            best = val
    return best

def best_move_max(board):
    """Lanca o dado e escolhe a melhor jogada valida para MAX."""
    dado = random.randint(1, 6)
    if dado <= 3:
        lado = "ESQUERDO (dado 1-3)"
        available_set = POSITIONS_LEFT
    else:
        lado = "DIREITO (dado 4-6)"
        available_set = POSITIONS_RIGHT

    empty = set(get_empty(board))
    available = list(empty & available_set)

    print(f"  🎲 Dado lançado: {dado} → lado {lado}")
    print(f"     Posições disponíveis para MAX: {sorted(available) if available else 'nenhuma!'}")

    if not available:
        print("     MAX não tem jogadas disponíveis — turno perdido.")
        return None, dado, lado

    best_val = float("-inf")
    best_pos = None
    for pos in available:
        board[pos] = 1
        val = expectiminimax(board, is_max_turn=False)
        board[pos] = 0
        if val > best_val:
            best_val = val
            best_pos = pos

    return best_pos, dado, lado

def best_move_min(board):
    """Escolhe o melhor movimento para MIN (minimax puro)."""
    best_val = float("inf")
    best_pos = None
    for pos in get_empty(board):
        board[pos] = -1
        val = expectiminimax(board, is_max_turn=True)
        board[pos] = 0
        if val < best_val:
            best_val = val
            best_pos = pos
    return best_pos

def simulate_game():
    board = create_board()
    turno = 1

    print("=" * 50)
    print("   JOGO DA VELHA - EXPECTIMINIMAX")
    print("   MAX = X  |  MIN = O")
    print("=" * 50)
    print_board(board)

    while not is_terminal(board):
        if turno == 1:
            print("--- Turno de MAX (X) ---")
            pos, dado, lado = best_move_max(board)
            if pos is not None:
                board[pos] = 1
                print(f"     MAX escolheu a posicao {pos}")
            print_board(board)
        else:
            print("--- Turno de MIN (O) ---")
            pos = best_move_min(board)
            board[pos] = -1
            print(f"     MIN escolheu a posicao {pos}")
            print_board(board)

        turno *= -1

    winner = check_winner(board)
    print("=" * 50)
    if winner == 1:
        print("  MAX (X) VENCEU!")
    elif winner == -1:
        print("  MIN (O) VENCEU!")
    else:
        print("  EMPATE!")
    print("=" * 50)

def run_simulations(n=100):
    results = {"MAX": 0, "MIN": 0, "EMPATE": 0}

    print(f"\nExecutando {n} simulações...\n")
    for i in range(n):
        board = create_board()
        turno = 1
        while not is_terminal(board):
            if turno == 1:
                pos, _, _ = best_move_max(board)
                if pos is not None:
                    board[pos] = 1
            else:
                pos = best_move_min(board)
                board[pos] = -1
            turno *= -1

        winner = check_winner(board)
        if winner == 1:
            results["MAX"] += 1
        elif winner == -1:
            results["MIN"] += 1
        else:
            results["EMPATE"] += 1

    print("=" * 40)
    print(f"  RESULTADOS ({n} partidas)")
    print("=" * 40)
    for k, v in results.items():
        pct = v / n * 100
        print(f"  {k:8s}: {v:4d} ({pct:5.1f}%)")
    print("=" * 40)

if __name__ == "__main__":
    print("\nEscolha o modo:")
    print("  1 - Simular uma partida (detalhada)")
    print("  2 - Rodar 100 simulações (estatísticas)")
    escolha = input("\nOpção: ").strip()

    if escolha == "2":
        run_simulations(100)
    else:
        simulate_game()
