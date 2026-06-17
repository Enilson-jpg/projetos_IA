# -*- coding: utf-8 -*-
"""Simulated Annealing e Algoritmo Genetico no problema das N-Rainhas."""

import matplotlib.pyplot as plt
import random
import math

def random_state(n):
    """Gera uma permutacao aleatoria para o tabuleiro."""
    state = list(range(n))
    random.shuffle(state)
    return state

state_exemplo = random_state(8)

"""Cada estado e uma permutacao; os conflitos restantes ficam nas diagonais."""

def conflicts(state):
    """Conta pares de rainhas na mesma diagonal."""
    n = len(state)
    count = 0
    for i in range(n):
        for j in range(i + 1, n):
            if abs(state[i] - state[j]) == abs(i - j):
                count += 1
    return count

"""Visualizacao do tabuleiro."""

def plot_board(state, title=None):
    """Desenha o tabuleiro das N-Rainhas."""
    n = len(state)

    plt.figure(figsize=(6, 6))

    for row in range(n):
        for col in range(n):
            color_value = 0.90 if (row + col) % 2 == 0 else 0.65
            plt.gca().add_patch(
                plt.Rectangle((col, row), 1, 1, color=str(color_value))
            )

    for col, row in enumerate(state):
        plt.text(
            col + 0.5,
            row + 0.5,
            "♛",
            ha="center",
            va="center",
            fontsize=26
        )

    plt.xlim(0, n)
    plt.ylim(0, n)

    plt.xticks([i + 0.5 for i in range(n)], list(range(n)))
    plt.yticks([i + 0.5 for i in range(n)], list(range(n)))

    plt.gca().invert_yaxis()

    plt.grid(color="black", linewidth=0.5)

    for spine in plt.gca().spines.values():
        spine.set_visible(False)

    plt.gca().set_aspect("equal")

    if title is None:
        title = f"Tabuleiro {n}x{n} - Conflitos: {conflicts(state)}"

    plt.title(title)
    plt.xlabel("Colunas")
    plt.ylabel("Linhas")
    plt.show()

plot_board(state_exemplo)

"""Vizinho: troca duas colunas de lugar mantendo a permutacao."""

def neighbor(state):
    """Gera um vizinho por troca de duas posicoes."""
    n = len(state)
    new_state = state[:]
    i, j = random.sample(range(n), 2)
    new_state[i], new_state[j] = new_state[j], new_state[i]
    return new_state

"""## 5. Simulated Annealing

Aceita melhoras sempre e, no comeco, tambem aceita algumas pioras para escapar de minimos locais.
"""

def simulated_annealing(initial, T=1000, alpha=0.95, min_T=0.01):
    """Executa o Simulated Annealing e guarda o historico de conflitos."""
    current = initial
    current_cost = conflicts(current)

    best = current[:]
    best_cost = current_cost

    history = [current_cost]

    while T > min_T:
        next_state = neighbor(current)
        next_cost = conflicts(next_state)

        delta = next_cost - current_cost

        if delta < 0:
            current = next_state
            current_cost = next_cost

        else:
            prob = math.exp(-delta / T)
            if random.random() < prob:
                current = next_state
                current_cost = next_cost

        if current_cost < best_cost:
            best = current[:]
            best_cost = current_cost

        history.append(current_cost)

        T *= alpha

    return best, history

initial_sa = random_state(8)
best_sa, history_sa = simulated_annealing(initial_sa)

print("Estado inicial:", initial_sa)
print("Conflitos iniciais:", conflicts(initial_sa))
print("Melhor solução encontrada:", best_sa)
print("Conflitos finais:", conflicts(best_sa))

plot_board(initial_sa, "Simulated Annealing - Estado inicial")
plot_board(best_sa, "Simulated Annealing - Melhor estado encontrado")

plt.figure(figsize=(8, 4))
plt.plot(history_sa)
plt.title("Simulated Annealing - Evolução do número de conflitos")
plt.xlabel("Iterações")
plt.ylabel("Conflitos")
plt.show()

"""## 6. Algoritmo Genetico"""

def initial_population(pop_size, n):
    """Cria a populacao inicial."""
    return [random_state(n) for _ in range(pop_size)]

def fitness(state):
    """Quanto menos conflitos, maior o fitness."""
    n = len(state)
    max_pairs = n * (n - 1) // 2
    return max_pairs - conflicts(state)

def tournament_selection(population, k=3):
    """Seleciona o melhor entre k individuos sorteados."""
    contestants = random.sample(population, k)
    return max(contestants, key=fitness)

def crossover(parent1, parent2):
    """Order Crossover preservando a permutacao."""
    n = len(parent1)
    start, end = sorted(random.sample(range(n), 2))

    child = [-1] * n
    child[start:end+1] = parent1[start:end+1]

    segment_set = set(child[start:end+1])
    fill_values = [gene for gene in parent2 if gene not in segment_set]

    idx = 0
    for i in range(n):
        if child[i] == -1:
            child[i] = fill_values[idx]
            idx += 1

    return child

def mutate(state, mutation_rate=0.1):
    """Aplica mutacao por troca."""
    if random.random() < mutation_rate:
        return neighbor(state)
    return state[:]

def genetic_algorithm(n=8, pop_size=100, generations=500, mutation_rate=0.1):
    """Executa o algoritmo genetico e guarda o melhor custo por geracao."""
    population = initial_population(pop_size, n)

    best = min(population, key=conflicts)
    best_cost = conflicts(best)
    history = [best_cost]

    for gen in range(generations):
        if best_cost == 0:
            break

        new_population = []

        new_population.append(best[:])

        while len(new_population) < pop_size:
            parent1 = tournament_selection(population)
            parent2 = tournament_selection(population)
            child = crossover(parent1, parent2)
            child = mutate(child, mutation_rate)
            new_population.append(child)

        population = new_population

        current_best = min(population, key=conflicts)
        current_cost = conflicts(current_best)
        if current_cost < best_cost:
            best = current_best[:]
            best_cost = current_cost

        history.append(best_cost)

    return best, history

best_ga, history_ga = genetic_algorithm(n=8, pop_size=100, generations=500)

print("Melhor solução encontrada (GA):", best_ga)
print("Conflitos finais (GA):", conflicts(best_ga))

plot_board(best_ga, "Algoritmo Genético - Melhor estado encontrado")

plt.figure(figsize=(8, 4))
plt.plot(history_ga)
plt.title("Algoritmo Genético - Evolução do número de conflitos")
plt.xlabel("Gerações")
plt.ylabel("Conflitos (melhor indivíduo)")
plt.show()

"""## 7. Comparacao entre SA e GA"""

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

def draw_board_on_ax(ax, state, title):
    n = len(state)
    for row in range(n):
        for col in range(n):
            color_value = 0.90 if (row + col) % 2 == 0 else 0.65
            ax.add_patch(plt.Rectangle((col, row), 1, 1, color=str(color_value)))
    for col, row in enumerate(state):
        ax.text(col + 0.5, row + 0.5, "♛", ha="center", va="center", fontsize=22)
    ax.set_xlim(0, n); ax.set_ylim(0, n)
    ax.set_xticks([i + 0.5 for i in range(n)]); ax.set_xticklabels(range(n))
    ax.set_yticks([i + 0.5 for i in range(n)]); ax.set_yticklabels(range(n))
    ax.invert_yaxis()
    ax.grid(color="black", linewidth=0.5)
    for spine in ax.spines.values(): spine.set_visible(False)
    ax.set_aspect("equal")
    ax.set_title(title)

draw_board_on_ax(ax1, best_sa, f"Simulated Annealing\nConflitos: {conflicts(best_sa)}")
draw_board_on_ax(ax2, best_ga, f"Algoritmo Genético\nConflitos: {conflicts(best_ga)}")
plt.suptitle("Comparação: SA vs GA — Problema das 8 Rainhas", fontsize=14)
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 5))
plt.plot(history_sa, label="Simulated Annealing", alpha=0.8)
plt.plot(history_ga, label="Algoritmo Genético",  alpha=0.8)
plt.title("Comparação de Convergência: SA vs GA")
plt.xlabel("Iterações / Gerações")
plt.ylabel("Conflitos")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.5)
plt.show()

print("=" * 40)
print("       RESUMO COMPARATIVO")
print("=" * 40)
print(f"  SA  → Conflitos finais : {conflicts(best_sa)}")
print(f"  GA  → Conflitos finais : {conflicts(best_ga)}")
print(f"  SA  → Iterações        : {len(history_sa)}")
print(f"  GA  → Gerações         : {len(history_ga)}")
print("=" * 40)
if conflicts(best_sa) == 0 and conflicts(best_ga) == 0:
    print("  Ambos encontraram solução ótima!")
elif conflicts(best_sa) == 0:
    print("  SA encontrou solução ótima.")
elif conflicts(best_ga) == 0:
    print("  GA encontrou solução ótima.")
else:
    print("  Nenhum encontrou solução perfeita nesta execução.")
    print("  Tente rodar novamente ou ajustar os parâmetros.")
