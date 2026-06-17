# -*- coding: utf-8 -*-
"""Enilson Silva - UFC IA - Busca com Informação

# **Laboratório: Busca com Informação**
Este notebook contém um exercício prático para aplicar os algoritmos de busca com informação vistos em sala **Greedy Best-first e A***. Nesta atividade, exploraremos como algoritmos de busca podem ser aplicados em cenários reais de tomada de decisão.

## **Objetivo**
Você deverá implementar os dois algoritmos mencionados previamente e aplicá-los aos problemas fornecidos. Siga as instruções passo a passo e veja o resultado ao final de cada seção. Lembre-se:
1. Greedy Best-First Search (Busca Gulosa): Foca apenas na estimativa de distância até o objetivo ($h(n)$).  
2. A* (A-Estrela): Combina o custo real acumulado ($g(n)$) com a estimativa futura ($h(n)$).
"""

import math
import heapq
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.lines import Line2D
import seaborn as sns
import networkx as nx
from itertools import combinations

"""Base simplificada inspirada em "Artificial Intelligence: A Modern Approach". A solução retorna o caminho e o custo total.
"""

class Node:
  def __init__(self, state, parent=None, action=None, path_cost=0):
    self.state = state
    self.parent = parent
    self.action = action
    self.path_cost = path_cost

  def __lt__(self, other):
    return self.path_cost < other.path_cost

  def __repr__(self):
    return f"<Node {self.state}>"


class Problem:
  def __init__(self, initial_state, goal_state=None):
    self.initial_state = initial_state
    self.goal_state = goal_state

  def actions(self, state):
    raise NotImplementedError

  def result(self, state, action):
    raise NotImplementedError

  def is_goal(self, state):
    return state == self.goal_state

  def action_cost(self, s, action, s1):
    return 1

def solution(node):
  actions = []
  total_cost = node.path_cost
  curr = node
  while curr.parent is not None:
    actions.append(curr.action)
    curr = curr.parent
  return actions[::-1], total_cost

def expand(problem, node):
  s = node.state
  for action in problem.actions(s):
    s1 = problem.result(s, action)
    cost = node.path_cost + problem.action_cost(s, action, s1)
    yield Node(state=s1, parent=node, action=action, path_cost=cost)

"""A busca uniforme servirá de referência."""

def uniform_cost_search(problem):
  node = Node(problem.initial_state)
  frontier = []
  heapq.heappush(frontier, (node.path_cost, node))
  reached = {problem.initial_state: node}
  while frontier:
    if not frontier:
      return None, float('inf')
    priority, node = heapq.heappop(frontier)
    if problem.is_goal(node.state):
      return solution(node)
    for child in expand(problem, node):
      s = child.state
      if s not in reached or child.path_cost < reached[s].path_cost:
        reached[s] = child
        heapq.heappush(frontier, (child.path_cost, child))
  return None, float('inf')

"""## Problema 1
### O Network do Ícaro
**Contexto**  
Seu amigo de infância, Ícaro, é um desenvolvedor talentoso que acaba de lançar um jogo de nicho. Ele é brilhante no código, mas invisível nas redes. O objetivo dele é fazer o jogo chegar ao "Zenith", o streamer de referência do gênero.  
Ícaro mapeou algumas conexões, mas o ecossistema de criadores de conteúdo é caótico. Conseguir uma introdução de um streamer para outro envolve favores e negociações cujos "custos" reais são difíceis de prever com precisão antes de começar a conversa.  
Por sorte, você tem metadados ricos, então pode usar a **Busca Gulosa**.

### Definição
"""

class SocialClimberProblem(Problem):
  def __init__(self, initial, goal, graph, profiles):
    super().__init__(initial, goal)
    self.graph = graph
    self.profiles = profiles

  def actions(self, state):
    return list(self.graph.get(state, {}).keys())

  def result(self, state, action):
    return action

  def action_cost(self, s, action, s1):
    return self.graph[s][s1]

"""### Instância"""

profiles = {
  'Icaro':        {'followers': 150,     'interests': [0.1, 0.8, 0.9, 0.1]},
  'Alpha_Play':   {'followers': 1200,    'interests': [0.18, 0.02, 0.16, 0.04]},
  'Bits_Pro':     {'followers': 5000,    'interests': [0.1, 0.1, 0.1, 0.1]},
  'Cyber_Neon':   {'followers': 10000,   'interests': [0.1, 0.1, 0.1, 0.1]},
  'Dev_Quest':    {'followers': 1000,    'interests': [0.09, 0.01, 0.08, 0.02]},
  'Early_Bird':   {'followers': 50000,   'interests': [0.7, 0.1, 0.7, 0.2]},
  'FPS_King':     {'followers': 1100,    'interests': [0.88, 0.12, 0.78, 0.22]},
  'Gamer_X':      {'followers': 1000,    'interests': [0.85, 0.15, 0.75, 0.25]},
  'Hunter_Dev':   {'followers': 15000,   'interests': [0.6, 0.2, 0.6, 0.2]},
  'Jester_Box':   {'followers': 900,     'interests': [0.8, 0.2, 0.7, 0.3]},
  'Knight_Table': {'followers': 600000,  'interests': [0.1, 0.9, 0.1, 0.9]},
  'Logic_Bot':    {'followers': 1500,    'interests': [0.27, 0.03, 0.24, 0.06]},
  'Mana_Bar':     {'followers': 900000,  'interests': [0.1, 0.9, 0.1, 0.9]},
  'Nerd_Cave':    {'followers': 5000,    'interests': [0.5, 0.3, 0.5, 0.2]},
  'Ogre_Layer':   {'followers': 800,     'interests': [0.7, 0.3, 0.6, 0.4]},
  'PC_Master':    {'followers': 20000,   'interests': [0.1, 0.1, 0.1, 0.1]},
  'Quest_Line':   {'followers': 150000,  'interests': [0.8, 0.1, 0.8, 0.2]},
  'Retro_Vibe':   {'followers': 300000,  'interests': [0.1, 0.9, 0.1, 0.9]},
  'Skill_Check':  {'followers': 30000,   'interests': [0.1, 0.1, 0.1, 0.1]},
  'Zenith':       {'followers': 1000000, 'interests': [0.9, 0.1, 0.8, 0.2]}
}

adj_matrix = {
  'Icaro': {'Nerd_Cave': 5, 'Dev_Quest': 8, 'Ogre_Layer': 12},
  'Nerd_Cave': {'Retro_Vibe': 15, 'Hunter_Dev': 20},
  'Dev_Quest': {'Bits_Pro': 25, 'Alpha_Play': 10},
  'Ogre_Layer': {'Jester_Box': 15},
  'Retro_Vibe': {'Knight_Table': 30},
  'Hunter_Dev': {'Early_Bird': 10, 'Quest_Line': 40},
  'Alpha_Play': {'Logic_Bot': 15},
  'Bits_Pro': {'Skill_Check': 50, 'Cyber_Neon': 30},
  'Jester_Box': {'Gamer_X': 20},
  'Knight_Table': {'Mana_Bar': 10},
  'Early_Bird': {'Quest_Line': 15},
  'Quest_Line': {'Zenith': 50},
  'Logic_Bot': {'Quest_Line': 20, 'Zenith': 100},
  'Cyber_Neon': {'PC_Master': 20, 'Skill_Check': 40},
  'Gamer_X': {'FPS_King': 10},
  'Mana_Bar': {'Zenith': 80},
  'PC_Master': {'Zenith': 150},
  'Skill_Check': {'Zenith': 200},
  'FPS_King': {'Zenith': 500},
  'Zenith': {}
}

pos_fixed = {
  'Icaro': (0, 0.0),
  'Nerd_Cave': (1, 0.0),
  'Hunter_Dev': (2, 0.0),
  'Quest_Line': (4, 0.0),
  'Zenith': (5, 0.0),
  'Early_Bird': (3, 0.5),
  'Ogre_Layer': (1, 2.0),
  'Retro_Vibe': (2, 1),
  'Jester_Box': (2, 2.0),
  'Knight_Table': (3, 1),
  'Gamer_X': (3, 2.0),
  'Mana_Bar': (4, 1),
  'FPS_King': (4, 2.0),
  'Dev_Quest': (2, -2),
  'Alpha_Play': (2, -1),
  'Logic_Bot': (3, -1),
  'Bits_Pro': (3, -2),
  'Skill_Check': (4, -1),
  'Cyber_Neon': (4, -2),
  'PC_Master': (5, -2)
}

prob_icaro = SocialClimberProblem('Icaro', 'Zenith', adj_matrix, profiles)

"""#### Visualização"""

data = []
for name, info in profiles.items():
  data.append({
    'Name': name,
    'Initial': name[0],
    'Followers': info['followers'],
    **{f'D{i}': val for i, val in enumerate(info['interests'])}
  })
df = pd.DataFrame(data)

dim_names = ['Tech', 'Lifestyle', 'Indie', 'Mobile']
style_map = {
  'Zenith': {'color': '#D4AF37', 'marker': '*', 'size': 250, 'alpha': 1.0, 'zorder': 10},
  'Icaro':  {'color': '#444444', 'marker': '^', 'size': 150, 'alpha': 1.0, 'zorder': 10},
  'default':{'color': 'lightblue', 'marker': 'o', 'size': 100, 'alpha': 0.5, 'zorder': 1}
}

df_sorted = df.sort_values('Followers')
plt.figure(figsize=(12, 8))

colors = [style_map.get(name, style_map['default'])['color'] for name in df_sorted['Name']]
bars = plt.barh(df_sorted['Name'], df_sorted['Followers'], color=colors, edgecolor='black', alpha=0.8)

plt.xscale('log')
for bar in bars:
  w = bar.get_width()
  offset, align, txt_col = (1.1, 'left', 'black') if w < 500 else (0.95, 'right', 'black')
  plt.text(w * offset, bar.get_y() + bar.get_height()/2, f'{int(w):,}',
            va='center', ha=align, fontsize=11, fontweight='bold', color=txt_col)

plt.title('Alcance: Seguidores (Escala Log)', fontsize=20)
plt.grid(axis='x', linestyle='--', alpha=0.4)
plt.tight_layout()
plt.show()

combos = list(combinations(range(4), 2))
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
axes = axes.flatten()

for i, (d_x, d_y) in enumerate(combos):
  ax = axes[i]
  for _, row in df.iterrows():
    s = style_map.get(row['Name'], style_map['default'])

    ax.scatter(row[f'D{d_x}'], row[f'D{d_y}'], c=s['color'], s=s['size'],
                marker=s['marker'], edgecolors='black', alpha=s['alpha'], zorder=s['zorder'])

    ax.text(row[f'D{d_x}'], row[f'D{d_y}'] + 0.03, row['Initial'],
            fontsize=9, ha='center', fontweight='bold')

  ax.set_xlabel(dim_names[d_x])
  ax.set_ylabel(dim_names[d_y])
  ax.set_xlim(-0.05, 1.05); ax.set_ylim(-0.05, 1.05)
  ax.grid(True, linestyle=':', alpha=0.6)
  ax.set_title(f'{dim_names[d_x]} vs {dim_names[d_y]}')

plt.suptitle('Espaço de Estados: Afinidade de Nicho (I=Ícaro, Z=Zenith)', fontsize=25, y=1.02)
plt.tight_layout()
plt.show()

"""### Implementação do Greedy Best-first"""

def greedy_best_first_search(problem, h_func):
  node = Node(problem.initial_state)
  frontier = []
  heapq.heappush(frontier, (h_func(node), node))
  reached = {problem.initial_state: node}
  while frontier:
    _, node = heapq.heappop(frontier)
    if problem.is_goal(node.state):
      return solution(node)
    for child in expand(problem, node):
      s = child.state
      if s not in reached:
        reached[s] = child
        heapq.heappush(frontier, (h_func(child), child))
  return None, float('inf')

"""### Heurística
Ícaro forneceu diversas informações sobre o Zenith e os demais influenciadores. Você deve implementar a sua própria heurística $h(n)$ para guiar a busca gulosa. Lembre-se que a busca gulosa ignora o passado, então uma heurística mal projetada pode fazer o Ícaro girar em círculos entre "famosos" que não têm nenhuma conexão real com o Zenith.
"""

def h_icarus(node):
  # Combina afinidade com o Zenith e diferença de alcance.
  current = node.state
  zenith_interests = profiles['Zenith']['interests']
  current_interests = profiles[current]['interests']
  dist_interests = math.sqrt(sum((a - b) ** 2 for a, b in zip(current_interests, zenith_interests)))
  zenith_followers = profiles['Zenith']['followers']
  current_followers = profiles[current]['followers']
  follower_gap = (zenith_followers - current_followers) / zenith_followers
  return 50 * dist_interests + 30 * max(0, follower_gap)

"""### Teste"""

print("--- RESULTADOS ÍCARO ---")
actions_greedy_icaro, cost_greedy_icaro = greedy_best_first_search(prob_icaro, h_icarus)
actions_ucs_icaro, cost_ucs_icaro = uniform_cost_search(prob_icaro)
print(f"Busca Gulosa: {' -> '.join(actions_greedy_icaro) if actions_greedy_icaro else 'Falha'} | Custo: {cost_greedy_icaro}")
print(f"Busca com Custo Uniforme: {' -> '.join(actions_ucs_icaro) if actions_ucs_icaro else 'Falha'} | Custo: {cost_ucs_icaro}")

"""#### Visualização"""

def visualizar_comparativo_grafo(problem, adj_matrix, sol_proposal, sol_ref, pos_layout):
  G = nx.DiGraph()
  for u, neighbors in adj_matrix.items():
    for v, weight in neighbors.items():
      G.add_edge(u, v, weight=weight)

  pos = pos_layout
  fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))

  def draw_graph(ax, solution_tuple, title, color):
    actions, total_cost = solution_tuple

    nx.draw_networkx_nodes(G, pos, ax=ax, node_color='lightblue', node_size=1000)
    nx.draw_networkx_edges(G, pos, ax=ax, edge_color='gray', alpha=0.3, arrowsize=20)

    if actions:
      path_states = [problem.initial_state]
      current_state = problem.initial_state
      for action in actions:
        current_state = problem.result(current_state, action)
        path_states.append(current_state)
      path_edges = list(zip(path_states, path_states[1:]))

      nx.draw_networkx_nodes(G, pos, ax=ax, nodelist=path_states, node_color=color, node_size=1000)
      nx.draw_networkx_edges(G, pos, ax=ax, edgelist=path_edges, edge_color=color, width=4, arrowsize=25)

    labels_dict = {node: node[0] for node in G.nodes()}
    nx.draw_networkx_labels(G, pos, ax=ax, labels=labels_dict, font_size=15)

    ax.set_title(f"{title}\nCusto Total: {total_cost if actions else 'N/A'}", fontsize=20)

    edge_labels = nx.get_edge_attributes(G, 'weight')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, ax=ax, font_size=15)

  draw_graph(ax1, sol_proposal, "Sua Solução", "salmon")
  draw_graph(ax2, sol_ref, "Referência", "lightgreen")

  plt.suptitle("Comparativo de Networking: Ícaro rumo ao Zenith", fontsize=30)
  plt.tight_layout()
  plt.show()

visualizar_comparativo_grafo(prob_icaro, adj_matrix, (actions_greedy_icaro, cost_greedy_icaro), (actions_ucs_icaro, cost_ucs_icaro), pos_fixed)

"""## Problema 2
### NPC em Ambiente de Risco
**Contexto**  
Você foi contratado por um grande estúdio de jogos para trabalhar em um título de stealth/survival. O projeto está em uma fase crítica e a imersão é a prioridade número 1.  
Durante uma missão de infiltração, o parceiro do jogador, Unit-R3, precisa ajudá-lo carregando suprimentos. Se o Unit-R3 for detectado por drones ou cair em poças de óleo, que reduzem sua velocidade, ele pode ser detectado e a imersão será quebrada por um Game Over frustrante causado por uma IA ruim.  
Sua tarefa como desenvolvedor de IA é implementar o cérebro do Unit-R3. Ele deve navegar por um grid 2D repleto de perigos e chegar o mais rápido possível. Você precisa garantir que ele encontre o caminho de Custo Mínimo (Ótimo), contornando zonas de risco (hordas de sentinelas) e obstáculos.  
Para este problema, a precisão é vital, então você usará o **Algoritmo A***.

### Definição
"""

class ScavengerGridProblem(Problem):
  def __init__(self, initial, goal, grid_map):
    super().__init__(initial, goal)
    self.grid = grid_map
    self.rows, self.cols = grid_map.shape

  def actions(self, state):
    x, y = state
    acts = []
    for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
      nx, ny = x + dx, y + dy
      if 0 <= nx < self.rows and 0 <= ny < self.cols:
        if self.grid[nx, ny] != float('inf'):
          acts.append((nx, ny))
    return acts

  def result(self, state, action):
    return action

  def action_cost(self, s, action, s1):
    return self.grid[s1]

"""### Instância"""

def apply_enemy_zone(grid, ex, ey):
  if not np.isinf(grid[ex, ey]):
    grid[ex, ey] = 50

  for r_offset in range(-2, 3):
    for c_offset in range(-2, 3):
      r, c = ex + r_offset, ey + c_offset

      if r_offset == 0 and c_offset == 0:
        continue

      if not (0 <= r < grid.shape[0] and 0 <= c < grid.shape[1]):
        continue

      if np.isinf(grid[r, c]):
        continue

      dist = max(abs(r_offset), abs(c_offset))
      if dist == 1:
        grid[r, c] = 25
      elif dist == 2:
        grid[r, c] = 10


# Mapa do cenário: terreno livre, óleo, paredes e áreas de risco.
world = np.ones((20, 20))

# Paredes
world[5, 0:15] = np.inf
world[15, 5:18] = np.inf
world[10:15, 10] = np.inf

# Poças de óleo
world[1:5, 8:12] = 5
world[16:19, 2:6] = 5
world[8:12, 12:16] = 5

# Inimigos
apply_enemy_zone(world, 10, 5)
apply_enemy_zone(world, 1, 14)
apply_enemy_zone(world, 13, 18)

start_pos = (0, 0)
goal_pos = (19, 19)

prob_unit_r3 = ScavengerGridProblem(start_pos, goal_pos, world)

"""#### Visualização"""

rows, cols = world.shape

display_grid = np.copy(world)
display_grid[np.isinf(display_grid)] = 100

colors = ['lightgray', '#F1C40F', 'lightcoral', 'firebrick', 'darkred', '#2C3E50']
cmap = ListedColormap(colors)
bounds = [0, 2, 8, 15, 30, 60, 110]
norm = BoundaryNorm(bounds, cmap.N)

fig, ax = plt.subplots(figsize=(10, 10))

img = ax.imshow(display_grid, cmap=cmap, norm=norm)

ax.plot(start_pos[1], start_pos[0], 'o', color='blue', markersize=15, label='Início (Unit-R3)', markeredgecolor='white')
ax.plot(goal_pos[1], goal_pos[0], '*', color='yellow', markersize=20, label='Extração (Goal)', markeredgecolor='black')

ey, ex = np.where(world == 50)
ax.scatter(ex, ey, marker='X', color='black', s=100, edgecolors='white', label='Núcleo Inimigo', zorder=5)

ax.set_xticks(np.arange(-0.5, cols, 1), minor=True)
ax.set_yticks(np.arange(-0.5, rows, 1), minor=True)
ax.grid(which="minor", color="black", linestyle='-', linewidth=0.5, alpha=0.3)
ax.set_xticks(np.arange(0, cols, 5))
ax.set_yticks(np.arange(0, rows, 5))

plt.title("Protocolo de Extração Unit-R3: Mapa", fontsize=18, pad=15)

custom_lines = [
  Line2D([0], [0], color='lightgray', lw=8, label='Livre (Custo 1)'),
  Line2D([0], [0], color='#F1C40F', lw=8, label='Óleo (Custo 5)'),
  Line2D([0], [0], color='lightcoral', lw=8, label='Suspeita (Custo 10)'),
  Line2D([0], [0], color='firebrick', lw=8, label='Visão (Custo 25)'),
  Line2D([0], [0], color='darkred', lw=8, label='Inimigo (Custo 50)'),
  Line2D([0], [0], color='#2C3E50', lw=8, label='Parede (Impassível)')
]
ax.legend(handles=custom_lines + [
  Line2D([0], [0], marker='o', color='blue', lw=0, label='Início'),
  Line2D([0], [0], marker='*', color='yellow', markeredgecolor='black', lw=0, label='Fim')
], loc='upper left', bbox_to_anchor=(1.02, 1), borderaxespad=0)

plt.tight_layout()
plt.show()

"""### Implementação do A*"""

def a_star_search(problem, h_func):
  node = Node(problem.initial_state)
  frontier = []
  heapq.heappush(frontier, (h_func(node) + node.path_cost, node))
  reached = {problem.initial_state: node}
  while frontier:
    _, node = heapq.heappop(frontier)
    if problem.is_goal(node.state):
      return solution(node)
    for child in expand(problem, node):
      s = child.state
      if s not in reached or child.path_cost < reached[s].path_cost:
        reached[s] = child
        f = child.path_cost + h_func(child)
        heapq.heappush(frontier, (f, child))
  return None, float('inf')

"""### Heurística
O Unit-R3 se move em um grid (Norte, Sul, Leste, Oeste). Você deve implementar a sua própria heurística $h(n)$ para guiar o A*. Lembre-se que o caminho será ótimo se a heurística for admissível (nunca superestimar o custo real).
"""

def h_unit_r3(node):
  # Manhattan é admissível aqui porque cada movimento custa pelo menos 1.
  x, y = node.state
  gx, gy = goal_pos
  return abs(x - gx) + abs(y - gy)

"""### Teste  
Se o Unit-R3 decidir atravessar uma zona de sentinelas (custo alto) tendo um caminho seguro por perto, você falhou na implementação da lógica e a jogatina do usuário foi interrompida por frustração.
"""

def coords_to_arrows(problem, actions):
  if not actions:
    return "Falha"

  arrows = []
  current_s = problem.initial_state

  for next_s in actions:
    dx = next_s[0] - current_s[0]
    dy = next_s[1] - current_s[1]

    if dx == -1:   arrows.append("↑")
    elif dx == 1:  arrows.append("↓")
    elif dy == 1:  arrows.append("→")
    elif dy == -1: arrows.append("←")

    current_s = next_s

  return " ".join(arrows)

print("--- RESULTADOS UNIT-R3 ---")

actions_astar_r3, cost_astar_r3 = a_star_search(prob_unit_r3, h_unit_r3)
actions_ucs_r3, cost_ucs_r3 = uniform_cost_search(prob_unit_r3)

str_astar_r3 = coords_to_arrows(prob_unit_r3, actions_astar_r3)
str_ucs_r3 = coords_to_arrows(prob_unit_r3, actions_ucs_r3)

print(f"Busca A*: Custo {cost_astar_r3} \n{str_astar_r3}")
print(f"Busca com Custo Uniforme: Custo {cost_ucs_r3} \n{str_ucs_r3}")

if actions_astar_r3 and actions_ucs_r3:
  if cost_astar_r3 == cost_ucs_r3:
    print("\nAmbos os algoritmos encontraram o caminho ótimo.")
  else:
    print("\nOs custos divergem. Verifique se a heurística é admissível.")

"""#### Visualização"""

def visualizar_comparativo_grid(problem, grid_map, sol_proposal, sol_ref):
  fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(22, 9))
  rows, cols = grid_map.shape

  def draw_grid_map(ax, solution_tuple, title):
    actions, total_cost = solution_tuple
    enemy_core_positions = []

    for r in range(rows):
      for c in range(cols):
        val = grid_map[r, c]
        color = 'lightgray'
        if np.isinf(val): color = '#2C3E50'
        elif val == 50:
            color = 'darkred'
            enemy_core_positions.append((c, r))
        elif val == 25: color = 'firebrick'
        elif val == 20: color = 'maroon'
        elif val == 10: color = 'lightcoral'
        elif val == 5:  color = '#F1C40F'

        ax.add_patch(plt.Rectangle((c, r), 1, 1, facecolor=color, edgecolor='black', linewidth=0.1))

    ax.set_xticks(np.arange(cols + 1))
    ax.set_yticks(np.arange(rows + 1))
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.grid(True, which='major', color='black', linestyle='-', linewidth=0.1)
    ax.set_xlim(0, cols)
    ax.set_ylim(rows, 0)
    ax.set_aspect('equal')

    start = problem.initial_state
    goal = problem.goal_state

    ax.plot(start[1] + 0.5, start[0] + 0.5, 'o', color='blue', markersize=15, zorder=5)
    ax.plot(goal[1] + 0.5, goal[0] + 0.5, '*', color='yellow', markersize=18, markeredgecolor='black', zorder=5)

    for ec_x, ec_y in enemy_core_positions:
      ax.plot(ec_x + 0.5, ec_y + 0.5, 'X', color='black', markersize=10, markeredgecolor='white', zorder=4)

    if actions:
      path_states = [problem.initial_state]
      curr = problem.initial_state
      for a in actions:
        curr = problem.result(curr, a)
        path_states.append(curr)
      path_array = np.array(path_states)

      path_cols = path_array[:, 1] + 0.5
      path_rows = path_array[:, 0] + 0.5

      ax.plot(path_cols, path_rows, color='magenta', linewidth=3, linestyle='-', zorder=3)
      ax.plot(path_cols, path_rows, 'o', color='white', markersize=4, markeredgecolor='magenta', zorder=4)

    ax.set_title(f"{title}\nCusto Total: {total_cost if actions else 'N/A'}", fontsize=18)

  draw_grid_map(ax1, sol_proposal, "Sua Solução")
  draw_graph_ref = draw_grid_map(ax2, sol_ref, "Referência")

  legend_handles = [
    Rectangle((0, 0), 1, 1, fc='lightgray', label='Livre (1)'),
    Rectangle((0, 0), 1, 1, fc='#F1C40F', label='Óleo (5)'),
    Rectangle((0, 0), 1, 1, fc='lightcoral', label='Suspeita (10)'),
    Rectangle((0, 0), 1, 1, fc='firebrick', label='Visão (25)'),
    Rectangle((0, 0), 1, 1, fc='darkred', label='Inimigo (50)'),
    Rectangle((0, 0), 1, 1, fc='#2C3E50', label='Parede (inf)'),
    plt.Line2D([0], [0], color='magenta', lw=3, label='Caminho'),
    plt.Line2D([0], [0], marker='o', color='blue', lw=0, label='Início'),
    plt.Line2D([0], [0], marker='*', color='yellow', markeredgecolor='black', lw=0, label='Fim')
  ]
  fig.legend(handles=legend_handles, loc='center right', bbox_to_anchor=(1, 0.5), fontsize=12)

  plt.suptitle("Protocolo de Extração Unit-R3: Comparativo de Rotas", fontsize=26)
  plt.tight_layout(rect=[0, 0, 0.92, 1])
  plt.show()

visualizar_comparativo_grid(prob_unit_r3, world, (actions_astar_r3, cost_astar_r3), (actions_ucs_r3, cost_ucs_r3))

"""## Bônus
Aplique o seu greedy best-first no Problema 2 e veja o resultado.
"""

print("--- RESULTADOS UNIT-R3 ---")

actions_greedy_r3, cost_greedy_r3 = greedy_best_first_search(prob_unit_r3, h_unit_r3)
actions_ucs_r3, cost_ucs_r3 = uniform_cost_search(prob_unit_r3)

str_greedy_r3 = coords_to_arrows(prob_unit_r3, actions_greedy_r3)
str_ucs_r3 = coords_to_arrows(prob_unit_r3, actions_ucs_r3)

print(f"Busca Gulosa: Custo {cost_greedy_r3} \n{str_greedy_r3}")
print(f"Busca com Custo Uniforme: Custo {cost_ucs_r3} \n{str_ucs_r3}")

if actions_greedy_r3 and actions_ucs_r3:
  if cost_greedy_r3 == cost_ucs_r3:
    print("\nAmbos os algoritmos encontraram o caminho ótimo.")
  else:
    print("\nOs custos divergem.")

visualizar_comparativo_grid(prob_unit_r3, world, (actions_greedy_r3, cost_greedy_r3), (actions_ucs_r3, cost_ucs_r3))