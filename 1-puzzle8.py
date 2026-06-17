"""
Trabalho Prático #01 - Resolução do 8-Puzzle
Disciplina: Inteligência Artificial - UFC
Algoritmos: BFS e DFS

Uso:
    python puzzle8.py BFS 1 2 3 4 0 6 7 5 8
    python puzzle8.py DFS 1 2 3 4 0 6 7 5 8
"""

import sys
import time
from collections import deque


ESTADO_OBJETIVO = (1, 2, 3, 4, 5, 6, 7, 8, 0)

# Tabuleiro representado como tupla de 9 posições (linha por linha):
#   0 1 2
#   3 4 5
#   6 7 8
ACOES = {
    "cima":     -3,
    "baixo":    +3,
    "esquerda": -1,
    "direita":  +1,
}

RESTRICOES = {
    "esquerda": [0, 3, 6],
    "direita":  [2, 5, 8],
    "cima":     [0, 1, 2],
    "baixo":    [6, 7, 8],
}


class No:
    def __init__(self, estado, pai=None, acao=None, profundidade=0):
        self.estado = estado
        self.pai = pai
        self.acao = acao
        self.profundidade = profundidade

    def __eq__(self, other):
        return self.estado == other.estado

    def __hash__(self):
        return hash(self.estado)


def estado_para_str(estado):
    linhas = []
    for i in range(0, 9, 3):
        linhas.append(" ".join(str(v) for v in estado[i:i+3]))
    return "\n".join(linhas)


def obter_sucessores(estado):
    pos_vazio = estado.index(0)
    sucessores = []

    for acao, delta in ACOES.items():
        if pos_vazio in RESTRICOES[acao]:
            continue
        nova_pos = pos_vazio + delta
        novo_estado = list(estado)
        novo_estado[pos_vazio], novo_estado[nova_pos] = novo_estado[nova_pos], novo_estado[pos_vazio]
        sucessores.append((acao, tuple(novo_estado)))

    return sucessores


def reconstruir_caminho(no):
    caminho = []
    atual = no
    while atual.pai is not None:
        caminho.append((atual.acao, atual.estado))
        atual = atual.pai
    caminho.reverse()
    return caminho


def exibir_resultado(algoritmo, no_solucao, nos_expandidos, tempo_execucao, estado_inicial):
    print(f"\n{'='*40}")
    print(f"Algoritmo: {algoritmo}")
    print(f"{'='*40}")
    print("\nEstado Inicial:")
    print(estado_para_str(estado_inicial))

    if no_solucao is None:
        print("\nSolução não encontrada.")
    else:
        for i, (acao, estado) in enumerate(reconstruir_caminho(no_solucao), start=1):
            print(f"\nPasso {i}:")
            print(estado_para_str(estado))
        print("\nEstado Final")
        print(f"\nProfundidade da solução: {no_solucao.profundidade}")
        print(f"Nós expandidos: {nos_expandidos}")
        print(f"Tempo de execução: {tempo_execucao:.4f}s")


def bfs(estado_inicial):
    no_raiz = No(estado=estado_inicial)

    if no_raiz.estado == ESTADO_OBJETIVO:
        return no_raiz, 0

    fronteira = deque([no_raiz])
    visitados = {estado_inicial}
    nos_expandidos = 0

    while fronteira:
        no_atual = fronteira.popleft()
        nos_expandidos += 1

        for acao, novo_estado in obter_sucessores(no_atual.estado):
            if novo_estado not in visitados:
                no_filho = No(
                    estado=novo_estado,
                    pai=no_atual,
                    acao=acao,
                    profundidade=no_atual.profundidade + 1
                )

                if no_filho.estado == ESTADO_OBJETIVO:
                    return no_filho, nos_expandidos

                visitados.add(novo_estado)
                fronteira.append(no_filho)

    return None, nos_expandidos


def dfs(estado_inicial):
    no_raiz = No(estado=estado_inicial)

    if no_raiz.estado == ESTADO_OBJETIVO:
        return no_raiz, 0

    fronteira = [no_raiz]
    visitados = {estado_inicial}
    nos_expandidos = 0

    while fronteira:
        no_atual = fronteira.pop()
        nos_expandidos += 1

        for acao, novo_estado in obter_sucessores(no_atual.estado):
            if novo_estado not in visitados:
                no_filho = No(
                    estado=novo_estado,
                    pai=no_atual,
                    acao=acao,
                    profundidade=no_atual.profundidade + 1
                )

                if no_filho.estado == ESTADO_OBJETIVO:
                    return no_filho, nos_expandidos

                visitados.add(novo_estado)
                fronteira.append(no_filho)

    return None, nos_expandidos


def main():
    if len(sys.argv) != 11:
        print("Uso: python puzzle8.py <BFS|DFS> <9 números do estado inicial>")
        print("Exemplo: python puzzle8.py BFS 1 2 3 4 0 6 7 5 8")
        sys.exit(1)

    algoritmo = sys.argv[1].upper()
    if algoritmo not in ("BFS", "DFS"):
        print("Erro: o algoritmo deve ser BFS ou DFS.")
        sys.exit(1)

    try:
        numeros = list(map(int, sys.argv[2:11]))
    except ValueError:
        print("Erro: os 9 valores do estado inicial devem ser números inteiros.")
        sys.exit(1)

    if sorted(numeros) != list(range(9)):
        print("Erro: o estado inicial deve conter exatamente os números de 0 a 8.")
        sys.exit(1)

    estado_inicial = tuple(numeros)

    inicio = time.time()

    if algoritmo == "BFS":
        no_solucao, nos_expandidos = bfs(estado_inicial)
    else:
        no_solucao, nos_expandidos = dfs(estado_inicial)

    tempo_execucao = time.time() - inicio

    exibir_resultado(algoritmo, no_solucao, nos_expandidos, tempo_execucao, estado_inicial)


if __name__ == "__main__":
    main()
