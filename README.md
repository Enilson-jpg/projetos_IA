# Projetos de Inteligencia Artificial

Colecao de atividades praticas da disciplina de Inteligencia Artificial, com exemplos de algoritmos de busca, otimizacao, jogos e redes Bayesianas.

## Conteudo

| Arquivo | Tema |
| --- | --- |
| `1-puzzle8.py` | Resolucao do 8-Puzzle com BFS e DFS |
| `2-busca_com_informação.py` | Busca informada com Greedy Best-First Search e A* |
| `3-busca_em_ambiente_complexos.py` | Simulated Annealing e Algoritmo Genetico para N-Rainhas |
| `4-minmax.py` | Jogo da velha com Expectiminimax |
| `5-redesbayesianas.py` | Exemplos de Redes Bayesianas com `pgmpy` |

## Requisitos

- Python 3.10 ou superior
- Bibliotecas listadas em `requirements.txt`

Instalacao das dependencias:

```bash
pip install -r requirements.txt
```

## Como executar

Execute os scripts individualmente pelo terminal:

```bash
python 1-puzzle8.py BFS 1 2 3 4 0 6 7 5 8
python 1-puzzle8.py DFS 1 2 3 4 0 6 7 5 8
python "2-busca_com_informação.py"
python 3-busca_em_ambiente_complexos.py
python 4-minmax.py
python 5-redesbayesianas.py
```

Alguns scripts exibem graficos ou geram imagens durante a execucao.

## Estrutura

```text
Projetos_IA/
├── 1-puzzle8.py
├── 2-busca_com_informação.py
├── 3-busca_em_ambiente_complexos.py
├── 4-minmax.py
├── 5-redesbayesianas.py
├── requirements.txt
├── .gitignore
└── README.md
```
