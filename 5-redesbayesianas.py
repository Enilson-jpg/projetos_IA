# -*- coding: utf-8 -*-
"""Exemplos de Redes Bayesianas com pgmpy."""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import networkx as nx

from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination

plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 12

ESTADOS = ['Nao', 'Sim']
print("Bibliotecas importadas!")

def visualizar_rede(modelo, titulo="Rede Bayesiana"):
    plt.figure(figsize=(10, 6))
    pos = nx.circular_layout(modelo)
    nx.draw_networkx_nodes(modelo, pos, node_color='lightblue', node_size=2200)
    nx.draw_networkx_edges(modelo, pos, edge_color='gray', arrows=True,
                           arrowstyle='-|>', arrowsize=20, node_size=2200)
    nx.draw_networkx_labels(modelo, pos, font_size=12, font_weight='bold')
    plt.title(titulo, fontsize=16, fontweight='bold')
    plt.axis('off'); plt.tight_layout(); plt.savefig(titulo.replace(' ', '_') + '.png', dpi=80)
    plt.close()

"""Exemplo 1: diagnostico de gripe."""

modelo_gripe = DiscreteBayesianNetwork([
    ('Gripe', 'Febre'),
    ('Gripe', 'Tosse'),
    ('Gripe', 'DorCabeca'),
])
print("Nós:", list(modelo_gripe.nodes()))
print("Arestas:", list(modelo_gripe.edges()))

visualizar_rede(modelo_gripe, "Diagnostico de Gripe")

cpd_gripe = TabularCPD(
    variable='Gripe', variable_card=2,
    values=[[0.9],
            [0.1]],
    state_names={'Gripe': ESTADOS}
)
print(cpd_gripe)
cpd_febre = TabularCPD(
    variable='Febre', variable_card=2,
    values=[[0.90, 0.20],
            [0.10, 0.80]],
    evidence=['Gripe'], evidence_card=[2],
    state_names={'Febre': ESTADOS, 'Gripe': ESTADOS}
)
print(cpd_febre)

cpd_tosse = TabularCPD(
    variable='Tosse', variable_card=2,
    values=[[0.85, 0.30],
            [0.15, 0.70]],
    evidence=['Gripe'], evidence_card=[2],
    state_names={'Tosse': ESTADOS, 'Gripe': ESTADOS}
)
cpd_dorcabeca = TabularCPD(
    variable='DorCabeca', variable_card=2,
    values=[[0.80, 0.40],
            [0.20, 0.60]],
    evidence=['Gripe'], evidence_card=[2],
    state_names={'DorCabeca': ESTADOS, 'Gripe': ESTADOS}
)
print("CPTs de Tosse e DorCabeca prontas.")

modelo_gripe.add_cpds(cpd_gripe, cpd_febre, cpd_tosse, cpd_dorcabeca)
print("Modelo válido?", modelo_gripe.check_model())

"""Inferencia sobre gripe a partir dos sintomas."""

inferencia_gripe = VariableElimination(modelo_gripe)

print("\n--- P(Gripe) sem evidência ---")
print(inferencia_gripe.query(['Gripe']))

print("\n--- P(Gripe | Febre=Sim) ---")
print(inferencia_gripe.query(['Gripe'], evidence={'Febre': 'Sim'}))

print("\n--- P(Gripe | Febre=Sim, Tosse=Sim) ---")
print(inferencia_gripe.query(['Gripe'], evidence={'Febre': 'Sim', 'Tosse': 'Sim'}))

print("\n--- P(Gripe | Febre=Sim, Tosse=Sim, DorCabeca=Sim) ---")
print(inferencia_gripe.query(['Gripe'], evidence={'Febre': 'Sim', 'Tosse': 'Sim', 'DorCabeca': 'Sim'}))

"""Exemplo 2: sistema de alarme residencial."""

modelo_alarme = DiscreteBayesianNetwork([
    ('Roubo',     'Alarme'),
    ('Terremoto', 'Alarme'),
    ('Alarme',    'LigacaoJoao'),
    ('Alarme',    'LigacaoMaria'),
])
visualizar_rede(modelo_alarme, "Sistema de Alarme")

cpd_roubo = TabularCPD('Roubo', 2, [[0.999], [0.001]],
                       state_names={'Roubo': ESTADOS})
cpd_terremoto = TabularCPD('Terremoto', 2, [[0.998], [0.002]],
                           state_names={'Terremoto': ESTADOS})

cpd_alarme = TabularCPD(
    'Alarme', 2,
    [[0.999, 0.710, 0.060, 0.050],
     [0.001, 0.290, 0.940, 0.950]],
    evidence=['Roubo', 'Terremoto'], evidence_card=[2, 2],
    state_names={'Alarme': ESTADOS, 'Roubo': ESTADOS, 'Terremoto': ESTADOS}
)

cpd_joao = TabularCPD(
    'LigacaoJoao', 2,
    [[0.95, 0.10],
     [0.05, 0.90]],
    evidence=['Alarme'], evidence_card=[2],
    state_names={'LigacaoJoao': ESTADOS, 'Alarme': ESTADOS}
)

cpd_maria = TabularCPD(
    'LigacaoMaria', 2,
    [[0.99, 0.30],
     [0.01, 0.70]],
    evidence=['Alarme'], evidence_card=[2],
    state_names={'LigacaoMaria': ESTADOS, 'Alarme': ESTADOS}
)

modelo_alarme.add_cpds(cpd_roubo, cpd_terremoto, cpd_alarme, cpd_joao, cpd_maria)
print("Modelo válido?", modelo_alarme.check_model())

"""Inferencia no alarme e efeito explaining away."""

inferencia_alarme = VariableElimination(modelo_alarme)

print("\n--- P(Roubo | João=Sim, Maria=Sim) ---")
print(inferencia_alarme.query(
    ['Roubo'],
    evidence={'LigacaoJoao': 'Sim', 'LigacaoMaria': 'Sim'}
))

print("\n--- P(Roubo | João=Sim, Maria=Sim, Terremoto=Sim) ---")
print(inferencia_alarme.query(
    ['Roubo'],
    evidence={'LigacaoJoao': 'Sim', 'LigacaoMaria': 'Sim', 'Terremoto': 'Sim'}
))

print("\n--- P(Roubo | João=Sim, Maria=Sim, Terremoto=Nao) ---")
print(inferencia_alarme.query(
    ['Roubo'],
    evidence={'LigacaoJoao': 'Sim', 'LigacaoMaria': 'Sim', 'Terremoto': 'Nao'}
))

print("\n--- P(Roubo, Terremoto | João=Sim, Maria=Sim) ---")
print(inferencia_alarme.query(
    ['Roubo', 'Terremoto'],
    evidence={'LigacaoJoao': 'Sim', 'LigacaoMaria': 'Sim'}
))

"""Independencias locais no modelo do alarme."""

print("\n--- Independências locais de LigacaoJoao ---")
print(modelo_alarme.local_independencies('LigacaoJoao'))

print("\n--- Independências locais de LigacaoMaria ---")
print(modelo_alarme.local_independencies('LigacaoMaria'))

"""Aprendendo probabilidades a partir de dados sinteticos."""

np.random.seed(42)
N = 2000
gripe = np.random.choice(ESTADOS, size=N, p=[0.9, 0.1])

def amostra(pai, p_sim_se_sim, p_sim_se_nao):
    return ['Sim' if np.random.rand() < (p_sim_se_sim if g == 'Sim' else p_sim_se_nao)
            else 'Nao' for g in pai]

df = pd.DataFrame({
    'Gripe':     gripe,
    'Febre':     amostra(gripe, 0.8, 0.10),
    'Tosse':     amostra(gripe, 0.7, 0.15),
    'DorCabeca': amostra(gripe, 0.6, 0.20),
})
print(df.head())
print(df['Gripe'].value_counts(normalize=True).round(3))

modelo_aprendido = DiscreteBayesianNetwork([
    ('Gripe', 'Febre'), ('Gripe', 'Tosse'), ('Gripe', 'DorCabeca')
])
modelo_aprendido.fit(df)
print("\n--- CPT aprendida: Gripe ---")
print(modelo_aprendido.get_cpds('Gripe'))
print("\n--- CPT aprendida: Febre ---")
print(modelo_aprendido.get_cpds('Febre'))

"""Rede simples para recomendacao de filmes."""

IDADE   = ['Jovem', 'Adulto']

modelo_filmes = DiscreteBayesianNetwork([
    ('IdadeUsuario',  'GostaCiencia'),
    ('IdadeUsuario',  'GostaAcao'),
    ('GostaCiencia',  'AssistiuFilme'),
    ('GostaAcao',     'AssistiuFilme'),
    ('AssistiuFilme', 'AvaliacaoPositiva'),
])
visualizar_rede(modelo_filmes, "Recomendacao de Filmes")

cpd_idade = TabularCPD(
    'IdadeUsuario', 2,
    [[0.4],
     [0.6]],
    state_names={'IdadeUsuario': IDADE}
)

cpd_cie = TabularCPD(
    'GostaCiencia', 2,
    [[0.40, 0.30],
     [0.60, 0.70]],
    evidence=['IdadeUsuario'], evidence_card=[2],
    state_names={'GostaCiencia': ESTADOS, 'IdadeUsuario': IDADE}
)

cpd_aca = TabularCPD(
    'GostaAcao', 2,
    [[0.20, 0.50],
     [0.80, 0.50]],
    evidence=['IdadeUsuario'], evidence_card=[2],
    state_names={'GostaAcao': ESTADOS, 'IdadeUsuario': IDADE}
)

cpd_ass = TabularCPD(
    'AssistiuFilme', 2,
    [[0.90, 0.70, 0.60, 0.20],
     [0.10, 0.30, 0.40, 0.80]],
    evidence=['GostaCiencia', 'GostaAcao'], evidence_card=[2, 2],
    state_names={
        'AssistiuFilme': ESTADOS,
        'GostaCiencia':  ESTADOS,
        'GostaAcao':     ESTADOS
    }
)

cpd_ava = TabularCPD(
    'AvaliacaoPositiva', 2,
    [[1.00, 0.25],
     [0.00, 0.75]],
    evidence=['AssistiuFilme'], evidence_card=[2],
    state_names={'AvaliacaoPositiva': ESTADOS, 'AssistiuFilme': ESTADOS}
)

modelo_filmes.add_cpds(cpd_idade, cpd_cie, cpd_aca, cpd_ass, cpd_ava)
print("Modelo válido?", modelo_filmes.check_model())

inf_filmes = VariableElimination(modelo_filmes)

print("\n--- P(AssistiuFilme) sem evidência ---")
print(inf_filmes.query(['AssistiuFilme']))

print("\n--- P(AssistiuFilme | IdadeUsuario='Jovem') ---")
print(inf_filmes.query(['AssistiuFilme'], evidence={'IdadeUsuario': 'Jovem'}))

print("\n--- P(IdadeUsuario | AvaliacaoPositiva='Sim') ← inferência 'para trás' ---")
print(inf_filmes.query(['IdadeUsuario'], evidence={'AvaliacaoPositiva': 'Sim'}))

