# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.4
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown] id="Th3tUn8N5n4q"
# # 第3章

# %% id="APJ2_62w5oLu" outputId="87f77a7b-f3cb-452e-a51e-dabceb73e0f6" colab={"base_uri": "https://localhost:8080/"}
# !pip install pgmpy==0.1.24
# !pip install pyro-ppl==1.8.6
# !pip install graphviz==0.20.1
# !pip install dowhy==0.11.1

# %% [markdown] id="l_ioUCQs6zzQ"
# ## リスト3.1

# %% id="vurkTE0s5oUK"
def true_dgp(jenny_inclination, brian_inclination, window_strength):
    jenny_throws_rock = jenny_inclination > 0.5
    brian_throws_rock = brian_inclination > 0.5
    if jenny_throws_rock and brian_throws_rock:
        strength_of_impact = 0.8
    elif jenny_throws_rock or brian_throws_rock:
        strength_of_impact = 0.6
    else:
        strength_of_impact = 0.0
    window_breaks = window_strength < strength_of_impact
    return jenny_throws_rock, brian_throws_rock, window_breaks


# %% [markdown] id="kXlxbId87B20"
# ## リスト3.2

# %% id="7Xvl8R9j7CBk"
from pgmpy.models import BayesianNetwork
model = BayesianNetwork(
       [
        ('A', 'E'),
        ('S', 'E'),
        ('E', 'O'),
        ('E', 'R'),
        ('O', 'T'),
        ('R', 'T')
     ]
)

# %% [markdown] id="FFvQMs-f7GTa"
# ## リスト3.3

# %% colab={"base_uri": "https://localhost:8080/", "height": 424} id="amScTPVM7Gl9" outputId="9b79f394-80ba-492b-968d-b935454e6377"
import pandas as pd
url='https://raw.githubusercontent.com/altdeep/causalAI/master/datasets/transportation_survey.csv'
data = pd.read_csv(url)
data

# %% [markdown] id="bvr9LhbW7SEd"
# ## リスト3.4

# %% colab={"base_uri": "https://localhost:8080/"} id="9rzqtLUm7Tib" outputId="aa6b4389-21d3-4d1b-b6d1-c62d852c323e"
from pgmpy.models import BayesianNetwork
model = BayesianNetwork(
      [
        ('A', 'E'),
        ('S', 'E'),
        ('E', 'O'),
        ('E', 'R'),
        ('O', 'T'),
        ('R', 'T')
     ]
)

model.fit(data)

causal_markov_kernels = model.get_cpds()
print(causal_markov_kernels)

# %% id="1WjTCbYc4W-N" outputId="727b10e1-ae9e-4653-f778-83790ba5ad2f" colab={"base_uri": "https://localhost:8080/"}
cmk_T = causal_markov_kernels[-1]
print(cmk_T)

print(causal_markov_kernels[2])

# %% [markdown] id="OKkTwKKF7ach"
# ## リスト3.5

# %% colab={"base_uri": "https://localhost:8080/"} id="7do-dizG7p0-" outputId="1007d7c4-7048-47f7-be04-15ed495b82fb"
from pgmpy.estimators import BayesianEstimator
estimator = BayesianEstimator(model, data)
model.fit(
    data,
    estimator=BayesianEstimator,
    prior_type="dirichlet",
    pseudo_counts=1
)
causal_markov_kernels = model.get_cpds()
cmk_T = causal_markov_kernels[-1]
print(cmk_T)


# %% [markdown] id="TBwJbZvv7xym"
# ## リスト3.6

# %% colab={"base_uri": "https://localhost:8080/", "height": 531, "referenced_widgets": ["a3864206db4b4ff3ada29549349e82b9", "9d98fa142ecf44e283095ef1e8fe638f", "c286ca4e47bc40c2951389b2fc8313d3", "1d0a9c732e3749528c3c587bb7da39d4", "fce395ebd7224e2e9afd8d6f78346e42", "56dd7e0ea9724fac8d62956e999e5d71", "87dd0843125449c9ab8ad86021036dfa", "0f15f3fe25e34ee290142bcf87336046", "9cc4e346eb2a4d2eb87fc59338724932", "f2dfdd5542904a5992129577f260cc40", "d8cdf919c24c44a7bafac38ed0fc6ae2"]} id="VDWOW-ll7yED" outputId="8ac33b03-b9f8-431d-e174-8a30e2d04b3d"
import pandas as pd
from pgmpy.models import BayesianNetwork
from pgmpy.estimators import ExpectationMaximization as EM
url='https://raw.githubusercontent.com/altdeep/causalAI/master/datasets/transportation_survey.csv'
data = pd.read_csv(url)
data_sans_E = data[['A', 'S', 'O', 'R', 'T']]
model_with_latent = BayesianNetwork(
       [
        ('A', 'E'),
        ('S', 'E'),
        ('E', 'O'),
        ('E', 'R'),
        ('O', 'T'),
        ('R', 'T')
     ],
     latents={"E"}
)
estimator = EM(model_with_latent, data_sans_E)
cmks_with_latent = estimator.get_parameters(latent_card={'E': 2})
print(cmks_with_latent[1].to_factor())

# %% [markdown]
# > **補足(訳者)**: pgmpy 0.1.25 以降では `get_parameters()` が返す CPD リストの
# > 順序に実行時ランダム性があり、`cmks_with_latent[1]` が E とは限りません
# > (本文の環境では `[1]` が E でした)。確実に E の因果マルコフカーネル
# > (本文 p.93 の表)を表示するには、次のように変数名で選択します。

# %%
cmk_E = next(cpd for cpd in cmks_with_latent if cpd.variable == "E")
print(cmk_E.to_factor())


# %% [markdown] id="TT4kdobC749Y"
# ## リスト3.7

# %% colab={"base_uri": "https://localhost:8080/"} id="STcTW75t75FM" outputId="530acc1a-3adb-4b15-843a-26657ae46545"
from pgmpy.inference import VariableElimination
inference = VariableElimination(model)
query1 = inference.query(['E'], evidence={"T": "train"})
query2 = inference.query(['E'], evidence={"T": "car"})
print("train")
print(query1)
print("car")
print(query2)


# %% [markdown] id="cyz8hkgn8An5"
# ## リスト3.8

# %% colab={"base_uri": "https://localhost:8080/", "height": 368} id="dkyLvwtj8Avu" outputId="5e23b1cb-cd69-4c63-b197-6bcf4cf9082e"
import torch
import pyro
from pyro.distributions import Categorical

A_alias = ['young', 'adult', 'old']
S_alias = ['M', 'F']
E_alias = ['high', 'uni']
O_alias = ['emp', 'self']
R_alias = ['small', 'big']
T_alias = ['car', 'train', 'other']

A_prob = torch.tensor([0.3,0.5,0.2])
S_prob = torch.tensor([0.6,0.4])
E_prob = torch.tensor([[[0.75,0.25], [0.72,0.28], [0.88,0.12]],
                     [[0.64,0.36], [0.7,0.3], [0.9,0.1]]])
O_prob = torch.tensor([[0.96,0.04], [0.92,0.08]])
R_prob = torch.tensor([[0.25,0.75], [0.2,0.8]])
T_prob = torch.tensor([[[0.48,0.42,0.1], [0.56,0.36,0.08]],
                     [[0.58,0.24,0.18], [0.7,0.21,0.09]]])

def model():
   A = pyro.sample("age", Categorical(probs=A_prob))
   S = pyro.sample("gender", Categorical(probs=S_prob))
   E = pyro.sample("education", Categorical(probs=E_prob[S][A]))
   O = pyro.sample("occupation", Categorical(probs=O_prob[E]))
   R = pyro.sample("residence", Categorical(probs=R_prob[E]))
   T = pyro.sample("transportation", Categorical(probs=T_prob[R][O]))
   return{'A': A,'S': S,'E': E,'O': O,'R': R,'T': T}

pyro.render_model(model)


# %% [markdown] id="SmL3KOSd8G9o"
# ## リスト3.9

# %% colab={"base_uri": "https://localhost:8080/", "height": 490} id="jw2WgGZ08HFK" outputId="1cc3180f-2ab5-4b3b-b90b-b1922a8decaa"
import numpy as np
import pyro
from pyro.distributions import Categorical
from pyro.infer import Importance, EmpiricalMarginal
import matplotlib.pyplot as plt

conditioned_model = pyro.condition(
    model,
    data={'transportation':torch.tensor(1.)}
)

m = 5000
posterior = pyro.infer.Importance(
    conditioned_model,
    num_samples=m
).run()
E_marginal = EmpiricalMarginal(posterior, "education")
E_samples = [E_marginal().item() for _ in range(m)]
E_unique, E_counts = np.unique(E_samples, return_counts=True)
E_probs = E_counts / m

plt.bar(E_unique, E_probs, align='center', alpha=0.5)
plt.xticks(E_unique, E_alias)
plt.ylabel('probability')
plt.xlabel('education')
plt.title('P(E | T = "train") - Importance Sampling')


# %% [markdown] id="CRjDV6Hp8POy"
# ## リスト3.10

# %% colab={"base_uri": "https://localhost:8080/", "height": 406} id="NWoJFO8C8PVZ" outputId="edcd5e8c-4f28-4e99-8877-3267afb2d764"
from dowhy import datasets

import networkx as nx
import matplotlib.pyplot as plt

sim_data = datasets.linear_dataset(
    beta=10.0,
    num_treatments=1,
    num_instruments=2,
    num_effect_modifiers=2,
    num_common_causes=5,
    num_frontdoor_variables=1,
    num_samples=100,

)
dag = nx.parse_gml(sim_data['gml_graph'])
pos = {
 'X0': (600, 350),
 'X1': (600, 250),
 'FD0': (300, 300),
 'W0': (0, 400),
 'W1': (150, 400),
 'W2': (300, 400),
 'W3': (450, 400),
 'W4': (600, 400),
 'Z0': (10, 250),
 'Z1': (10, 350),
 'v0': (100, 300),
 'y': (500, 300)
}
options = {
    "font_size": 12,
    "node_size": 800,
    "node_color": "white",
    "edgecolors": "black",
    "linewidths": 1,
    "width": 1,
}
nx.draw_networkx(dag, pos, **options)
ax = plt.gca()
ax.margins(x=0.40)
plt.axis("off")
plt.show()
