# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.4
#   kernelspec:
#     display_name: Python 3
#     name: python3
# ---

# %% [markdown] id="04EsYE9GSGSx"
# # 第10章
# ## リスト10.1

# %% colab={"base_uri": "https://localhost:8080/", "height": 485} id="RYQtWQwORXmr" outputId="4864e836-84bb-4ce3-b1f8-1b056dbebf9e"
# !pip install git+https://github.com/y0-causal-inference/y0.git@v0.2.0
from y0.dsl import P, Variable
E = Variable("E")
I = Variable("I")
query = P[E](I)
query

# %% [markdown] id="VlDdEA4qeNZ4"
# ## リスト10.2

# %% id="uXmrkBlMAfUR"
# !pip install graphviz==0.20.3
# !apt install libgraphviz-dev
# !pip install pygraphviz==1.13
# !pip install networkx==2.8.8

# %% colab={"base_uri": "https://localhost:8080/", "height": 1000} id="1N6zhxf4dPQE" outputId="a461f11e-e876-435c-d6c8-baff5f21f3cf"
import requests
def download_code(url):
    response = requests.get(url)
    if response.status_code == 200:
        code_content = response.text
        print("Code fetched successfully.")
        return code_content
    else:
        print("Failed to fetch code.")
        return None
url = (
    "https://raw.githubusercontent.com/altdeep/"
    "causalAI/master/book/chapter%2010/id_utilities.py"
)
utilities_code = download_code(url)
print(utilities_code)

from y0.graph import NxMixedGraph as Y0Graph
from y0.dsl import P, Variable
G = Variable("G")
E = Variable("E")
I = Variable("I")
dag = Y0Graph.from_edges(
    directed=[
        (G, E),
        (G, I),
        (E, I)
    ]
)
gv_draw(dag)

# %% [markdown] id="pd5nsJbnea5r"
# ## リスト10.3

# %% id="jveAVZLfehkh" colab={"base_uri": "https://localhost:8080/"} outputId="bfef9816-ae9a-464a-829b-b0dd5261a8af"
e = E
check_identifiable(
    dag,
    query=P(I @ e),
    distribution=P(G, E, I)
)

# %% [markdown] id="m2QbmFXBe9Rg"
# ## リスト10.4

# %% id="2wA20Vgme9Zv" colab={"base_uri": "https://localhost:8080/"} outputId="c5770a59-ac81-4a96-f27e-634db44f5ef0"
check_identifiable(
    dag,
    query=P(I @ e),
    distribution=P(E, I)
)

# %% [markdown] id="kg8Xl-jHfxzF"
# ## リスト10.5

# %% id="zGW2352If8Gi" colab={"base_uri": "https://localhost:8080/", "height": 40} outputId="08af772b-ea0e-4698-a682-c7c86604e1a1"
from y0.graph import NxMixedGraph as Y0Graph
from y0.dsl import P, Variable
from y0.algorithm.identify import Identification, identify

query=P(I @ e)
base_distribution = P(I, E, G)

identification_task = Identification.from_expression(
    graph=dag,
    query=query,
    estimand=base_distribution)

identify(identification_task)

# %% [markdown] id="KrJVU7v6gHDj"
# ## リスト10.6

# %% id="5bRCN6l9gHMO" colab={"base_uri": "https://localhost:8080/", "height": 40} outputId="03e4d95c-1bd0-4cbd-856e-9898b5f7e7df"
from y0.graph import NxMixedGraph as Y0Graph
from y0.dsl import P, Variable
G = Variable("G")
E = Variable("E")
I = Variable("I")
W = Variable("W")
e = E
dag = Y0Graph.from_edges(
    directed=[
        (G, E),
        (G, I),
        (E, W),
        (W, I)
    ]
)

query=P(I @ e)
base_distribution = P(I, E, W)

identification_task = Identification.from_expression(
    graph=dag,
    query=query,
    estimand=base_distribution)
identify(identification_task)

# %% [markdown] id="bJasnneNhdJr"
# ## リスト10.7

# %% id="nr2bQ1HKhdRS" colab={"base_uri": "https://localhost:8080/", "height": 364} outputId="7ac164aa-c821-40cd-e866-ac919f2cfd20"
T = Variable("T")
W = Variable("W")
B = Variable("B")
V = Variable("V")
C = Variable("C")
A = Variable("A")
t, a, w, v, b = T, A, W, V, B
dag = Y0Graph.from_edges(directed=[
    (T, W),
    (W, A),
    (B, V),
    (V, A),
    (C, T),
    (C, A),
    (C, B)
])
gv_draw(dag)

# %% [markdown] id="CvJ4lmejielR"
# ## リスト10.8

# %% id="JOwIj21wieuk" colab={"base_uri": "https://localhost:8080/", "height": 48} outputId="add9391d-6b72-4314-d4d6-6907790474a8"
from y0.algorithm.identify.idc_star import idc_star

idc_star(
    dag,
    outcomes={A @ -t: +a},
    conditions={T: +t}
)

# %% [markdown] id="YYtqS1UTi4bk"
# ## リスト10.9

# %% id="6rHwWd-vi4j8" colab={"base_uri": "https://localhost:8080/", "height": 460} outputId="f3dc61e0-721e-4b37-bb8e-204f1a9c0ca3"
from y0.algorithm.identify.cg import make_parallel_worlds_graph
parallel_world_graph = make_parallel_worlds_graph(
    dag,
     {frozenset([+t])}
)
gv_draw(parallel_world_graph)

# %% [markdown] id="wBY3TRgejFl1"
# ## リスト10.10

# %% id="uzs-HVzRjFtA" colab={"base_uri": "https://localhost:8080/", "height": 364} outputId="aed27ea5-33d9-49af-8f43-70355f2174e2"
from y0.algorithm.identify.cg import make_counterfactual_graph

events = {A @ -t: +a, T: +t}
cf_graph, _ = make_counterfactual_graph(dag, events)
gv_draw(cf_graph)

# %% [markdown] id="JB9uG6h4jpXD"
# ## リスト10.11

# %% id="oUlorTBmjpfL" colab={"base_uri": "https://localhost:8080/", "height": 460} outputId="91817b13-a966-49c4-b402-1bd80dc59042"
parallel_world_graph = make_parallel_worlds_graph(
    dag,
   {frozenset([-t]), frozenset([-b])}
)
gv_draw(parallel_world_graph)

# %% [markdown] id="LxXNdeOcknSj"
# ## リスト10.12

# %% id="AYNoaJCjknbL" colab={"base_uri": "https://localhost:8080/", "height": 364} outputId="7bfcfba1-6b3e-4e63-9448-806411cc5486"
joint_query = {A @ -t: +a, T: +t, B: -b, V @ -b: +v}
cf_graph, _ = make_counterfactual_graph(dag, joint_query)
gv_draw(cf_graph)

# %% [markdown] id="YElKhgJbk5YI"
# ## リスト10.13

# %% id="r2fzVr0Kk5gj" colab={"base_uri": "https://localhost:8080/"} outputId="e4c30680-5999-427e-9b3a-84fa4b664446"
# !pip install numpyro==0.15.0
# !pip install funsor==0.4.5
import jax.numpy as np
from jax import random
from numpyro import sample
from numpyro.handlers import condition, do
from numpyro.distributions import Bernoulli, Normal
from numpyro.infer import MCMC, NUTS
import matplotlib.pyplot as plt

rng = random.PRNGKey(1)

def model():
    p_member = 0.5
    is_guild_member = sample(
        "Guild Membership",
        Bernoulli(p_member)
    )
    p_engaged = (0.8*is_guild_member + 0.2*(1-is_guild_member))
    is_highly_engaged = sample(
        "Side-quest Engagement",
        Bernoulli(p_engaged)
    )
    p_won_engaged = (.9*is_highly_engaged + .1*(1-is_highly_engaged))
    high_won_items = sample("Won Items", Bernoulli(p_won_engaged))
    mu = (
        37.95*(1-is_guild_member)*(1-high_won_items) +
        54.92*(1-is_guild_member)*high_won_items +
        223.71*(is_guild_member)*(1-high_won_items) +
        125.50*(is_guild_member)*high_won_items
    )
    sigma = (
        23.80*(1-is_guild_member)*(1-high_won_items) +
        4.92*(1-is_guild_member)*high_won_items +
        5.30*(is_guild_member)*(1-high_won_items) +
        53.49*(is_guild_member)*high_won_items
    )
    in_game_purchases = sample("In-game Purchases", Normal(mu, sigma))

# %% [markdown] id="xCAIPpZqmF3r"
# ## リスト10.14

# %% id="FpX7qkYqmGCu" colab={"base_uri": "https://localhost:8080/"} outputId="8f2c1477-7c38-413d-bb0a-731e6153d2a5"
intervention_model = do(model, {"Side-quest Engagement": np.array(0.)})
intervention_kernel = NUTS(intervention_model)
intervention_model_sampler = MCMC(
    intervention_kernel,
    num_samples=5000,
    num_warmup=200
)
intervention_model_sampler.run(rng)
intervention_samples = intervention_model_sampler.get_samples()
int_purchases_samples = intervention_samples["In-game Purchases"]

# %% [markdown] id="fz3t_Xa3mYZ6"
# ## リスト10.15

# %% id="oy8chWgMmYgt" colab={"base_uri": "https://localhost:8080/"} outputId="37ce177d-9e0a-4821-af7f-9726b7de7d76"
cond_and_int_model = condition(
    intervention_model,
     {"Side-quest Engagement": np.array(1.)}
)
int_cond_kernel = NUTS(cond_and_int_model)
int_cond_model_sampler = MCMC(
    int_cond_kernel,
    num_samples=5000,
    num_warmup=200
)
int_cond_model_sampler.run(rng)
int_cond_samples = int_cond_model_sampler.get_samples()
int_cond_purchases_samples = int_cond_samples["In-game Purchases"]

# %% [markdown] id="RZx3PiNFoioM"
# ## リスト10.16

# %% id="7D2GFH-WtLFp" colab={"base_uri": "https://localhost:8080/", "height": 430} outputId="c598e3bb-65e0-46dc-f27d-978862ba7052"
plt.hist(
    int_purchases_samples,
    bins=30,
    alpha=0.5,
    label='$P(I_{E=0})$'
)
plt.hist(
    int_cond_purchases_samples,
    bins=30,
    alpha=0.5,
    label='$P(I_{E=0}|E=1)$'
)
plt.legend(loc='upper left')
plt.show()
