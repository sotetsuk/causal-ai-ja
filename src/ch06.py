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

# %% [markdown] id="HZk5tKUPWCz1"
# # 第6章

# %% colab={"base_uri": "https://localhost:8080/"} id="k5t759R1OOsm" outputId="10712f42-5a62-4360-81c4-394a7229380a"
# !pip install pyro-ppl==1.9.0
# !pip install pgmpy==0.1.25
# !pip install matplotlib==3.7.1

# %% [markdown] id="EKPeuhZDAsyH"
# ## リスト6.1

# %% colab={"base_uri": "https://localhost:8080/"} id="b583pQrCAQPY" outputId="82246879-22d5-4282-ba47-3b7e88156fe0"
from pyro.distributions import Normal
from pyro import sample

def cgm_model():
    x = sample("x", Normal(47., 2.3))
    y = sample("y", Normal(25. + 3*x, 3.3))
    return x, y


cgm_model()

# %% [markdown] id="Nb5IPa95Akh3"
# ## リスト6.2

# %% colab={"base_uri": "https://localhost:8080/"} id="1ha4VYBrAY2Z" outputId="407697e5-a4fb-4b7c-8255-52a646cc080a"
from pyro.distributions import Normal
from pyro import sample

def scm_model():
    n_x = sample("n_x", Normal(0., 2.3))
    n_y = sample("n_y", Normal(0., 3.3))
    x = 47. + n_x
    y = 25. + 3.*x + n_y
    return x, y


scm_model()

# %% [markdown] id="NMsI-S47CfM4"
# ## リスト6.3

# %% colab={"base_uri": "https://localhost:8080/"} id="-qdwoljNCfUc" outputId="2f952da4-dc40-4ca8-e81c-882e9262eeca"
import pandas as pd
import random

def true_dgp(
    jenny_inclination,
    brian_inclination,
    window_strength):
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

generated_outcome = true_dgp(
    jenny_inclination=random.uniform(0, 1),
    brian_inclination=random.uniform(0, 1),
    window_strength=random.uniform(0, 1)
)

generated_outcome

# %% [markdown] id="UvdnAhuMg8sI"
# ## リスト6.4

# %% id="84uuWOxOg814"
from pgmpy.factors.discrete.CPD import TabularCPD
f_host_door_selection = TabularCPD(
    variable='Host Door Selection',
    variable_card=3,
    values=[
        [0,0,0,0,1,1,0,1,1,0,0,0,0,0,1,0,1,0],
        [1,0,1,0,0,0,1,0,0,0,0,1,0,0,0,1,0,1],
        [0,1,0,1,0,0,0,0,0,1,1,0,1,1,0,0,0,0]
    ],
    evidence=[
        'Host Inclination',
        'Door with Car',
        'Player First Choice'
    ],
    evidence_card=[2, 3, 3],
    state_names={
        'Host Door Selection':['1st', '2nd', '3rd'],
        'Host Inclination': ['left', 'right'],
        'Door with Car': ['1st', '2nd', '3rd'],
        'Player First Choice': ['1st', '2nd', '3rd']
    }
)

# %% [markdown] id="E-eb9dlXhD7g"
# ## リスト6.5

# %% id="n7Doyfi1hEDu"
from pgmpy.factors.discrete.CPD import TabularCPD
f_second_choice = TabularCPD(
    variable='Player Second Choice',
    variable_card=3,
    values=[
        [1,0,0,1,0,0,1,0,0,0,0,0,0,0,1,0,1,0],
        [0,1,0,0,1,0,0,1,0,1,0,1,0,1,0,1,0,1],
        [0,0,1,0,0,1,0,0,1,0,1,0,1,0,0,0,0,0]
    ],
    evidence=[
        'Strategy',
        'Host Door Selection',
        'Player First Choice'
    ],
    evidence_card=[2, 3, 3],
    state_names={
        'Player Second Choice': ['1st', '2nd', '3rd'],
        'Strategy': ['stay', 'switch'],
        'Host Door Selection': ['1st', '2nd', '3rd'],
        'Player First Choice': ['1st', '2nd', '3rd']
    }
)

# %% [markdown] id="c6galsxYhKaP"
# ## リスト6.6

# %% id="hNGtshHehKg5"
from pgmpy.models import BayesianNetwork
from pgmpy.factors.discrete.CPD import TabularCPD

monty_hall_model = BayesianNetwork([
    ('Host Inclination', 'Host Door Selection'),
    ('Door with Car', 'Host Door Selection'),
    ('Player First Choice', 'Host Door Selection'),
    ('Player First Choice', 'Player Second Choice'),
    ('Host Door Selection', 'Player Second Choice'),
    ('Strategy', 'Player Second Choice'),
    ('Player Second Choice', 'Win or Lose'),
    ('Door with Car', 'Win or Lose')
])

# %% [markdown] id="j38hiE-5hSP9"
# ## リスト6.7

# %% id="6LsD6Zg1hSYZ"
p_host_inclination = TabularCPD(
    variable='Host Inclination',
    variable_card=2,
    values=[[.5], [.5]],
    state_names={'Host Inclination': ['left', 'right']}
)

p_door_with_car = TabularCPD(
    variable='Door with Car',
    variable_card=3,
    values=[[1/3], [1/3], [1/3]],
    state_names={'Door with Car': ['1st', '2nd', '3rd']}
)

p_player_first_choice = TabularCPD(
    variable='Player First Choice',
    variable_card=3,
    values=[[1/3], [1/3], [1/3]],
    state_names={'Player First Choice': ['1st', '2nd', '3rd']}
)

p_host_strategy = TabularCPD(
    variable='Strategy',
    variable_card=2,
    values=[[.5], [.5]],
    state_names={'Strategy': ['stay', 'switch']}
)


# %% [markdown] id="UU3GI-21heXX"
# ## リスト6.8

# %% id="W7DzpT1Qheeg"
f_win_or_lose = TabularCPD(
    variable='Win or Lose',
    variable_card=2,
    values=[
        [1,0,0,0,1,0,0,0,1],
        [0,1,1,1,0,1,1,1,0],
    ],
    evidence=['Player Second Choice', 'Door with Car'],
    evidence_card=[3, 3],
    state_names={
        'Win or Lose': ['win', 'lose'],
        'Player Second Choice': ['1st', '2nd', '3rd'],
        'Door with Car': ['1st', '2nd', '3rd']
    }
)

# %% [markdown] id="IkkA76m2hiLa"
# ## リスト6.9

# %% id="lR0aKgWQhiTJ"
monty_hall_model.add_cpds(
    p_host_inclination,
    p_door_with_car,
    p_player_first_choice,
    p_host_strategy,
    f_host_door_selection,
    f_second_choice,
    f_win_or_lose
)

# %% [markdown] id="d7zL5VnjhoPi"
# ## リスト6.10

# %% id="gKjhvAYWhoZD" colab={"base_uri": "https://localhost:8080/"} outputId="3ee29036-19ba-459c-ddcb-8ec7221ce11e"
from pgmpy.inference import VariableElimination

infer = VariableElimination(monty_hall_model)
q1 = infer.query(['Win or Lose'], evidence={'Strategy': 'stay'})
print(q1)
q2 = infer.query(['Win or Lose'], evidence={'Strategy': 'switch'})
print(q2)
q3 = infer.query(['Strategy'], evidence={'Win or Lose': 'win'})
print(q3)

# %% [markdown] id="DKeb7BG4izNz"
# ## リスト6.11

# %% id="F0vZ52TwizYK"
from pyro import sample
from pyro.distributions import Normal

def linear_gaussian():
    n_x = sample("N_x", Normal(9., 3.))
    n_y = sample("N_y", Normal(9., 3.))
    x = 10. + n_x
    y = 2. * x + n_y
    return x, y


# %% [markdown] id="VKjyEuw1nW-c"
# ## リスト6.12

# %% id="HdyR1orSnXHk"
from pyro import sample
from pyro.distributions import Gamma

def LiNGAM():
    n_x = sample("N_x", Gamma(9., 1.))
    n_y = sample("N_y", Gamma(9., 1.))
    x = 10. + n_x
    y = 2. * x + n_y
    return x, y


# %% [markdown] id="OR0LDp-jswFz"
# ## リスト6.13

# %% id="UROROw8Jc1-8"
from torch import nn

class EnzymeModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.β = nn.Parameter(torch.randn(1, 1))

    def forward(self, x):
        x = torch.mul(x, self.β)
        x = x.log().sigmoid()
        x = torch.mul(x, 100.)
        return x


# %% [markdown] id="hbeN7JdKc6EQ"
# ## リスト6.14

# %% id="gzC5hPusswd7" colab={"base_uri": "https://localhost:8080/"} outputId="2cc82d8e-ffa5-41d5-f05b-fe3757ccd07b"
import pandas as pd
from torch import tensor
import torch

df = pd.read_csv("https://raw.githubusercontent.com/altdeep/causalAI/master/datasets/enzyme-data.csv")
X = torch.tensor(df['x'].values).unsqueeze(1).float()
Y = torch.tensor(df['y'].values).unsqueeze(1).float()

def train(X, Y, model, loss_function, optim, num_epochs):
    loss_history = []
    for epoch in range(num_epochs):
        Y_pred = model(X)
        loss = loss_function(Y_pred, Y)
        loss.backward()
        optim.step()
        optim.zero_grad()
        if epoch % 1000 == 0:
            print(round(loss.data.item(), 6))

torch.manual_seed(1)
enzyme_model = EnzymeModel()
optim = torch.optim.Adam(enzyme_model.parameters(), lr=0.00001)
loss_function = nn.MSELoss()

train(X, Y, enzyme_model, loss_function, optim, num_epochs=60000)

# %% [markdown] id="pQZa4R3qs4zJ"
# ## リスト6.15

# %% id="IuhQtLy2s5aP" colab={"base_uri": "https://localhost:8080/"} outputId="9e73993f-01eb-4c03-d092-adda4b764eb4"
import pyro
from pyro.distributions import Beta, Normal, Uniform
from pyro.infer.mcmc import NUTS, MCMC

def g(u):
  return u / (1 + u)

def model(N):
    β = pyro.sample("β", Beta(0.5, 5.0))
    with pyro.plate("data", N):
        x = pyro.sample("X", Uniform(0.0, 101.0))
        y = pyro.sample("Y", Normal(100.0 * g(β * x), x**.5))
    return x, y

conditioned_model = pyro.condition(
    model,
    data={"X": X.squeeze(1), "Y":  Y.squeeze(1)}
)

N = X.shape[0]
pyro.set_rng_seed(526)
nuts_kernel = NUTS(conditioned_model, adapt_step_size=True)
mcmc = MCMC(nuts_kernel, num_samples=1500, warmup_steps=500)
mcmc.run(N)


# %% [markdown] id="zg_k9zEys8ll"
# ## リスト6.16

# %% id="OIA7Hj9Ws0Lt" colab={"base_uri": "https://localhost:8080/"} outputId="4fd48a03-15f3-4fba-f49c-0ecc161961dc"
from pyro.distributions.transforms import conditional_spline
print(conditional_spline(input_dim=1, context_dim=1))

# %% [markdown] id="Jc7fmqjNtE5o"
# ## リスト6.17

# %% id="WmyxmSXttHwH"
from pyro.distributions import TransformedDistribution
from pyro.distributions.transforms import AffineTransform


NxDist = Uniform(torch.zeros(1), torch.ones(1))
f_x = AffineTransform(loc=1., scale=100.0)
XDist = TransformedDistribution(NxDist, [f_x])


# %% [markdown] id="f0pbEK6HtKGN"
# ## リスト6.18

# %% id="eYtUXxg_tMgk"
import pyro
from pyro.distributions import (
    ConditionalTransformedDistribution,
    Normal, Uniform,
    TransformedDistribution
)
from pyro.distributions.transforms import (
    conditional_spline, spline
)
import torch
from torch.distributions.transforms import AffineTransform

pyro.set_rng_seed(348)

NxDist = Uniform(torch.zeros(1), torch.ones(1))
f_x = AffineTransform(loc=1., scale=100.0)
XDist = TransformedDistribution(NxDist, [f_x])

NyDist = Normal(torch.zeros(1), torch.ones(1))
f_y = conditional_spline(input_dim=1, context_dim=1)
YDist = ConditionalTransformedDistribution(NyDist, [f_y])

# %% [markdown] id="NDCiRo_QtQ21"
# ## リスト6.19

# %% id="dgWKSwAttRSY" colab={"base_uri": "https://localhost:8080/", "height": 490} outputId="4a32fb59-53ec-436f-c09d-0629b8c18f15"
import matplotlib.pyplot as plt

modules = torch.nn.ModuleList([f_y])
optimizer = torch.optim.Adam(modules.parameters(), lr=3e-3)
losses = []
maxY = max(Y)
Ynorm = Y / maxY
for step in range(800):
    optimizer.zero_grad()
    log_prob_x = XDist.log_prob(X)
    log_prob_y = YDist.condition(X).log_prob(Ynorm)
    loss = -(log_prob_x + log_prob_y).mean()
    loss.backward()
    optimizer.step()
    XDist.clear_cache()
    YDist.clear_cache()
    losses.append(loss.item())

plt.plot(losses[1:])
plt.title("Loss")
plt.xlabel("step")
plt.ylabel("loss")

# %% [markdown] id="MrKhD3SatXRE"
# ## リスト6.20

# %% id="dQ8bdAANtaUB" colab={"base_uri": "https://localhost:8080/", "height": 534} outputId="be3de9d4-4c97-48f0-c1d4-77874ad4521f"
x_flow = XDist.sample(torch.Size([100,]))
y_flow = YDist.condition(x_flow).sample(torch.Size([100,])) * maxY

plt.title("""
Observed values of enzyme concentration X\n
and protein concentration Y""")
plt.xlabel('X')
plt.ylabel('Y')
plt.xlim(0, 105)
plt.ylim(0, 120)
plt.scatter(
    X.squeeze(1), Y.squeeze(1), color='firebrick',
    label='Actual Data',
    alpha=0.5
)
plt.scatter(
    x_flow.squeeze(1), y_flow.squeeze(),
    label='Generated values from trained model',
    alpha=0.5
)
plt.legend()
plt.show()
