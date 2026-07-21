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

# %% [markdown] id="9jBwRpVVsmtu"
# # 第9章

# %% id="3avePuoHFg-8" outputId="f4e7c487-5bc0-4120-f352-7c4fba9217fb" colab={"base_uri": "https://localhost:8080/"}
# !pip install pgmpy==0.1.25
# !pip install graphviz==0.20.3
# !apt install libgraphviz-dev
# !pip install pygraphviz==1.13
# !pip install networkx==2.8.8
# !pip install torch==1.13.0
# !pip install pyro-ppl==1.1.0

# %% [markdown] id="7mE6UyH9URQm"
# ## リスト9.1

# %% colab={"base_uri": "https://localhost:8080/"} id="5_P8FI8fSP1k" outputId="57db44b8-9e4e-4ab5-819b-0688fb023010"
import graphviz
import networkx as nx
from networkx.drawing.nx_agraph import write_dot
def plot_graph(G):
    dot_format = nx.nx_pydot.to_pydot(G).to_string()
    return graphviz.Source(dot_format)

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

url_do = (
    "https://raw.githubusercontent.com/altdeep/"
    "causalAI/master/book/pgmpy_do.py"
)
code_do = download_code(url_do)

url_clone = (
    "https://raw.githubusercontent.com/altdeep/"
    "causalAI/master/book/chapter%209/hyp_function.py"
)
code_clone = download_code(url_clone)

print(code_do)
print(code_clone)
exec(code_do)
exec(code_clone)

# %% [markdown] id="ng0kbczNUYNv"
# ## リスト9.2

# %% id="KnaOGdNhSoRl"
from pgmpy.factors.discrete.CPD import TabularCPD

p_door_with_car = TabularCPD(
    variable='Car Door Die Roll',
    variable_card=3,
    values=[[1/3], [1/3], [1/3]],
    state_names={'Car Door Die Roll': ['1st', '2nd', '3rd']}
)

p_player_first_choice = TabularCPD(
    variable='1st Choice Die Roll',
    variable_card=3,
    values=[[1/3], [1/3], [1/3]],
    state_names={'1st Choice Die Roll': ['1st', '2nd', '3rd']}
)

p_coin_flip = TabularCPD(
    variable='Coin Flip',
    variable_card=2,
    values=[[.5], [.5]],
    state_names={'Coin Flip': ['tails', 'heads']}
)

# %% [markdown] id="diktnrtGUpCM"
# ## リスト9.3

# %% id="jttEi4zVTQtJ"
f_strategy = TabularCPD(
    variable='Strategy',
    variable_card=2,
    values=[[1, 0], [0, 1]],
    evidence=['Coin Flip'],
    evidence_card=[2],
    state_names={
        'Strategy': ['stay', 'switch'],
        'Coin Flip': ['tails', 'heads']}
)

# %% [markdown] id="ySM0gr5OUznD"
# ## リスト9.4

# %% id="gStcx0kPU0RI"
f_host_door_selection = TabularCPD(
    variable='Host Door Selection',
    variable_card=3,
    values=[
        [0,0,0,0,1,1,0,1,1,0,0,0,0,0,1,0,1,0],
        [1,0,1,0,0,0,1,0,0,0,0,1,0,0,0,1,0,1],
        [0,1,0,1,0,0,0,0,0,1,1,0,1,1,0,0,0,0]
    ],
    evidence=['Coin Flip',
              'Car Door Die Roll',
              '1st Choice Die Roll'],
    evidence_card=[2, 3, 3],
    state_names={
        'Host Door Selection':['1st', '2nd', '3rd'],
        'Coin Flip': ['tails', 'heads'],
        'Car Door Die Roll': ['1st', '2nd', '3rd'],
        '1st Choice Die Roll': ['1st', '2nd', '3rd']
    }
)

# %% [markdown] id="IlT-XebZU-no"
# ## リスト9.5

# %% id="19XfoiHyU-yl"
f_second_choice = TabularCPD(
    variable='2nd Choice',
    variable_card=3,
    values=[
        [1,0,0,1,0,0,1,0,0,0,0,0,0,0,1,0,1,0],
        [0,1,0,0,1,0,0,1,0,1,0,1,0,1,0,1,0,1],
        [0,0,1,0,0,1,0,0,1,0,1,0,1,0,0,0,0,0]
    ],
    evidence=['Strategy', 'Host Door Selection',
              '1st Choice Die Roll'],
    evidence_card=[2, 3, 3],
    state_names={
        '2nd Choice': ['1st', '2nd', '3rd'],
        'Strategy': ['stay', 'switch'],
        'Host Door Selection': ['1st', '2nd', '3rd'],
        '1st Choice Die Roll': ['1st', '2nd', '3rd']
    }
)

# %% [markdown] id="KOHBX8q3Y9CV"
# ## リスト9.6

# %% id="gfbqJ1ZLY9ZT"
f_win_or_lose = TabularCPD(
    variable='Win or Lose',
    variable_card=2,
    values=[
        [1,0,0,0,1,0,0,0,1],
        [0,1,1,1,0,1,1,1,0],
    ],
    evidence=['2nd Choice', 'Car Door Die Roll'],
    evidence_card=[3, 3],
    state_names={
        'Win or Lose': ['win', 'lose'],
        '2nd Choice': ['1st', '2nd', '3rd'],
        'Car Door Die Roll': ['1st', '2nd', '3rd']
    }
)

# %% [markdown] id="LM_KT3UGZM7F"
# ## リスト9.7

# %% id="_iC_rKM2ZN3w"
exogenous_vars = ["Car Door Die Roll",
                  "Coin Flip",
                  "1st Choice Die Roll"]
endogenous_vars = ["Host Door Selection",
                   "Strategy",
                   "2nd Choice", "Win or Lose"]

actual_world_edges = [
    ('Coin Flip', 'Host Door Selection'),
    ('Coin Flip', 'Strategy'),
    ('Car Door Die Roll', 'Host Door Selection'),
    ('1st Choice Die Roll', 'Host Door Selection'),
    ('1st Choice Die Roll', '2nd Choice'),
    ('Host Door Selection', '2nd Choice'),
    ('Strategy', '2nd Choice'),
    ('2nd Choice', 'Win or Lose'),
    ('Car Door Die Roll', 'Win or Lose')
]

possible_world_edges = [
    (a + " Hyp" if a in endogenous_vars else a,
     b + " Hyp" if b in endogenous_vars else b)
    for a, b in actual_world_edges
]

# %% [markdown] id="awiVlukEsp_P"
# ## リスト9.8

# %% id="8Fm96ZgKNAMk" outputId="d825eca7-2ead-46ed-8099-431d0e061c22" colab={"base_uri": "https://localhost:8080/", "height": 368}
from pgmpy.models import BayesianNetwork

twin_world_graph = BayesianNetwork(
    actual_world_edges +
    possible_world_edges
)

twin_world_graph.add_cpds(
    p_door_with_car,
    p_player_first_choice,
    p_coin_flip,
    f_strategy,
    f_host_door_selection,
    f_second_choice,
    f_win_or_lose,
    clone(f_strategy),
    clone(f_host_door_selection),
    clone(f_second_choice),
    clone(f_win_or_lose),
)

plot_graph(twin_world_graph)

# %% [markdown] id="FSpHBS4tZpVF"
# ## リスト9.9

# %% colab={"base_uri": "https://localhost:8080/"} id="mPaiAc5TZplK" outputId="3db62601-8074-4a0e-b557-80b08878ab7d"
from pgmpy.inference import VariableElimination
infer = VariableElimination(twin_world_graph)
strategy_outcome = infer.query(
    ['Win or Lose'],
    evidence={"Strategy": "switch"}
)
print(strategy_outcome)

# %% [markdown] id="Ygyd4c2nZ_S9"
# ## リスト9.10

# %% colab={"base_uri": "https://localhost:8080/"} id="xh7avBKqZ_dG" outputId="eccc1beb-7499-4c07-8c16-8c2f2aa52714"
cf_model = do(twin_world_graph, {'Strategy Hyp': 'switch'})
infer = VariableElimination(cf_model)

cf_dist1 = infer.query(
    ['Win or Lose Hyp'],
    evidence={'Strategy': 'stay', 'Win or Lose': 'lose'}
)
print(cf_dist1)

cf_dist2 = infer.query(
    ['Win or Lose Hyp'],
    evidence={'Win or Lose': 'lose'}
)
print(cf_dist2)


# %% [markdown] id="8tOKlSFoaRyl"
# ## リスト9.11

# %% id="X3cdv7fsaWe5"
from torch import tensor
from pyro.distributions import Bernoulli, Normal
from pyro import sample

from functools import partial
PseudoDelta = partial(Normal, scale=.01)

def f_sex(N_sex):
    return sample("sex", Bernoulli(N_sex))

def f_femur(sex, N_femur):
    if sex == tensor(1.0):
        μ = 43.7 + 2.3 * N_femur
    else:
        μ = 40.238 + 1.9 * N_femur
    return sample("femur", PseudoDelta(μ))

def f_height(femur, sex, N_height):
    if sex == tensor(1.0):
        μ = 61.41 + 2.21 * femur + 7.62 * N_height
    else:
        μ = 54.1 + 2.47 * femur + 7 * N_height
    return sample("height", PseudoDelta(μ))

def model(exogenous):
    N_sex = sample("N_sex", exogenous['N_sex'])
    N_femur = sample("N_femur", exogenous['N_femur'])
    N_height = sample("N_height", exogenous['N_height'])
    sex = f_sex(N_sex)
    femur = f_femur(sex, N_femur)
    height = f_height(femur, sex, N_height)
    return sex, femur, height

exogenous = {
    'N_sex': Bernoulli(.5),
    'N_femur': Normal(0., 1.),
    'N_height': Normal(0., 1.),
}

# %% [markdown] id="5suBXKqTamED"
# ## リスト9.12

# %% colab={"base_uri": "https://localhost:8080/", "height": 455} id="1vcxBbnVamPO" outputId="2844ee0a-9dcc-4696-d8a1-9a995a2c467b"
import matplotlib.pyplot as plt
import pyro

int_model = pyro.do(model, data={"femur": tensor(46.0)})
int_samples = []
for _ in range(10000):
    _, _, int_height = int_model(exogenous)
    int_samples.append(float(int_height))

plt.hist(
    int_samples,
    bins=20,
    alpha=0.5,
    label="Intervention Samples",
    density=True
)
plt.ylim(0., .35)
plt.legend()
plt.xlabel("Height")
plt.show()

# %% [markdown] id="hczkmOgQa_mj"
# ## リスト9.13

# %% id="9QET0nsAbG9-"
import torch.distributions.constraints as constraints
from pyro.primitives import param
from pyro.distributions import Delta

def guide(exogenous):
    p = param("p", tensor(.5),
              constraint=constraints.unit_interval)
    n_sex = sample("N_sex", Bernoulli(p))
    sex = sample("sex", Bernoulli(n_sex))
    n_femur_loc = param("n_femur_loc", tensor(0.0))
    n_femur_scale = param(
        "n_femur_scale",
        tensor(1.0),
        constraint=constraints.positive
    )
    femur_dist = Normal(n_femur_loc, n_femur_scale)
    n_femur = sample("N_femur", femur_dist)
    n_height_loc = param("n_height_loc", tensor(0.0))
    n_height_scale = param(
        "n_height_scale",
        tensor(1.0),
        constraint=constraints.positive
    )
    height_dist =  Normal(n_height_loc, n_height_scale)
    n_height = sample("N_height", height_dist)
    femur = sample("femur", Delta(n_femur))
    height = sample("height", Delta(n_height))


# %% [markdown] id="IFxcpRfsvX4i"
# ## リスト9.14

# %% id="rVRX1Q0Kvgs2"
conditioned_model = pyro.condition(
    model,
    data={"femur": tensor(44.0), "height": tensor(165.0)}
)

# %% [markdown] id="jt67uth0vtfC"
# ## リスト9.15

# %% colab={"base_uri": "https://localhost:8080/", "height": 490} id="cg3rzqLxv3Ro" outputId="0eba0f24-9aa5-4f56-c11a-73fa28946d8f"
from pyro.infer import SVI, Trace_ELBO
from pyro.optim import Adam

pyro.util.set_rng_seed(123)
pyro.clear_param_store()
svi = SVI(
          model=conditioned_model,
          guide=guide,
          optim=Adam({"lr": 0.003}),
          loss=Trace_ELBO()
)
losses = []
num_steps = 5000
for t in range(num_steps):
    losses.append(svi.step(exogenous))

plt.plot(losses)
plt.title("Loss During Training")
plt.xlabel("step")
plt.ylabel("loss")

# %% [markdown] id="zagdPm2DwGcu"
# ## リスト9.16

# %% id="Qgaa-yfvwWwP"
n_sex_p = param("p").item()
n_femur_loc = param("n_femur_loc").item()
n_femur_scale = param("n_femur_scale").item()
n_height_loc = param("n_height_loc").item()
n_height_scale = param("n_height_scale").item()

exogenous_posterior = {
    'N_sex': Bernoulli(n_sex_p),
    'N_femur': Normal(n_femur_loc, n_femur_scale),
    'N_height': Normal(n_height_loc, n_height_scale),
}

# %% [markdown] id="dZUkOTivwaFE"
# ## リスト9.17

# %% id="JNRm7SoFzUHN"
cf_samples = []
for _ in range(10000):
    _, _, cf_height = int_model(exogenous_posterior)
    cf_samples.append(float(cf_height))

# %% [markdown] id="Xx1XD1ZUzfOH"
# ## リスト9.18

# %% colab={"base_uri": "https://localhost:8080/", "height": 455} id="dooKckKIznA_" outputId="3bf0cf58-b69d-4c77-9bda-177ecf373afd"
plt.hist(
    int_samples,
    bins=20,
    alpha=0.5,
    label="Intervention Samples",
    density=True
)
plt.hist(
    cf_samples,
    bins=20,
    alpha=0.5,
    label="Counterfactual Samples",
    density=True
)
plt.ylim(0., .35)
plt.legend()
plt.xlabel("Height")
plt.show()

# %% [markdown] id="mToiyu9fzycv"
# ## リスト9.19

# %% colab={"base_uri": "https://localhost:8080/", "height": 446} id="RX_TLRgFz21_" outputId="bcdf363a-c814-4f93-a5ca-32e57ef0510f"
import torch
from matplotlib import pyplot as plt

import io
import urllib.request
import numpy as np
url = ('https://github.com/altdeep/causalAI/blob/master/'
       'book/chapter%209/sprites_example.npz?raw=true')
with urllib.request.urlopen(url) as response:
    data = response.read()
file = io.BytesIO(data)
npzfile = np.load(file)
img_dict = dict(npzfile)
img = torch.tensor(img_dict['image'].astype(np.float32) )
plt.imshow(img, cmap='Greys_r', interpolation='nearest')
plt.axis('off')
plt.title('original')
plt.show()
causal_factor = torch.from_numpy(img_dict['label']).unsqueeze(0)
print(causal_factor)

# %% [markdown] id="TiyK1XXrz8bG"
# ## リスト9.20

# %% id="ju9-kjpo0IzW"
import requests
import torch.nn as nn

CARDINALITY = [1, 3, 6, 40, 32, 32]

class EncoderCausalFactors(nn.Module):
    def __init__(self, image_dim, factor_dim):
        super(EncoderCausalFactors, self).__init__()
        self.image_dim = image_dim
        self.factor_dim = factor_dim
        hidden_dim = 1000
        self.fc1 = nn.Linear(image_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, factor_dim)
        self.softplus = nn.Softplus()
        self.sigmoid = nn.Sigmoid()

    def forward(self, img):
        img = img.reshape(-1, self.image_dim)
        hidden1 = self.softplus(self.fc1(img))
        hidden2 = self.softplus(self.fc2(hidden1))
        p_loc = self.sigmoid(self.fc3(hidden2))
        return p_loc

encoder_n_causal_factors = EncoderCausalFactors(
    image_dim=64*64,
    factor_dim=sum(CARDINALITY)
)


# %% [markdown] id="DW6k4iOUJytb"
# ## リスト9.21

# %% id="z9tzqLiJaxcI" outputId="e08c239e-8157-43b2-9d4f-4b187ecfad18" colab={"base_uri": "https://localhost:8080/"}
url = ('https://github.com/altdeep/causalAI/raw/master/'
       'book/chapter%209/sprites-model-encoder-causal-factors.pt')
response = requests.get(url)
response.raise_for_status()
with open('temp_weights.pt', 'wb') as f:
    f.write(response.content)
state_dict = torch.load(
    'temp_weights.pt',
    map_location=torch.device('cpu')
)
encoder_n_causal_factors.load_state_dict(state_dict)

# %% [markdown] id="uBM6ZPDc03Un"
# ## リスト9.22

# %% colab={"base_uri": "https://localhost:8080/"} id="u31ZhIpG03fE" outputId="f7a9d8c5-d863-409a-edf7-57ea3e91bd50"
from pyro import distributions as dist

def decode_one_hot(factor_encoded, cardinality=CARDINALITY):
    split = [
        torch.split(element, cardinality)
        for element in factor_encoded
    ]
    labels = [[int(torch.argmax(vec)) for vec in item]
              for item in split]
    return torch.tensor(labels)

def sample_one_hot(p_encoded, cardinality=CARDINALITY):
    split = [torch.split(element, cardinality)
             for element in p_encoded]
    sample_list = [
        [
            dist.OneHotCategorical(p_vec).sample()
            for p_vec in item
        ] for item in split
    ]
    sample = torch.stack([
        torch.cat(samples, -1)
        for samples in sample_list
    ])
    return sample

inferred_cause_p = encoder_n_causal_factors.forward(img)
sampled_factors = sample_one_hot(
    inferred_cause_p
)
print(decode_one_hot(sampled_factors))


# %% [markdown] id="vxuMCZhj1SoF"
# ## リスト9.23

# %% id="xR98jsmy1l0v"
class EncoderNImage(nn.Module):
    def __init__(self, image_dim, factor_dim, n_image_dim):
        super(EncoderNImage, self).__init__()
        self.image_dim = image_dim
        self.factor_dim = factor_dim
        self.n_image_dim = n_image_dim
        hidden_dim = 1000
        self.fc1 = nn.Linear(
            self.image_dim + self.factor_dim, hidden_dim
        )
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc31 = nn.Linear(hidden_dim, n_image_dim)
        self.fc32 = nn.Linear(hidden_dim, n_image_dim)
        self.softplus = nn.Softplus()

    def forward(self, img, factor):
        img = img.reshape(-1, self.image_dim)
        inputs = torch.cat((img, factor), -1)
        hidden1 = self.softplus(self.fc1(inputs))
        hidden2 = self.softplus(self.fc2(hidden1))
        n_image_loc = self.fc31(hidden2)
        n_image_scale = torch.exp(self.fc32(hidden2))
        return n_image_loc, n_image_scale

encoder_n_image = EncoderNImage(
    image_dim=64*64,
    factor_dim=sum(CARDINALITY),
    n_image_dim=50
)


# %% [markdown] id="Ox3DDPuxwDtD"
# ## リスト9.24

# %% id="Hla0-_9fwGFf"
def encode_one_hot(factor, cardinality=CARDINALITY):
    new_factor = []
    for i, factor_length in enumerate(cardinality):
        new_factor.append(
            torch.nn.functional.one_hot(
                factor[:,i].to(torch.int64), int(factor_length)
            )
        )
    new_factor = torch.cat(new_factor, -1)
    return new_factor.to(torch.float32)


# %% [markdown] id="qpiIKJ5kv3Fv"
# ## リスト9.25

# %% id="3ChjZ7rrZxai"
weight_url = ("https://github.com/altdeep/causalAI/raw/master/"
              "book/chapter%209/sprites-model-encoder-n-image.pt")
response = requests.get(weight_url)
response.raise_for_status()
with open('temp_weights.pt', 'wb') as f:
    f.write(response.content)
state_dict = torch.load(
    'temp_weights.pt',
    map_location=torch.device('cpu')
)
encoder_n_image.load_state_dict(state_dict)
n_image_loc, n_image_scale = encoder_n_image.forward(
    img,
    encode_one_hot(causal_factor)
)
n_image = torch.normal(n_image_loc, n_image_scale)


# %% [markdown] id="U0P-Sl0i2J-l"
# ## リスト9.26

# %% id="BwEE1fDT2S2M"
class Decoder(nn.Module):
    def __init__(self, image_dim, factor_dim, n_image_dim):
        super(Decoder, self).__init__()
        hidden_dim = 1000
        self.fc1 = nn.Linear(n_image_dim + factor_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, image_dim)
        self.softplus = nn.Softplus()
        self.sigmoid = nn.Sigmoid()

    def forward(self, n_image, factor):
        inputs = torch.cat((n_image, factor), -1)
        hidden1 = self.softplus(self.fc1(inputs))
        hidden2 = self.softplus(self.fc2(hidden1))
        hidden3 = self.softplus(self.fc3(hidden2))
        p_img = self.sigmoid(self.fc4(hidden3))
        return p_img

decoder = Decoder(
    image_dim=64*64,
    factor_dim=sum(CARDINALITY),
    n_image_dim=50
)

# %% [markdown] id="MtgsQlY40Tng"
# ## リスト9.27

# %% id="b28MJBTTaoyS" outputId="972fb0a7-6201-4249-83e6-57b5391f271e" colab={"base_uri": "https://localhost:8080/"}
dcdr_url = ("https://github.com/altdeep/causalAI/raw/master/"
       "book/chapter%209/sprites-model-decoder.pt")
response = requests.get(dcdr_url)
response.raise_for_status()
with open('temp_weights.pt', 'wb') as f:
    f.write(response.content)
state_dict = torch.load(
    'temp_weights.pt',
    map_location=torch.device('cpu')
)
decoder.load_state_dict(state_dict)


# %% [markdown] id="7fsuRWqV2bAl"
# ## リスト9.28

# %% id="l2D4nJkb2gl8"
def compare_reconstruction(original, generated):
    fig = plt.figure()
    ax0 = fig.add_subplot(121)
    plt.imshow(
        original.cpu().reshape(64, 64),
        cmap='Greys_r',
        interpolation='nearest'
    )
    plt.axis('off')
    plt.title('actual')
    ax1 = fig.add_subplot(122)
    plt.imshow(
        generated.reshape(64, 64),
        cmap='Greys_r', interpolation='nearest')
    plt.axis('off')
    plt.title('counterfactual')
    plt.show()


# %% [markdown] id="r1WhQ2aE2l1f"
# ## リスト9.29

# %% id="cCquV6uO2tQS"
def p_n_image(n_image_params):
    n_image_loc, n_image_scale, n_unif_upper = n_image_params
    n_image_norm = dist.Normal(
        n_image_loc, n_image_scale
    ).to_event(1).sample()
    n_image_unif = dist.Uniform(0, n_unif_upper).expand(
        torch.Size([1, 64*64])
    ).sample()
    n_image = n_image_norm, n_image_unif
    return n_image

def f_image(factor, n_image):
    n_image_norm, n_image_unif = n_image
    p_output = decoder.forward(
        n_image_norm,
        encode_one_hot(factor)
    )
    sim_img = (n_image_unif <= p_output).int()
    return sim_img


# %% [markdown] id="D6aewwPm23_n"
# ## リスト9.30

# %% id="Fr_v-ICe29Eu" colab={"base_uri": "https://localhost:8080/", "height": 284} outputId="775fa714-01ef-4880-b636-4c317a8927e6"
def abduct(img, factor, smoother=1e-3):
    n_image_loc, n_image_scale = encoder_n_image.forward(
        img, encode_one_hot(factor)
    )
    n_unif_upper = decoder.forward(
        n_image_loc,
        encode_one_hot(factor)
    )
    n_unif_upper = n_unif_upper * (1 - 2 * smoother) + smoother
    p_image_params = n_image_loc, n_image_scale, n_unif_upper
    return p_image_params

def do_action(factor, element=1, val=2):
    intervened_factor = factor.clone()
    intervened_factor[0][element] = val
    return intervened_factor

def predict(intervened_factor, n_image_params):
    n_image = p_n_image(n_image_params)
    sim_img = f_image(intervened_factor, n_image)
    return sim_img

def counterfactual(img, factor):
    p_image_params = abduct(img, factor)
    intervened_factor = do_action(factor)
    pred_recon = predict(intervened_factor, p_image_params)
    compare_reconstruction(img, pred_recon)

counterfactual(img, causal_factor)
