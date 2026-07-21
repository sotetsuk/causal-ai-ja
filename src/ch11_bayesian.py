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

# %% [markdown] id="1fVlLeJQPfAh"
# # 第11章

# %% colab={"base_uri": "https://localhost:8080/"} id="u5dqByst7jkT" outputId="c7e588c1-6dc9-48aa-d1f9-bf186bbb309f"
# !pip install pyro-ppl==1.9
# !pip install graphviz==0.20
# !pip install pandas==1.5.3
# !pip install torch==2.2.1+cu121

# %% [markdown] id="JthKrQTy-6Yo"
# ## リスト11.20

# %% id="OcbPTLUh6Vo1"
import pandas as pd
import torch

url = ("https://raw.githubusercontent.com/altdeep/"
       "causalAI/master/datasets/online_game_ate.csv")
df = pd.read_csv(url)
df = df[["Guild Membership", "Side-quest Engagement",
         "Won Items", "In-game Purchases"]]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
data = {
    col: torch.tensor(df[col].values, dtype=torch.float32).to(device)
    for col in df.columns
}

# %% [markdown] id="PFEBx7ZP_OMm"
# ## リスト11.21

# %% id="jCOOCnmx7w3M"
import torch.nn as nn

class Confounders2Engagement(nn.Module):
    def __init__(
        self,
        input_dim=1+1,
        hidden_dim=5
    ):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.f_engagement_ρ = nn.Linear(hidden_dim, 1)
        self.softplus = nn.Softplus()
        self.sigmoid = nn.Sigmoid()

    def forward(self, input):
        input = input.t()
        hidden = self.softplus(self.fc1(input))
        ρ_engagement = self.sigmoid(self.f_engagement_ρ(hidden))
        ρ_engagement = ρ_engagement.t().squeeze(0)
        return ρ_engagement


# %% [markdown] id="t9cFqzaV71Hs"
# ## リスト11.22

# %% id="_nHShtMB7zQw"
class PurchasesNetwork(nn.Module):
    def __init__(
        self,
        input_dim=1+1+1,
        hidden_dim=5
    ):
        super().__init__()
        self.f_hidden = nn.Linear(input_dim, hidden_dim)
        self.f_purchase_μ = nn.Linear(hidden_dim, 1)
        self.f_purchase_σ = nn.Linear(hidden_dim, 1)
        self.softplus = nn.Softplus()

    def forward(self, input):
        input = input.t()
        hidden = self.softplus(self.f_hidden(input))
        μ_purchases = self.f_purchase_μ(hidden)
        σ_purchases = 1e-6 + self.softplus(self.f_purchase_σ(hidden))
        μ_purchases = μ_purchases.t().squeeze(0)
        σ_purchases = σ_purchases.t().squeeze(0)
        return μ_purchases, σ_purchases


# %% [markdown] id="TLcNw0xT8Egv"
# ## リスト11.23

# %% id="-Q4aEu4v8Jti"
from pyro import sample
from pyro.distributions import Bernoulli, Normal
from torch import tensor, stack

def model(params, device=device):
    z_dist = Normal(
        tensor(0.0, device=device),
        tensor(1.0, device=device))
    z = sample("Z", z_dist)
    member_dist = Bernoulli(params['ρ_member'])
    is_guild_member = sample("Guild Membership", member_dist)
    engagement_input = stack((is_guild_member, z)).to(device)
    ρ_engagement = confounders_2_engagement(engagement_input)
    engage_dist = Bernoulli(ρ_engagement)
    is_highly_engaged = sample("Side-quest Engagement", engage_dist)
    p_won = (
        params['ρ_won_engaged'] * is_highly_engaged +
        params['ρ_won_not_engaged'] * (1 - is_highly_engaged)
    )
    won_items = sample("Won Items", Bernoulli(p_won))
    purchase_input = stack((won_items, is_guild_member, z)).to(device)
    μ_purchases, σ_purchases = purchases_network(purchase_input)
    purchase_dist = Normal(μ_purchases, σ_purchases)
    in_game_purchases = sample("In-game Purchases", purchase_dist)


# %% [markdown] id="P0kQ6pho8L8y"
# ## リスト11.24

# %% colab={"base_uri": "https://localhost:8080/", "height": 516} id="uFgQjHP28Na-" outputId="bc52d0df-9e47-4866-916b-e0f5c5a453f4"
import pyro
from pyro import render_model, plate
from pyro.distributions import Beta
from pyro import render_model

confounders_2_engagement = Confounders2Engagement().to(device)
purchases_network = PurchasesNetwork().to(device)
def data_model(data, device=device):
    pyro.module("confounder_2_engagement", confounders_2_engagement)
    pyro.module("confounder_2_purchases", purchases_network)
    two = tensor(2., device=device)
    five = tensor(5., device=device)
    params = {
        'ρ_member': sample('ρ_member', Beta(five, five)),
        'ρ_won_engaged': sample('ρ_won_engaged', Beta(five, two)),
        'ρ_won_not_engaged': sample('ρ_won_not_engaged', Beta(two, five)),
    }
    N = len(data["In-game Purchases"])
    with plate("N", N):
        model(params)

render_model(data_model, (data, ))


# %% [markdown] id="KYiPELdk8TJi"
# ## リスト11.25

# %% id="dVzgb8qs8WsC"
class Encoder(nn.Module):
    def __init__(self, input_dim=3,
                 z_dim=1,
                 hidden_dim=5):
        super().__init__()
        self.f_hidden = nn.Linear(input_dim, hidden_dim)
        self.f_loc = nn.Linear(hidden_dim, z_dim)
        self.f_scale = nn.Linear(hidden_dim, z_dim)
        self.softplus = nn.Softplus()

    def forward(self, input):
        input = input.t()
        hidden = self.softplus(self.f_hidden(input))
        z_loc = self.f_loc(hidden)
        z_scale = 1e-6 + self.softplus(self.f_scale(hidden))
        return z_loc.t().squeeze(0), z_scale.t().squeeze(0)


# %% [markdown] id="GfMf4iVx8geJ"
# ## リスト11.26

# %% id="K5ar8i3G8gmQ"
from pyro import param
from torch.distributions.constraints import positive

encoder = Encoder().to(device)

def guide(data, device=device):
    pyro.module("encoder", encoder)
    α_member = param("α_member", tensor(1.0, device=device),
                     constraint=positive)
    β_member = param("β_member", tensor(1.0, device=device),
                        constraint=positive)
    sample('ρ_member', Beta(α_member, β_member))
    α_won_engaged = param("α_won_engaged", tensor(5.0, device=device),
                         constraint=positive)
    β_won_engaged = param("β_won_engaged", tensor(2.0, device=device),
                        constraint=positive)
    sample('ρ_won_engaged', Beta(α_won_engaged, β_won_engaged))
    α_won_not_engaged = param("α_won_not_engaged",
                        tensor(2.0, device=device),
                        constraint=positive)
    β_won_not_engaged = param("β_won_not_engaged",
                        tensor(5.0, device=device),
                        constraint=positive)
    beta_dist = Beta(α_won_not_engaged, β_won_not_engaged)
    sample('ρ_won_not_engaged', beta_dist)
    N = len(data["In-game Purchases"])
    with pyro.plate("N", N):
        z_input = torch.stack(
            (data["Guild Membership"],
             data["Side-quest Engagement"],
             data["In-game Purchases"])
        ).to(device)
        z_loc, z_scale = encoder(z_input)
        pyro.sample("Z", Normal(z_loc, z_scale))


# %% [markdown] id="QSggSzQa99sn"
# ## リスト11.27

# %% colab={"base_uri": "https://localhost:8080/", "height": 1000} id="BpvmHY7x-Bwf" outputId="615605f6-c455-4538-d712-0e3064cc5b1f"
from pyro.infer import SVI, Trace_ELBO
from pyro.optim import Adam
from pyro import condition

pyro.clear_param_store()
adam_params = {"lr": 0.0001, "betas": (0.90, 0.999)}
optimizer = Adam(adam_params)
training_model = condition(data_model, data)
svi = SVI(training_model, guide, optimizer, loss=Trace_ELBO())
elbo_values = []
N = len(data['In-game Purchases'])
for step in range(500_000):
    loss = svi.step(data) / N
    elbo_values.append(loss)
    if step % 500 == 0:
        print(loss)

# %% [markdown] id="2faIpDk--GaW"
# ## リスト11.28

# %% colab={"base_uri": "https://localhost:8080/", "height": 472} id="0x1b1_pG-Gi3" outputId="c01f2669-6cdf-4cfc-90fd-a8aea5cd752e"
import math
import matplotlib.pyplot as plt

plt.plot([math.log(item) for item in elbo_values])
plt.xlabel('Step')
plt.ylabel('Log-Loss')
plt.title('Log Training Loss')
plt.show()


# %% [markdown] id="fuabVQHw-MkZ"
# ## リスト11.29

# %% colab={"base_uri": "https://localhost:8080/"} id="128tXbIp-Mq6" outputId="a95893bf-ae8a-4d9f-8473-d18329efac40"
print((
     pyro.param("α_member"),
     pyro.param("β_member"),
     pyro.param("α_won_engaged"),
     pyro.param("β_won_engaged"),
     pyro.param("α_won_not_engaged"),
     pyro.param("β_won_not_engaged")
))

# %% [markdown] id="eSU44Iv8-T1v"
# ## リスト11.30

# %% colab={"base_uri": "https://localhost:8080/", "height": 472} id="XlBYJwUu-T9O" outputId="559564c0-2649-4e3b-caad-a1fa3cbfa948"
import matplotlib.pyplot as plt
import seaborn as sns
from pyro.infer import Predictive

predictive = Predictive(data_model, guide=guide, num_samples=1000)
predictive_samples_all = predictive(data)
predictive_samples = predictive_samples_all["In-game Purchases"]
for i, sample_data in enumerate(predictive_samples):
    if i == 0:
        sns.kdeplot(sample_data,
            color="lightgrey", label="Predictive density")
    else:
        sns.kdeplot(sample_data,
            color="lightgrey", linewidth=0.2, alpha=0.5)

sns.kdeplot(
    data['In-game Purchases'],
    color="black",
    linewidth=1,
    label="Empirical density"
)

plt.legend()
plt.title("Posterior Predictive Check of In-game Purchases")
plt.xlabel("Value")
plt.ylabel("Density")
plt.show()


# %% [markdown] id="lh0emMuu-c0p"
# ## リスト11.31

# %% id="IHTSki4D-c-q"
from pyro.infer import Predictive
from pyro import do

data_model_low_engagement = do(
    data_model, {"Side-quest Engagement": 0.})
predictive_low_engagement = Predictive(
    data_model_low_engagement, guide=guide, num_samples=1000)
predictive_low_engagement_samples = predictive_low_engagement(data)

data_model_high_engagement = do(
    data_model, {"Side-quest Engagement": 1.})
predictive_high_engagement = Predictive(
    data_model_high_engagement, guide=guide, num_samples=1000)
predictive_high_engagement_samples = predictive_high_engagement(data)

# %% [markdown] id="bkp7Plf6-jNZ"
# ## リスト11.32

# %% colab={"base_uri": "https://localhost:8080/", "height": 474} id="DSFuEvvM-osm" outputId="20c49489-d75d-4095-d813-2eb792804132"
low_samples = predictive_low_engagement_samples["In-game Purchases"]
for i, sample_data in enumerate(low_samples):
    if i == 0:
        sns.kdeplot(sample_data,
            clip=(0, 35000), color="darkgrey", label="$P(I_{E=0})$")
    else:
        sns.kdeplot(sample_data,
            clip=(0, 35000), color="darkgrey",
            linewidth=0.2, alpha=0.5)

high_samples = predictive_high_engagement_samples["In-game Purchases"]
for i, sample_data in enumerate(high_samples):
    if i == 0:
        sns.kdeplot(sample_data,
            clip=(0, 35000), color="lightgrey", label="$P(I_{E=1})$")
    else:
        sns.kdeplot(sample_data,
            clip=(0, 35000), color="lightgrey",
            linewidth=0.2, alpha=0.5)
title = ("Posterior predictive sample density "
         "curves of $P(I_{E=1})$ & $P(I_{E=0})$")
plt.title(title)
plt.legend()
plt.xlabel("Value")
plt.ylabel("Density")
plt.ylim((0, .0010))
plt.xlim((0, 4000))
plt.show()

# %% [markdown] id="ISQymG7f-qWG"
# ## リスト11.33

# %% colab={"base_uri": "https://localhost:8080/", "height": 472} id="xrtvCLau-qgt" outputId="c4ca379d-6a85-42a8-cafe-20c73a4e31bc"
samp_high = predictive_high_engagement_samples['In-game Purchases']
exp_high = samp_high.mean(1)
samp_low = predictive_low_engagement_samples['In-game Purchases']
exp_low = samp_low.mean(1)
ate_distribution = exp_high - exp_low

sns.kdeplot(ate_distribution)
plt.title("Posterior distribution of the ATE")
plt.xlabel("Value")
plt.ylabel("Density")
plt.show()
