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

# %% [markdown] id="i7zVXx0JkLWa"
# # 第7章

# %% colab={"base_uri": "https://localhost:8080/"} id="dHfRCUVXR1Vp" outputId="33fdc823-b2a8-49f9-9dc9-7b3fe74c9edb"
# !pip install pyro-ppl==1.9.0
# !pip install pandas==2.2.1
# !pip install pgmpy==0.1.25

# %% [markdown] id="VwYiFp-SkLWc"
# ## リスト7.1

# %% colab={"base_uri": "https://localhost:8080/", "height": 175} id="TP-bCjzR1TKj" outputId="4e3c73d1-f2a0-424a-d40f-5167a6cd1364"
import pandas as pd
data_url = (
    "https://raw.githubusercontent.com/altdeep/causalAI/master/"
    "datasets/sidequests_and_purchases_obs.csv"
)
df = pd.read_csv(data_url)
summary = df.drop('User ID', axis=1).groupby(
    ["Side-quest Engagement"]
).agg(
    ['count', 'mean', 'std']
)
summary

# %% [markdown] id="UrEdUX4xUxWe"
# ## リスト7.2

# %% colab={"base_uri": "https://localhost:8080/"} id="rkWUiKYuVD8V" outputId="a39f0f29-63f0-4eba-c15c-bba790190dc6"
n1, n2 = summary['In-game Purchases']['count']
m1, m2 = summary['In-game Purchases']['mean']
s1, s2 =  summary['In-game Purchases']['std']
pooled_std = (s1**2 / n1 + s2**2 / n2) **.5
z_score = (m1 - m2) / pooled_std
abs(z_score) > 2.

# %% [markdown] id="0aRmwqDKkLWd"
# ## リスト7.3

# %% colab={"base_uri": "https://localhost:8080/"} id="lrKQNEy58ZcG" outputId="f04de646-c32b-4dc3-aa77-6d7c3cb02649"
import pandas as pd
exp_data_url = (
    "https://raw.githubusercontent.com/altdeep/causalAI/master/"
    "datasets/sidequests_and_purchases_exp.csv"
)
df = pd.read_csv(exp_data_url)
summary = df.drop('User ID', axis=1).groupby(
    ["Side-quest Engagement"]
).agg(
    ['count', 'mean', 'std']
)
print(summary)

# %% [markdown] id="gyZ5vUxFsXLD"
# ## リスト7.4

# %% colab={"base_uri": "https://localhost:8080/"} id="ZXTl0x_YrsX5" outputId="ad010464-8b35-4962-f6e4-f2d1ebfb2788"
n1, n2 = summary['In-game Purchases']['count']
m1, m2 = summary['In-game Purchases']['mean']
s1, s2 =  summary['In-game Purchases']['std']
pooled_std = (s1**2 / n1 + s2**2 / n2) **.5
z_score = (m1 - m2) / pooled_std
abs(z_score) > 2.

# %% [markdown] id="4iZCNPqDNKwn"
# ## リスト7.5

# %% id="IyfgGcndVoht"
import pandas as pd
full_obs_url = (
    "https://raw.githubusercontent.com/altdeep/causalAI/master/"
    "datasets/sidequests_and_purchases_full_obs.csv"
)
df = pd.read_csv(full_obs_url)

# %% colab={"base_uri": "https://localhost:8080/"} id="6Gf-ONSkkLWe" outputId="a8ebc584-aa74-4824-d28d-8d99109d4812"
membership_counts = df['Guild Membership'].value_counts()
dist_guild_membership = membership_counts / sum(membership_counts)
print(dist_guild_membership)

# %% colab={"base_uri": "https://localhost:8080/"} id="8QhkMBTDkLWe" outputId="e3e040a1-90ef-42f4-ba54-737d2bc20842"
dist_guild_membership.member

# %% [markdown] id="2CuCrKoNNerw"
# ## リスト7.6

# %% colab={"base_uri": "https://localhost:8080/"} id="ySBVE0ts3nA8" outputId="0583830b-1678-4ebb-f470-d0fefd7a505d"
member_subset = df[(df['Guild Membership'] == 'member')]
member_engagement_counts = (
    member_subset['Side-quest Engagement'].value_counts()
)
dist_engagement_member = (
    member_engagement_counts / sum(member_engagement_counts)
)
print(dist_engagement_member)

nonmember_subset = df[(df['Guild Membership'] == 'nonmember')]
nonmember_engagement_counts = (
    nonmember_subset['Side-quest Engagement'].value_counts()
)
dist_engagement_nonmember = (
    nonmember_engagement_counts / sum(nonmember_engagement_counts)
)
print(dist_engagement_nonmember)

# %% [markdown] id="ngM9h303N85O"
# ## リスト7.7

# %% colab={"base_uri": "https://localhost:8080/"} id="yAMYV-oEW1zV" outputId="77006ee4-032f-4ef8-b6f3-eb0bea8f7259"
purchase_dist_nonmember_low_engagement = df[
    (df['Guild Membership'] == 'nonmember') &
    (df['Side-quest Engagement'] == 'low')
].drop(
  ['User ID', 'Side-quest Engagement', 'Guild Membership'], axis=1
).agg(['mean', 'std'])
print(round(purchase_dist_nonmember_low_engagement, 2))

purchase_dist_nonmember_high_engagement = df[
    (df['Guild Membership'] == 'nonmember') &
    (df['Side-quest Engagement'] == 'high')
].drop(
  ['User ID', 'Side-quest Engagement', 'Guild Membership'], axis=1
).agg(['mean', 'std'])
print(round(purchase_dist_nonmember_high_engagement, 2))

purchase_dist_member_low_engagement = df[
    (df['Guild Membership'] == 'member') &
    (df['Side-quest Engagement'] == 'low')
].drop(
  ['User ID', 'Side-quest Engagement', 'Guild Membership'], axis=1
).agg(['mean', 'std'])
print(round(purchase_dist_member_low_engagement, 2))

purchase_dist_member_high_engagement = df[
    (df['Guild Membership'] == 'member') &
    (df['Side-quest Engagement'] == 'high')
].drop(
  ['User ID', 'Side-quest Engagement', 'Guild Membership'], axis=1
).agg(['mean', 'std'])
print(round(purchase_dist_member_high_engagement, 2))

# %% [markdown] id="XgDpro60OoHl"
# ## リスト7.8

# %% id="jEHi29iNRx0F"
import pyro
from torch import tensor
from pyro.distributions import Bernoulli, Normal

def model():
    p_member = tensor(0.5)
    is_guild_member = pyro.sample(
        "Guild Membership",
        Bernoulli(p_member)
    )
    p_engaged = (tensor(0.8)*is_guild_member +
                 tensor(.2)*(1-is_guild_member))
    is_highly_engaged = pyro.sample(
        "Side-quest Engagement",
        Bernoulli(p_engaged)
    )
    get_purchase_param = lambda param1, param2, param3, param4: (
        param1 * (1-is_guild_member) * (1-is_highly_engaged) +
        param2 * (1-is_guild_member) * (is_highly_engaged) +
        param3 * (is_guild_member)   * (1-is_highly_engaged) +
        param4 * (is_guild_member)   * (is_highly_engaged)
    )
    μ = get_purchase_param(37.95, 54.92, 223.71, 125.50)
    σ = get_purchase_param(23.80, 4.92, 5.30, 53.49)
    in_game_purchases = pyro.sample(
        "In-game Purchases",
        Normal(μ, σ)
    )
    guild_membership = "member" if is_guild_member else "nonmember"
    engagement = "high" if is_highly_engaged else "low"
    in_game_purchases = float(in_game_purchases)
    return guild_membership, engagement, in_game_purchases

# %% id="f14wnt5tVzkG" colab={"base_uri": "https://localhost:8080/", "height": 272} outputId="529e3dc0-5b0b-4f7f-ee98-42df6643d56f"
pyro.render_model(model)

# %% id="qaEjxTxmr3DK" colab={"base_uri": "https://localhost:8080/", "height": 175} outputId="48b157c8-24f8-4c94-e3bd-57df12b7ce82"
pyro.util.set_rng_seed(123)
simulated_observational_data = [model() for _ in range(1000)]
sim_full_obs_df = pd.DataFrame(
    simulated_observational_data,
    columns=["Guild Membership", "Side-quest Engagement", "In-Game Purchases"]
)
sim_obs_df = sim_full_obs_df.drop("Guild Membership", axis=1)
sim_obs_df.groupby(["Side-quest Engagement"]).agg(['count', 'mean', 'std'])

# %% id="lphZ0nkGwL6T" colab={"base_uri": "https://localhost:8080/", "height": 175} outputId="28385ad8-8e9e-4b1f-cf17-d81e41ebe205"
df.drop(["Guild Membership", "User ID"], axis=1).groupby(["Side-quest Engagement"]).agg(['count', 'mean', 'std'])

# %% [markdown] id="7EsYyI796MDb"
# ## リスト7.9

# %% id="GKFV3wrF7N2y"
int_engaged_model = pyro.do(
    model,
    {"Side-quest Engagement": tensor(1.)}
)
int_unengaged_model = pyro.do(
    model,
    {"Side-quest Engagement": tensor(0.)}
)

# %% [markdown] id="-RiY7HfLT7A9"
# ## リスト7.10

# %% id="kWrPuirw1JIH" colab={"base_uri": "https://localhost:8080/"} outputId="22d7ca24-7b1e-4a7e-f6fb-76b9654e800a"
pyro.util.set_rng_seed(123)
simulated_experimental_data = [
    int_engaged_model() for _ in range(500)
] + [
    int_unengaged_model() for _ in range(500)
]
simulated_experimental_data = pd.DataFrame(
    simulated_experimental_data,
    columns=[
        "Guild Membership",
        "Side-quest Engagement",
        "In-Game Purchases"
    ]
)
sim_exp_df = simulated_experimental_data.drop(
    "Guild Membership", axis=1)
summary = sim_exp_df.groupby(
        ["Side-quest Engagement"]
    ).agg(
        ['count', 'mean', 'std']
    )
print(summary)

# %% id="gDqlZsK8BbkX" colab={"base_uri": "https://localhost:8080/"} outputId="d3f82d16-2147-4f0b-cdbd-266fcc7e5c91"
exp_data_url = (
    "https://raw.githubusercontent.com/altdeep/causalAI/master/"
    "datasets/sidequests_and_purchases_exp.csv"
)
exp_df = pd.read_csv(exp_data_url)

summary = exp_df.drop('User ID', axis=1).groupby(
        ["Side-quest Engagement"]
    ).agg(
        ['count', 'mean', 'std']
    )
print(summary)

# %% [markdown] id="RG3cnR7G6lUQ"
# ## リスト7.11

# %% id="2HoNLjrl6lpa"
from pgmpy.base import DAG

G = DAG([
    ('Guild Membership', 'Side-quest Engagement'),
    ('Side-quest Engagement', 'In-game Purchases'),
    ('Guild Membership', 'In-game Purchases')
])
G_int = G.do('Side-quest Engagement')

# %% [markdown] id="JXDgYGRb6mIS"
# ## リスト7.12

# %% id="60r5SUTj6mSI" colab={"base_uri": "https://localhost:8080/", "height": 795} outputId="4aa8bd44-8391-476f-fd12-1c31d3863855"
import pylab as plt
import networkx as nx

pos = {
    'Guild Membership': [0.0, 1.0],
    'Side-quest Engagement': [-1.0, 0.0],
    'In-game Purchases': [1.0, -0.5]
}

ax = plt.subplot()
ax.margins(0.3)
nx.draw(G, ax=ax, pos=pos, node_size=3000,
        node_color='w', with_labels=True)
plt.show()

ax = plt.subplot()
ax.margins(0.3)
nx.draw(G_int, ax=ax, pos=pos,
        node_size=3000, node_color='w', with_labels=True)
plt.show()

# %% [markdown] id="YZSf3ryd6mXm"
# ## リスト7.13

# %% id="7eGlZ4K_6mhS"
int_engaged_model = pyro.do(
    model,
    {"Side-quest Engagement": tensor(1.)}
)
int_unengaged_model = pyro.do(
    model,
    {"Side-quest Engagement": tensor(0.)}
)
