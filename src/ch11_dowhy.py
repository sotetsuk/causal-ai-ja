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

# %% [markdown] id="5Ipq518geRcD"
# # 第11章

# %% colab={"base_uri": "https://localhost:8080/"} id="cvkoqysCeZpS" outputId="03ebfcca-cdad-4d72-b68b-5017dcbe6f69"
# !apt-get install graphviz libgraphviz-dev pkg-config
# !pip install pygraphviz==1.12
# !pip install dowhy==0.11
# !pip install econml==0.15

# %% [markdown] id="DCBawB0QUz-r"
# ## リスト11.1

# %% colab={"base_uri": "https://localhost:8080/", "height": 652} id="8ADuBXs015qN" outputId="f6466d2d-896f-4cff-9d72-046b6b27c9f8"
import pygraphviz as pgv
from IPython.display import Image

causal_graph = """
digraph {
    "Prior Experience" -> "Player Skill Level";
    "Prior Experience" -> "Time Spent Playing";
    "Time Spent Playing" -> "Player Skill Level";
    "Guild Membership" -> "Side-quest Engagement";
    "Guild Membership" -> "In-game Purchases";
    "Player Skill Level" -> "Side-quest Engagement";
    "Player Skill Level" -> "In-game Purchases";
    "Time Spent Playing" -> "Side-quest Engagement";
    "Time Spent Playing" -> "In-game Purchases";
    "Side-quest Group Assignment" -> "Side-quest Engagement";
    "Customization Level" -> "Side-quest Engagement";
    "Side-quest Engagement" -> "Won Items";
    "Won Items" -> "In-game Purchases";
    "Won Items" -> "Total Inventory";
    "In-game Purchases" -> "Total Inventory";
}
"""
G = pgv.AGraph(string=causal_graph)
G.draw('/tmp/causal_graph.png', prog='dot')
Image('/tmp/causal_graph.png')

# %% [markdown] id="KgEsn612b573"
# ## リスト11.2

# %% colab={"base_uri": "https://localhost:8080/"} id="RHvs5CLMIOsf" outputId="50b75f0e-0425-45d0-9005-23f2c38bb484"
import pandas as pd
data = pd.read_csv(
    "https://raw.githubusercontent.com/altdeep/causalAI/master/datasets/online_game_example_do_why.csv"
)
print(data.columns)

# %% [markdown] id="XyTgNvyIcQEe"
# ## リスト11.3

# %% colab={"base_uri": "https://localhost:8080/"} id="XxGQ9hbYzUfR" outputId="421cd43c-cc2d-48b6-8e21-8ed4960f9336"
from dowhy import CausalModel

model = CausalModel(
    data=data,
    treatment='Side-quest Engagement',
    outcome='In-game Purchases',
    graph=causal_graph
)

# %% [markdown] id="S-dLarPDiCBU"
# ## リスト11.4

# %% colab={"base_uri": "https://localhost:8080/"} id="u8TzWYPeqHxk" outputId="552cf1d1-ff77-438c-ade8-f131c86cea03"
identified_estimand = model.identify_effect()
print(identified_estimand)

# %% [markdown] id="1oPk684Fi433"
# ## リスト11.5・11.6

# %% colab={"base_uri": "https://localhost:8080/"} id="HPn-BWlOsotr" outputId="9f2651dc-453b-4562-e07c-4b3a14145163"
causal_estimate_reg = model.estimate_effect(
    identified_estimand,
    method_name="backdoor.linear_regression",
    confidence_intervals=True
)
print(causal_estimate_reg)

# %% [markdown] id="bP38voZAp0cn"
# ## リスト11.7

# %% id="eb_5Ny5Lp2XI" colab={"base_uri": "https://localhost:8080/"} outputId="b7a4c54e-247d-40de-ba32-eb691db1d26b"
causal_estimate_strat = model.estimate_effect(
    identified_estimand,
    method_name="backdoor.propensity_score_stratification",
    target_units="ate",
    confidence_intervals=True
)

print(causal_estimate_strat)

# %% [markdown] id="mcGNxs5hzaIx"
# ## リスト11.8

# %% id="t6gIZuKEvaoL" colab={"base_uri": "https://localhost:8080/"} outputId="824f608c-2f55-4a29-ee89-8c68cbb7d595"
causal_estimate_match = model.estimate_effect(
    identified_estimand,
    method_name="backdoor.propensity_score_matching",
    target_units="ate",
    confidence_intervals=True
)
print(causal_estimate_match)

# %% [markdown] id="pYFIkPMvvZ1l"
# ## リスト11.9

# %% colab={"base_uri": "https://localhost:8080/"} id="yVXvBf97vtj-" outputId="b9a21e2f-5fbf-45ee-ad51-c22dc6496c51"
causal_estimate_ipw = model.estimate_effect(
    identified_estimand,
    method_name="backdoor.propensity_score_weighting",
    target_units = "ate",
    method_params={"weighting_scheme":"ips_weight"},
    confidence_intervals=True
)
print(causal_estimate_ipw)

# %% [markdown] id="81JeFFJhqkoW"
# ## リスト11.10

# %% colab={"base_uri": "https://localhost:8080/"} id="BvtkvQVtqjoQ" outputId="52f678df-d215-426c-f68b-870829976f54"
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LassoCV
from sklearn.ensemble import GradientBoostingRegressor

gb_estimate = model.estimate_effect(
    identified_estimand,
    method_name="backdoor.econml.dml.DML",
    control_value = 0,
    treatment_value = 1,
    method_params={
        "init_params":{
            'model_y':GradientBoostingRegressor(),
            'model_t': GradientBoostingRegressor(),
            "model_final":LassoCV(fit_intercept=False),
            'featurizer':PolynomialFeatures(degree=1, include_bias=False)
        },
        "fit_params":{}
    }
)
print(gb_estimate)

# %% [markdown] id="SvBntf_L1T6x"
# ## リスト11.11

# %% colab={"base_uri": "https://localhost:8080/"} id="0hoVsyTm0aLK" outputId="a3987750-6789-49a9-b0f7-9735b6260ba1"
from sklearn.ensemble import RandomForestRegressor
metalearner_estimate = model.estimate_effect(
    identified_estimand,
    method_name="backdoor.econml.metalearners.TLearner",
    method_params={
        "init_params": {'models': RandomForestRegressor()},
        "fit_params": {}
    }
)

print(metalearner_estimate)

# %% [markdown] id="M-gGZxJ7UotG"
# ## リスト11.12

# %% colab={"base_uri": "https://localhost:8080/"} id="bqNQ3sDIU5Iu" outputId="ff51e7c6-7ace-4641-d1d7-f8836ffb6530"
causal_estimate_fd = model.estimate_effect(
    identified_estimand,
    method_name="frontdoor.two_stage_regression",
    target_units = "ate",
    method_params={"weighting_scheme": "ips_weight"},
    confidence_intervals=True
)
print(causal_estimate_fd)

# %% [markdown] id="tsLoiBD4vxuW"
# ## リスト11.13

# %% colab={"base_uri": "https://localhost:8080/"} id="FZD_S6u_v1Xl" outputId="cedcce8f-e67e-4e29-f113-3b8e3fb1bc0b"
causal_estimate_iv = model.estimate_effect(
    identified_estimand,
    method_name="iv.instrumental_variable",
    method_params = {
        "iv_instrument_name": "Side-quest Group Assignment"
    },
    confidence_intervals=True
)
print(causal_estimate_iv)

# %% [markdown] id="qtPvwg7sv4mo"
# ## リスト11.14

# %% id="yC9t3Va7v8yW"
# %%capture
causal_estimate_regdist = model.estimate_effect(
    identified_estimand,
    method_name="iv.regression_discontinuity",
    method_params={
        'rd_variable_name':'Customization Level',
        'rd_threshold_value':0.5,
        'rd_bandwidth': 0.15
    },
    confidence_intervals=True,
)

# %% colab={"base_uri": "https://localhost:8080/"} id="9Dw3cAq-8Rdi" outputId="d57101af-6f83-4385-f5da-b3d6bd971183"
print(causal_estimate_regdist)

# %% [markdown] id="WYi6yUov1M8q"
# ## リスト11.15

# %% colab={"base_uri": "https://localhost:8080/"} id="WL9wSZVjzS45" outputId="977e9f93-e283-4f57-dce4-81c31fc157b9"
identified_estimand.set_identifier_method("frontdoor")
res_subset = model.refute_estimate(
    identified_estimand,
    causal_estimate_fd,
    method_name="data_subset_refuter",
    subset_fraction=0.8,
    num_simulations=100
)
print(res_subset)

# %% [markdown] id="DAWEyzzM2H2B"
# ## リスト11.16

# %% id="4zyNBy4v2Bwo" colab={"base_uri": "https://localhost:8080/"} outputId="0c1f85e7-0190-4731-9ef3-f035809e1639"
identified_estimand.set_identifier_method("backdoor")

res_random = model.refute_estimate(
    identified_estimand,
    gb_estimate,
    method_name="random_common_cause",
    num_simulations=100,
    n_jobs=2
)
print(res_random)


# %% [markdown] id="_rstWy7V6upL"
# ## リスト11.17

# %% id="kqCyNguG2zCj" colab={"base_uri": "https://localhost:8080/"} outputId="ba703f77-369b-4d8d-e62a-4540a55ab965"
identified_estimand.set_identifier_method("backdoor")

res_placebo = model.refute_estimate(
    identified_estimand,
    causal_estimate_ipw,
    method_name="placebo_treatment_refuter",
    placebo_type="permute",
    num_simulations=100
)

print(res_placebo)

# %% [markdown] id="5PfiXOTF7IYU"
# ## リスト11.18

# %% id="Mms5qfR63SpC" colab={"base_uri": "https://localhost:8080/"} outputId="82a0ed7b-88a1-45d1-ba99-bdd24f01bf21"
import numpy as np

coefficients = np.array([100.0, 50.0])
bias = 50.0
def linear_gen(df):
    subset = df[['guild_membership','player_skill_level']]
    y_new = np.dot(subset.values, coefficients) + bias
    return y_new

identified_estimand.set_identifier_method("frontdoor")
ref = model.refute_estimate(
    identified_estimand,
    causal_estimate_fd,
    method_name="dummy_outcome_refuter",
    outcome_function=linear_gen
)

res_dummy_outcome = ref[0]
print(res_dummy_outcome)

# %% [markdown]
# > **訳者補足**: dowhy 0.11 ではこの反証は機能しません。`outcome_function` は dowhy の公式ドキュメントには登場するものの実装上、無視されています。


# %% [markdown] id="tmtk16EI4cGr"
# ## リスト11.19

# %% id="bqTWKg1f3ukC" colab={"base_uri": "https://localhost:8080/", "height": 612} outputId="e7f13298-a318-4d35-9915-0bcfcd7cd28e"
identified_estimand.set_identifier_method("backdoor")
res_unobserved = model.refute_estimate(
    identified_estimand,
    causal_estimate_strat,
    method_name="add_unobserved_common_cause"
)

print(res_unobserved)
