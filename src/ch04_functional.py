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

# %% [markdown] id="11lFMQh-GA8u"
# # 第4章

# %% colab={"base_uri": "https://localhost:8080/"} id="xG2Uj-oSMPbP" outputId="85f32011-1009-482a-aefc-26d0632d607c"
# !pip install pgmpy==0.1.19
# !pip install pandas==1.4.3

# %% [markdown]
# ## リスト4.10

# %% colab={"base_uri": "https://localhost:8080/"} id="8qHOago0jluH" outputId="35c19b26-6d80-405e-b5e1-64e91c450acd"
from functools import partial
import numpy as np
import pandas as pd

data_url = "https://raw.githubusercontent.com/altdeep/causalAI/master/datasets/cigs_and_cancer.csv"
data = pd.read_csv(data_url)
cost_lower = np.quantile(data["C"], 1/3)
cost_upper = np.quantile(data["C"], 2/3)
def discretize_three(val, lower, upper):
    if val < lower:
        return "Low"
    if val < upper:
        return "Med"
    return "High"

data_disc = data.assign(
    C = lambda df: df['C'].map(
            partial(
                discretize_three,
                lower=cost_lower,
                upper=cost_upper
            )
        )
)
data_disc = data_disc.assign(
    L = lambda df: df['L'].map(str),
)
print(data_disc)

# %% [markdown] id="oszkvW4_GA8z"
# ## リスト4.11

# %% id="21UjYSgSNuRJ"
from pgmpy.inference import VariableElimination
from pgmpy.models import NaiveBayes

model_L_given_CST = NaiveBayes()
model_L_given_CST.fit(data_disc, 'L')
infer_L_given_CST = VariableElimination(model_L_given_CST)

def p_L_given_CST(L_val, C_val, S_val, T_val):
    result_out = infer_L_given_CST.query(
        variables=["L"],
        evidence={'C': C_val, 'S': S_val, 'T': T_val},
        show_progress=False
    )
    var_outcomes = result_out.state_names["L"]
    var_values = result_out.values
    prob = dict(zip(var_outcomes, var_values))
    return prob[L_val]


# %% [markdown] id="LKB8ppj6GA80"
# ## リスト4.12

# %% colab={"base_uri": "https://localhost:8080/"} id="0Wb3rJJ_ZuyA" outputId="14df5936-ca84-4293-f935-543031f9013e"
p_L_given_CST("True", "Low", "Low", "Low")

# %% id="SMV0HWodspi9"
model_S_given_C = NaiveBayes()
model_S_given_C.fit(data_disc, 'S')
infer_S_given_C = VariableElimination(model_S_given_C)
def p_S_given_C(S_val, C_val):
    result_out = infer_S_given_C.query(
        variables=['S'],
        evidence={'C': C_val},
        show_progress=False
    )
    var_names = result_out.state_names["S"]
    var_values = result_out.values
    prob = dict(zip(var_names, var_values))
    return prob[S_val]


# %% [markdown] id="lhIR7ebcIlEW"
# ## リスト4.13

# %% id="hauWYTVIGF-j"
def h_function(L, C, T):
    summ = 0
    for s in ["Low", "Med", "High"]:
        summ += p_L_given_CST(L, C, s, T) * p_S_given_C(s, C)
    return summ


# %% [markdown] id="HZt_DioPGA80"
# ## リスト4.14

# %% id="arB79aEcaJJx"
ctl_outcomes = pd.DataFrame(
    [
        (C, T, L)
        for C in ["Low", "Med", "High"]
        for T in ["Low", "High"]
        for L in ["False", "True"]
    ],
    columns = ['C', 'T', 'L']
)

# %% id="SrL2meffYb_n" outputId="7af08082-a486-4f76-d2cf-6c9e8fae7cae" colab={"base_uri": "https://localhost:8080/"}
print(ctl_outcomes)

# %% [markdown] id="GqJpoh4NGA81"
# ## リスト4.15

# %% colab={"base_uri": "https://localhost:8080/"} id="oSX3KkyArFdO" outputId="ee098d5b-742d-4cdc-bacb-d0f6967a227d"
h_dist = ctl_outcomes.assign(
    h_func = ctl_outcomes.apply(
        lambda row: h_function(
            row['L'], row['C'], row['T']), axis = 1
    )
)
print(h_dist)

# %% [markdown]
# ## リスト4.16

# %% colab={"base_uri": "https://localhost:8080/"} id="VvpWmC9hgWJF" outputId="ea7dc643-aaff-4f32-d0aa-abc1a50103b5"
df_mod = data_disc.merge(h_dist, on=['C', 'T', 'L'], how='left')
print(df_mod)

# %% [markdown]
# ## リスト4.17

# %% colab={"base_uri": "https://localhost:8080/", "height": 496} id="ehJpVrPrBTP9" outputId="6e09c2cd-2a65-4e75-9712-cef8dc300040"
df_mod.boxplot("h_func", "C")

# %% [markdown]
# ## リスト4.18

# %% colab={"base_uri": "https://localhost:8080/"} id="5mnYJyQn6Kbi" outputId="afb5324a-b93b-450d-baea-69319e207cdf"
from statsmodels.formula.api import ols
import statsmodels.api as sm

model = ols('h_func ~ C', data=df_mod).fit()
aov_table = sm.stats.anova_lm(model, typ=2)
aov_table["PR(>F)"]["C"]

# %% colab={"base_uri": "https://localhost:8080/"} id="kdd3jGMuAphc" outputId="434ce5e8-1496-468a-8b54-5b341bdc5b99"
model = ols('h_func ~ T', data=df_mod).fit()
aov_table = sm.stats.anova_lm(model, typ=2)
aov_table["PR(>F)"]["T"]

# %% colab={"base_uri": "https://localhost:8080/"} id="BX1erMulA-Xw" outputId="0acd11e8-77de-4335-819b-b94399271129"
model = ols('h_func ~ L', data=df_mod).fit()
aov_table = sm.stats.anova_lm(model, typ=2)
aov_table["PR(>F)"]["L"]
