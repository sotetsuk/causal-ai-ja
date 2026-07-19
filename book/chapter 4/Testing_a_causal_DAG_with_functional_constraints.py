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
# # Chapter 4 - Testing a causal DAG with functional constraints
#
# The notebook is a code companion to chapter 4 of the book [Causal AI](https://www.manning.com/books/causal-ai) by [Robert Osazuwa Ness](https://www.linkedin.com/in/osazuwa/).
#
# <a href="https://colab.research.google.com/github/altdeep/causalML/blob/master/book/chapter%204/Testing_a_causal_DAG_with_functional_constraints.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# %% colab={"base_uri": "https://localhost:8080/"} id="xG2Uj-oSMPbP" outputId="85f32011-1009-482a-aefc-26d0632d607c"
# Using pgmpy 0.1.19 because versions 0.1.20 - 0.1.24 (current at time of writing) have a bug that prevents this code from running. An issue has been created on Github.
# For stability, I also use the version of pandas that was available when pgmpy 0.1.19 was available.
# !pip install pgmpy==0.1.19
# !pip install pandas==1.4.3

# %% [markdown] id="7lfhlmCEGA8y"
# In this tutorial, we illustrate how to falsify a causal DAG using a functional constraint called a Verma constraints. Verma constraints are like conditional independence constraints -- they are a type of contraint on the joint probability of the observed variables in your model that holds assuming your DAG is correct. But to test a conditional independence constraint, you need to observe all the variables involved in the constraint. Verma constraints are constraints you can test when there are latent variables.
#
# We look at the following model:
#
# ![verma_DAG](https://github.com/altdeep/causalML/blob/master/book/chapter%204/images/verma_DAG.png?raw=1)
#
# This DAG implies the following conditional independencies:
#
# * S ⊥ L | D, T
# * T ⊥ C | S
# * T ⊥ D | S
# * L ⊥ C | D, S
# * L ⊥ C | D, T
# * C ⊥ D
#
# These are six conditional independence constraints that we can test. But let's assume genetics (D) is unobserved. Then we can only test T ⊥ C | S. However, the following Verma constraint involving the observed variables is:
#
# $h(L, T, C) \perp C$ where $h(l, t, c) = \sum_{S=s} P(L=l|C=c, S=s, T=t) P(S=s|C=c)$
#
# So in this tutorial, for each row's values of l, s, t, c, we calculate h(L, T, C), then test this quantities independence relative to C.

# %% [markdown] id="1XS66gFuSQWe"
# First, I'll load the data discretize C to make things easier. With this discretization we will lose some bit of information, but again, the goal is to test our DAG, rather than achieving some ideal level of mathematical precision.

# %% colab={"base_uri": "https://localhost:8080/"} id="8qHOago0jluH" outputId="35c19b26-6d80-405e-b5e1-64e91c450acd"
from functools import partial
import numpy as np
import pandas as pd

data_url = "https://raw.githubusercontent.com/altdeep/causalML/master/datasets/cigs_and_cancer.csv"
data = pd.read_csv(data_url)    #A
cost_lower = np.quantile(data["C"], 1/3)    #B
cost_upper = np.quantile(data["C"], 2/3)    #B
def discretize_three(val, lower, upper):    #B
    if val < lower:    #B
        return "Low"    #B
    if val < upper:    #B
        return "Med"    #B
    return "High"    #B

data_disc = data.assign(    #B
    C = lambda df: df['C'].map(    #B
            partial(    #B
                discretize_three,    #B
                lower=cost_lower,    #B
                upper=cost_upper    #B
            )    #B
        )    #B
)    #B
data_disc = data_disc.assign(    #C
    L = lambda df: df['L'].map(str),    #C
)    #C
print(data_disc)
#A Load the csv into a Panda data frame.
#B Discretize cost (C) into a discrete variable with three levels to facilitate conditional impendence tests.
#C Turn lung cancer (L) from a Boolean to a string, so the conditional independence test will treat it as a discrete variable.

# %% [markdown] id="oszkvW4_GA8z"
# ## Listing 4.11 Fit naïve Bayes classifier of  for P(l| c, s, t)
#
# First, we use a naive Bayes classifier to model the distributions P(L=l|C=c, S=s, T=t) and P(S=s|C=c), the two components of the function g(l, c, s, t).

# %% id="21UjYSgSNuRJ"
from pgmpy.inference import VariableElimination
from pgmpy.models import NaiveBayes

model_L_given_CST = NaiveBayes()    #A
model_L_given_CST.fit(data_disc, 'L')    #A
infer_L_given_CST = VariableElimination(model_L_given_CST)    #A

def p_L_given_CST(L_val, C_val, S_val, T_val): #A
    result_out = infer_L_given_CST.query(    #A
        variables=["L"],    #A
        evidence={'C': C_val, 'S': S_val, 'T': T_val},    #A
        show_progress=False    #A
    )    #A
    var_outcomes = result_out.state_names["L"]    #A
    var_values = result_out.values    #A
    prob = dict(zip(var_outcomes, var_values))    #A
    return prob[L_val]    #A
#A We’ll use a naïve Bayes classifier in pgmpy to calculate the probability value for a given value of L given values of C, S, and T. In this case I use variable elimination.


# %% [markdown] id="LKB8ppj6GA80"
# ## Listing 4.12 Fit naïve Bayes classifier of P(s|c)
#
# Simulating results for L=True, C=Low, S=Low, and T=Low.

# %% colab={"base_uri": "https://localhost:8080/"} id="0Wb3rJJ_ZuyA" outputId="14df5936-ca84-4293-f935-543031f9013e"
p_L_given_CST("True", "Low", "Low", "Low")

# %% id="SMV0HWodspi9"
model_S_given_C = NaiveBayes()    #A
model_S_given_C.fit(data_disc, 'S')    #A
infer_S_given_C = VariableElimination(model_S_given_C)    #A
def p_S_given_C(S_val, C_val):    #A
    result_out = infer_S_given_C.query(    #A
        variables=['S'],    #A
        evidence={'C': C_val},    #A
        show_progress=False    #A
    )    #A
    var_names = result_out.state_names["S"]    #A
    var_values = result_out.values    #A
    prob = dict(zip(var_names, var_values))    #A
    return prob[S_val]    #A
#A Fit a naïve Bayes classifier of P(S|C)


# %% [markdown] id="lhIR7ebcIlEW"
# ## Listing 4.13 Combine models to create h(L, C, T)
#
# Next, I implement the h(.) function.

# %% id="hauWYTVIGF-j"
def h_function(L, C, T):    #A
    summ = 0    #B
    for s in ["Low", "Med", "High"]:    #B
        summ += p_L_given_CST(L, C, s, T) * p_S_given_C(s, C)    #B
    return summ
#A Implement h(L, C, T)
#B Implement summation of P(l|c,s,t) * P(s|c) over s


# %% [markdown] id="HZt_DioPGA80"
# ## Listing 4.14 Calculate outcome combinations of C, T, and L
#
# Next, we'll enumerate all combinations of C, T, and L so that we can calculate h(C, L, T) one each of these combinations.

# %% id="arB79aEcaJJx"
ctl_outcomes = pd.DataFrame(
    [    #A
        (C, T, L)    #A
        for C in ["Low", "Med", "High"]    #A
        for T in ["Low", "High"]    #A
        for L in ["False", "True"]    #A
    ],    #A
    columns = ['C', 'T', 'L']    #A
)
#A Now I'll calculate these values for each possible combination of outcomes of L, C, and T. First, I use list comprehensions to make a data frame containing all the combinations.

# %% id="SrL2meffYb_n" outputId="7af08082-a486-4f76-d2cf-6c9e8fae7cae" colab={"base_uri": "https://localhost:8080/"}
print(ctl_outcomes)

# %% [markdown] id="GqJpoh4NGA81"
# ## Listing 4.15 Calculate h(L, C, T) for each outcome of C, T, L
#
# Next for each combination of {L, C, T} in the dataset, we'll get values of h(L=l, C=c, T=t).

# %% colab={"base_uri": "https://localhost:8080/"} id="oSX3KkyArFdO" outputId="ee098d5b-742d-4cdc-bacb-d0f6967a227d"
h_dist = ctl_outcomes.assign(    #A
    h_func = ctl_outcomes.apply(    #A
        lambda row: h_function(    #A
            row['L'], row['C'], row['T']), axis = 1    #A
    )    #A
)    #A
print(h_dist)
# Calculate h(L, C, T) for each combination.

# %% [markdown] id="k6omw8nyGA81"
# Finally, we join the sum-product back on the data, so we have its value for each row of the data.

# %% colab={"base_uri": "https://localhost:8080/"} id="VvpWmC9hgWJF" outputId="ea7dc643-aaff-4f32-d0aa-abc1a50103b5"
df_mod = data_disc.merge(h_dist, on=['C', 'T', 'L'], how='left')    #A
print(df_mod)
# Add a column representing the variable h(C, T, L)

# %% [markdown] id="_9FbyToKGA81"
# To visualize independence of this quantity with C, we can use a boxplot.

# %% colab={"base_uri": "https://localhost:8080/", "height": 496} id="ehJpVrPrBTP9" outputId="6e09c2cd-2a65-4e75-9712-cef8dc300040"
df_mod.boxplot("h_func", "C")

# %% [markdown] id="F6PJtmrBGA82"
# The overlap of the boxes indicates independence.
#
# We can also use an [F-test](https://en.wikipedia.org/wiki/F-test) to test independence. The null hypothesis is independence. Let's assume a typical significance threshold of .1. A p-value below the threshold means one should reject the null hypothesis and favor the conclusion of dependence. That doesn't happen in this case.

# %% colab={"base_uri": "https://localhost:8080/"} id="5mnYJyQn6Kbi" outputId="afb5324a-b93b-450d-baea-69319e207cdf"
from statsmodels.formula.api import ols
import statsmodels.api as sm

model = ols('h_func ~ C', data=df_mod).fit()    #A
aov_table = sm.stats.anova_lm(model, typ=2)    #A
aov_table["PR(>F)"]["C"]    #A

# %% [markdown] id="9YexawRYGA82"
# As a sanity check, we can test independence with T and L. These have small p-values that fall below a .1 significance threshold.

# %% colab={"base_uri": "https://localhost:8080/"} id="kdd3jGMuAphc" outputId="434ce5e8-1496-468a-8b54-5b341bdc5b99"
model = ols('h_func ~ T', data=df_mod).fit()    #B
aov_table = sm.stats.anova_lm(model, typ=2)    #B
aov_table["PR(>F)"]["T"]    #B

# %% colab={"base_uri": "https://localhost:8080/"} id="BX1erMulA-Xw" outputId="0acd11e8-77de-4335-819b-b94399271129"
model = ols('h_func ~ L', data=df_mod).fit()    #C
aov_table = sm.stats.anova_lm(model, typ=2)    #C
aov_table["PR(>F)"]["L"]    #C
