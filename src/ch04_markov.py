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

# %% id="xQs8VigU0TdV" colab={"base_uri": "https://localhost:8080/"} outputId="cbd3b401-f71f-4a27-f3e1-4ddeb6327e2b"
# !pip install pgmpy==0.1.24
# !pip install pandas==2.0.3
# !pip install networkx==3.3

# %% [markdown] id="cKRSpaC2PbGs"
# ## リスト4.1

# %% id="lKE0cLBEyQpc" colab={"base_uri": "https://localhost:8080/"} outputId="5cf9b7bf-d432-47ec-a97a-209e0a71973a"
from networkx import is_d_separator
from pgmpy.base import DAG
dag = DAG([
    ('I', 'U'),
    ('I', 'M'),
    ('M', 'U'),
    ('J', 'V'),
    ('J', 'M'),
    ('M', 'V')
])
print(is_d_separator(dag, {"U"}, {"V"}, {"M"}))
print(is_d_separator(dag, {"U"}, {"V"}, {"M", "I", "J"}))
print(is_d_separator(dag, {"U"}, {"V"}, {"M", "I"}))
print(is_d_separator(dag, {"U"}, {"V"}, {"M", "J"}))

# %% [markdown] id="83VxALQbYwQI"
# ## リスト4.2

# %% colab={"base_uri": "https://localhost:8080/"} id="u1J-kSqXYw1S" outputId="0a29778c-e9dd-49cb-e5ed-10348766e02d"
from pgmpy.base import DAG
dag = DAG([
    ('I', 'U'),
    ('I', 'M'),
    ('M', 'U'),
    ('J', 'V'),
    ('J', 'M'),
    ('M', 'V')
])
dag.get_independencies()

# %% [markdown] id="CNzAhjzYPbGz"
# ## リスト4.3

# %% colab={"base_uri": "https://localhost:8080/"} id="NdwrB2CPZblr" outputId="1152e3cf-7472-4b1c-adde-ba4ffd6bd4f3"
import pandas as pd
survey_url = "https://raw.githubusercontent.com/altdeep/causalAI/master/datasets/transportation_survey.csv"
fulldata = pd.read_csv(survey_url)

data = fulldata[0:30]
print(data[0:5])

# %% [markdown] id="gVi7OqdsatW-"
# ## リスト4.4

# %% id="me9JRP0hyK3o" outputId="5ba4f354-ce66-4ba7-d17a-227ff47af09d" colab={"base_uri": "https://localhost:8080/"}
from pgmpy.estimators.CITests import chi_square
significance = .05
result = chi_square(
    X="E", Y="T", Z=["O", "R"],
    data=data,
    boolean=False,
    significance_level=significance
)
print(result)

# %% [markdown] id="TG-cx5ZHbc_z"
# ## リスト4.5

# %% colab={"base_uri": "https://localhost:8080/"} id="0Db_lc3jbdJq" outputId="0ba8d7fd-ce8a-4ca9-86fc-c5f01a2c1bfd"
from pgmpy.estimators.CITests import chi_square
significance = .05
result = chi_square(
    X="E", Y="T", Z=["O", "R"],
    data=data,
    boolean=True,
    significance_level=significance
)
print(result)

# %% [markdown] id="WW5z6fHIcpg4"
# ## リスト4.6

# %% colab={"base_uri": "https://localhost:8080/"} id="RM_LX5D9cpqT" outputId="0b2d94a9-1121-4024-9ce3-6fd6dacd2c38"
from pprint import pprint

from pgmpy.base import DAG
from pgmpy.independencies import IndependenceAssertion

dag = DAG([
    ('A', 'E'),
    ('S', 'E'),
    ('E', 'O'),
    ('E', 'R'),
    ('O', 'T'),
    ('R', 'T')
])
dseps = dag.get_independencies()

def test_dsep(dsep):
    test_outputs = []
    for X in list(dsep.get_assertion()[0]):
        for Y in list(dsep.get_assertion()[1]):
            Z = list(dsep.get_assertion()[2])
            test_result = chi_square(
                X=X, Y=Y, Z=Z,
                data=data,
                boolean=True,
                significance_level=significance
            )
            assertion = IndependenceAssertion(X, Y, Z)
            test_outputs.append((assertion, test_result))
    return test_outputs

results = [test_dsep(dsep) for dsep in dseps.get_assertions()]
results = dict([item for sublist in results for item in sublist])
pprint(results)

# %% [markdown] id="G883fCKWc0Fz"
# ## リスト4.7

# %% colab={"base_uri": "https://localhost:8080/"} id="Wau1NCCqc0NM" outputId="2a202042-9e7e-4dad-8a10-e004ad3d16c5"
num_pass = sum(results.values())
num_dseps = len(dseps.independencies)
num_fail = num_dseps - num_pass
print(num_fail / num_dseps)

# %% [markdown] id="_jkvftCadATK"
# ## リスト4.8

# %% id="d3HeU4E0daVi"
from numpy import mean, quantile

def sample_p_val(data_size, data, alpha):
    bootstrap_data = data.sample(n=data_size, replace=True)
    result = chi_square(
        X="E", Y="T", Z=["O", "R"],
        data=bootstrap_data,
        boolean=False,
        significance_level = alpha
    )
    p_val = result[1]
    return p_val

def estimate_p_val(data_size, data=fulldata, boot_size=1000, α=.05):
    samples = [
        sample_p_val(data_size, data=fulldata, alpha=α)
        for _ in range(boot_size)
    ]
    positive_tests = [p_val > significance for p_val in samples]
    prob_conclude = mean(positive_tests)
    p_estimate = mean(samples)
    quantile_05, quantile_95 = quantile(samples, [.05, .95])
    lower_error = p_estimate - quantile_05
    higher_error = quantile_95 - p_estimate
    return p_estimate, lower_error, higher_error, prob_conclude

data_size = range(30, 1000, 20)
result = list(zip(*[estimate_p_val(size) for size in data_size]))

# %% [markdown] id="6reVSvFJJn0w"
# ## リスト4.9

# %% colab={"base_uri": "https://localhost:8080/", "height": 945} id="k3GgxR9GJpwI" outputId="8b5da849-2b76-4bd1-f983-35306937279d"
import numpy as np
import matplotlib.pyplot as plt

p_vals, lower_bars, higher_bars, probs_conclude_indep = result
plt.title('Data size vs. p-value (Ind. of E & T | O & R)')
plt.xlabel("Number of examples in data")
plt.ylabel("Expected p-value")
error_bars = np.array([lower_bars, higher_bars])
plt.errorbar(
    data_size,
    p_vals,
    yerr=error_bars,
    ecolor="grey",
    elinewidth=.5
)
plt.hlines(significance, 0, 1000, linestyles="dashed")
plt.show()
plt.title('Probability of favoring independence given data size')
plt.xlabel("Number of examples in data")
plt.ylabel("Probability of test favoring conditional independence")
plt.plot(data_size, probs_conclude_indep)
