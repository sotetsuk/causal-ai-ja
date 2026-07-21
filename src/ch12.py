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

# %% [markdown] id="39G_Dg8ot-f1"
# # 第12章

# %% colab={"base_uri": "https://localhost:8080/"} id="jGwC-t-VMtFX" outputId="b4882eab-49a1-4ba6-f323-d41997437c9f"
# !pip install pgmpy==0.1.24

# %% [markdown] id="KI-kM27lubKU"
# ## リスト12.1

# %% colab={"base_uri": "https://localhost:8080/"} id="BE1qfdF9Myyv" outputId="3d8d9b4c-1cbf-4618-9dc7-8e1053932f37"
# !pip install pgmpy==0.1.24
from pgmpy.models import BayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination
import numpy as np

model = BayesianNetwork([
    ('C', 'X'),
    ('C', 'Y'),
    ('X', 'Y'),
    ('Y', 'U')
])

# %% [markdown] id="7j9yl0HFvvr4"
# ## リスト12.2

# %% id="HieUnx4_M3tr"
cpd_c = TabularCPD(
    variable='C',
    variable_card=2,
    values=[[0.5], [0.5]],
    state_names={'C': ['bear', 'bull']}
)

cpd_x = TabularCPD(
    variable='X',
    variable_card=2,
    values=[[0.8, 0.2], [0.2, 0.8]],
    evidence=['C'],
    evidence_card=[2],
    state_names={'X': ['debt', 'equity'], 'C': ['bear', 'bull']}
)

cpd_y = TabularCPD(
    variable='Y',
    variable_card=2,
    values= [[0.3, 0.9, 0.7, 0.6], [0.7, 0.1, 0.3, 0.4]],
    evidence=['X', 'C'],
    evidence_card=[2, 2],
    state_names={
        'Y': ['failure', 'success'],
        'X': ['debt', 'equity'],
        'C': ['bear', 'bull']
    }
)


# %% [markdown] id="9I7WgYQwwauT"
# ## リスト12.3

# %% colab={"base_uri": "https://localhost:8080/"} id="HpMSnXYQwn_B" outputId="3e61b7f8-9229-4c6a-b6d4-ab622cdb08ed"
cpd_u = TabularCPD(
    variable='U',
    variable_card=2,
    values=[[1., 0.], [0., 1.]],
    evidence=['Y'],
    evidence_card=[2],
    state_names={'U': [-1000, 99000], 'Y': ['failure', 'success']}
)
print(cpd_u)
model.add_cpds(cpd_c, cpd_x, cpd_y, cpd_u)

# %% [markdown] id="ri0DPe0Hw0Gk"
# ## リスト12.4

# %% colab={"base_uri": "https://localhost:8080/"} id="NiiRCqb0w1C-" outputId="fe6c1c0a-7813-4e17-9979-ebf0c4b0d3a6"
import requests

url = "https://raw.githubusercontent.com/altdeep/causalAI/master/book/pgmpy_do.py"
response = requests.get(url)
content = response.text

print("Downloaded script content:\n")
print(content)
confirm = input("\nDo you want to execute this script? (yes/no): ")
if confirm.lower() == 'yes':
    exec(content)
else:
    print("Script execution cancelled.")


# %% [markdown] id="dee7MlWzxBOn"
# ## リスト12.5

# %% colab={"base_uri": "https://localhost:8080/"} id="Hoi3CL11xBUD" outputId="5a59f726-98f4-49ee-9c2d-5c18bacfe44b"
def get_expectation(marginal):
    u_values = marginal.state_names["U"]
    probs = marginal.values
    expectation = sum([x * p for x, p in zip(u_values, probs)])
    return expectation

infer = VariableElimination(model)
marginal_u_given_debt = infer.query(
    variables=['U'], evidence={'X': 'debt'})
marginal_u_given_equity = infer.query(
    variables=['U'], evidence={'X': 'equity'})
e_u_given_x_debt = get_expectation(marginal_u_given_debt)
e_u_given_x_equity = get_expectation(marginal_u_given_equity)
print("E(U(Y)|X=debt)=", e_u_given_x_debt)
print("E(U(Y)|X=equity)=", e_u_given_x_equity)

int_model_x_debt = do(model, {"X": "debt"})
infer_debt = VariableElimination(int_model_x_debt)
marginal_u_given_debt = infer_debt.query(variables=['U'])
expectation_u_given_debt = get_expectation(marginal_u_given_debt)
print("E(U(Y_{X=debt}))=", expectation_u_given_debt)
int_model_x_equity = do(model, {"X": "equity"})
infer_equity = VariableElimination(int_model_x_equity)
marginal_u_given_equity = infer_equity.query(variables=['U'])
expectation_u_given_equity = get_expectation(marginal_u_given_equity)
print("E(U(Y_{X=equity}))=", expectation_u_given_equity)


# %% [markdown] id="PI7DpjNxxcy9"
# ## リスト12.6

# %% id="WQ8JdAIwyos4"
model2 = BayesianNetwork([
    ('C', 'X'),
    ('C', 'Y'),
    ('X', 'Y'),
    ('Y', 'U')
])

cpd_y2 = TabularCPD(
    variable='Y',
    variable_card=2,
    values=[[0.3, 0.9, 0.7, 0.4],  [0.7, 0.1, 0.3, 0.6]],
    evidence=['X', 'C'],
    evidence_card=[2, 2],
    state_names={
        'Y': ['failure', 'success'],
        'X': ['debt', 'equity'],
        'C': ['bear', 'bull']
    }
)

model2.add_cpds(cpd_c, cpd_x, cpd_y2, cpd_u)

# %% [markdown] id="v_fLAfDuygbP"
# ## リスト12.7

# %% colab={"base_uri": "https://localhost:8080/"} id="sDk0B4b5y1zv" outputId="dcfcf5a2-fe71-4a4b-f7cb-a046e18c8d7a"
infer = VariableElimination(model2)
marginal_u_given_debt = infer.query(
    variables=['U'], evidence={'X': 'debt'})
marginal_u_given_equity = infer.query(
    variables=['U'], evidence={'X': 'equity'})
e_u_given_x_debt = get_expectation(marginal_u_given_debt)
e_u_given_x_equity = get_expectation(marginal_u_given_equity)
print("E(U(Y)|X=debt)=", e_u_given_x_debt)
print("E(U(Y)|X=equity)=", e_u_given_x_equity)

int_model_x_debt = do(model2, {"X": "debt"})
infer_debt = VariableElimination(int_model_x_debt)
marginal_u_given_debt = infer_debt.query(variables=['U'])
expectation_u_given_debt = get_expectation(marginal_u_given_debt)
print("E(U(Y_{X=debt}))=", expectation_u_given_debt)
int_model_x_equity = do(model2, {"X": "equity"})
infer_equity = VariableElimination(int_model_x_equity)
marginal_u_given_equity = infer_equity.query(variables=['U'])
expectation_u_given_equity = get_expectation(marginal_u_given_equity)
print("E(U(Y_{X=equity}))=", expectation_u_given_equity)


# %% [markdown] id="E2eYtFKmQCao"
# ## リスト12.8

# %% id="uPPJej3mELa0"
model = BayesianNetwork(
    [
        ('intent', 'AI prediction'),
        ('intent', 'choice'),
        ('AI prediction', 'box B'),
        ('choice', 'U'),
        ('box B', 'U'),
    ]
)

# %% [markdown] id="rhbXwXmQQRVV"
# ## リスト12.9

# %% colab={"base_uri": "https://localhost:8080/"} id="QlvR1LSNEfpG" outputId="da4d9b6f-9f9f-4337-9d8a-2bcebd834772"
cpd_intent = TabularCPD(
    'intent', 2, [[0.5], [.5]],
    state_names={'intent': ['B', 'both']}
)
print(cpd_intent)

cpd_choice = TabularCPD(
    'choice', 2, [[1, 0], [0, 1]],
    evidence=['intent'],
    evidence_card=[2],
    state_names={
        'choice': ['B', 'both'],
        'intent': ['B', 'both']
    }
)
print(cpd_choice)

# %% [markdown] id="CvLYtad7QjfS"
# ## リスト12.10

# %% colab={"base_uri": "https://localhost:8080/"} id="6_gaDsDCE8RI" outputId="e9198c45-9498-4c06-f70a-3967f0898014"
cpd_AI = TabularCPD(
    'AI prediction', 2, [[.95, 0.05], [.05, .95]],
    evidence=['intent'],
    evidence_card=[2],
    state_names={
        'AI prediction': ['B', 'both'],
        'intent': ['B', 'both']
    }
)
print(cpd_AI)

cpd_box_b_content = TabularCPD(
    'box B', 2, [[0, 1], [1, 0]],
    evidence=['AI prediction'],
    evidence_card=[2],
    state_names={
        'box B': [0, 1000000],
        'AI prediction': ['B', 'both']
    }
)
print(cpd_box_b_content)

# %% [markdown] id="RvD_qre3Q7-8"
# ## リスト12.11

# %% colab={"base_uri": "https://localhost:8080/"} id="qxIK0UgoEin9" outputId="031288c0-a205-4787-8998-e4810c22bc8f"
cpd_u = TabularCPD(
    'U', 4,
    [
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1],
    ],
    evidence=['box B', 'choice'],
    evidence_card=[2, 2],
    state_names={
        'U': [0, 1000, 1000000, 1001000],
        'box B': [0, 1000000],
        'choice': ['B', 'both']
    }
)
print(cpd_u)

model.add_cpds(cpd_intent, cpd_choice, cpd_AI)
model.add_cpds(cpd_box_b_content, cpd_u)

# %% [markdown] id="MEFAuKo7RMJb"
# ## リスト12.12

# %% colab={"base_uri": "https://localhost:8080/"} id="llhzdyMU2l6O" outputId="037bbafa-9b1d-416c-8c09-aa6012ae970a"
int_model_x_both = do(model, {"choice": "both"})
infer_both = VariableElimination(int_model_x_both)
marginal_u_given_both = infer_both.query(
    variables=['U'], evidence={'intent': 'both'})
expectation_u_given_both = get_expectation(marginal_u_given_both)
print("E(U(Y_{choice=both}|intent=both))=", expectation_u_given_both)
int_model_x_box_B = do(model, {"choice": "B"})
infer_box_B = VariableElimination(int_model_x_box_B)
marginal_u_given_box_B = infer_box_B.query(
    variables=['U'], evidence={'intent': 'both'})
expectation_u_given_box_B = get_expectation(marginal_u_given_box_B)
print("E(U(Y_{choice=box B}|intent=both))=", expectation_u_given_box_B)
int_model_x_both = do(model, {"choice": "both"})
infer_both = VariableElimination(int_model_x_both)
marginal_u_given_both = infer_both.query(
    variables=['U'], evidence={'intent': 'B'})
expectation_u_given_both = get_expectation(marginal_u_given_both)
print("E(U(Y_{choice=both}|intent=B))=", expectation_u_given_both)
int_model_x_box_B = do(model, {"choice": "B"})
infer_box_B = VariableElimination(int_model_x_box_B)
marginal_u_given_box_B = infer_box_B.query(
    variables=['U'], evidence={'intent': 'B'})
expectation_u_given_box_B = get_expectation(marginal_u_given_box_B)
print("E(U(Y_{choice=box B}|intent=B))=", expectation_u_given_box_B)
