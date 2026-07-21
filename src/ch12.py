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
# # Chapter 12 - Causal decisions and reinforcement learning
#
# The notebook is a code companion to chapter 12 of the book [Causal AI](https://www.manning.com/books/causal-ai) by [Robert Osazuwa Ness](https://www.linkedin.com/in/osazuwa/). This code is aligned with the code in the text.
#
# <a href="https://colab.research.google.com/github/altdeep/causalAI/blob/master/book/chapter%2012/chapter_12_causal_decision.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# %% colab={"base_uri": "https://localhost:8080/"} id="jGwC-t-VMtFX" outputId="b4882eab-49a1-4ba6-f323-d41997437c9f"
# !pip install pgmpy==0.1.24

# %% [markdown] id="KI-kM27lubKU"
# ## Listing 12.1 DAG for investment decision model
#
# A major source of confusion for causal decision modeling is the difference between actions and interventions. In many decision contexts, especially in reinforcement learning, the action is a thing that the agent does that changes their environment. Yet, the action is also a variable driven by the environment. We see this when we look again at the investment example:
#
# ![investment DAG](https://raw.githubusercontent.com/altdeep/causalAI/master/book/chapter%2012/images/investment.png)
#
# Most conventional approaches to decision-making, including in reinforcement learning, focus on maximizing E(U(Y)|X=x) rather than $E(U(Y_{X=x}))$.

# %% colab={"base_uri": "https://localhost:8080/"} id="BE1qfdF9Myyv" outputId="3d8d9b4c-1cbf-4618-9dc7-8e1053932f37"
# !pip install pgmpy==0.1.24
from pgmpy.models import BayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination
import numpy as np

model = BayesianNetwork([    #A
    ('C', 'X'),    #A
    ('C', 'Y'),    #A
    ('X', 'Y'),    #A
    ('Y', 'U')    #A
])    #A
#A Setup the DAG

# %% [markdown] id="7j9yl0HFvvr4"
# ## Listing 12.2 Create causal Markov kernels for C, X, and Y
#
# Next we build the causal Markov kernels for Economy (C), Debt v. Equity (X), and business success (Y).

# %% id="HieUnx4_M3tr"
cpd_c = TabularCPD(    #A
    variable='C',    #A
    variable_card=2,    #A
    values=[[0.5], [0.5]],    #A
    state_names={'C': ['bear', 'bull']}    #A
)    #A

cpd_x = TabularCPD(    #B
    variable='X',    #B
    variable_card=2,    #B
    values=[[0.8, 0.2], [0.2, 0.8]],    #B
    evidence=['C'],    #B
    evidence_card=[2],    #B
    state_names={'X': ['debt', 'equity'], 'C': ['bear', 'bull']}    #B
)    #B

cpd_y = TabularCPD(    #C
    variable='Y',    #C
    variable_card=2,    #C
    values= [[0.3, 0.9, 0.7, 0.6], [0.7, 0.1, 0.3, 0.4]],    #C
    evidence=['X', 'C'],    #C
    evidence_card=[2, 2],    #C
    state_names={    #C
        'Y': ['failure', 'success'],    #C
        'X': ['debt', 'equity'],    #C
        'C': ['bear', 'bull']    #C
    }    #C
)    #C
#A Setup causal Markov kernel for C (economy). It takes two values "bull" (good economic conditions) and "bear" bad economic conditions.
#B Setup causal Markov kernel for action X, either making a debt investment or equity investment. Historic analysis shows investors prefer equity investing in a bull market and debt investment in a bear market.
#C Setup causal Markov kernel for business outcome Y, either success or failure, depending on the type of investment provided (X), and the economy Y.


# %% [markdown] id="9I7WgYQwwauT"
# ## Listing 12.3 Implement the utility node and initialize the model
#
# Finally, we add the utility node U. We use probabilities of 1 and 0 to represent a deterministic function of Y. We end by adding all the kernels to the model.

# %% colab={"base_uri": "https://localhost:8080/"} id="HpMSnXYQwn_B" outputId="3e61b7f8-9229-4c6a-b6d4-ab622cdb08ed"
cpd_u = TabularCPD(    #A
    variable='U',    #A
    variable_card=2,    #A
    values=[[1., 0.], [0., 1.]],    #E
    evidence=['Y'],    #A
    evidence_card=[2],    #A
    state_names={'U': [-1000, 99000], 'Y': ['failure', 'success']}    #A
)    #A
print(cpd_u)    #A
model.add_cpds(cpd_c, cpd_x, cpd_y, cpd_u)
#A Setup the utility node.

# %% [markdown] id="ri0DPe0Hw0Gk"
# ## Listing 12.4 Download helper function for implementing an ideal intervention
#
# Before proceeding, download and load a helper function that implements an ideal intervention. To allay security concerns of directly executing download code, the logic displays the script and prompts you to confirm before executing.

# %% colab={"base_uri": "https://localhost:8080/"} id="NiiRCqb0w1C-" outputId="fe6c1c0a-7813-4e17-9979-ebf0c4b0d3a6"
import requests

url = "https://raw.githubusercontent.com/altdeep/causalAI/master/book/pgmpy_do.py"   #A
response = requests.get(url)   #A
content = response.text   #A

print("Downloaded script content:\n")    #B
print(content)    #B
confirm = input("\nDo you want to execute this script? (yes/no): ")    #B
if confirm.lower() == 'yes':    #B
    exec(content)    #B
else:    #B
    print("Script execution cancelled.")    #B
#A Load an implementation of an ideal intervention.
#B To allay security concerns, you can inspect the downloaded script and confirm before running.


# %% [markdown] id="dee7MlWzxBOn"
# ## Listing 12.5 Calculate E(U(Y)|X=x) and $E(U(Y_{X=x}))$
#
# By now in this book, you should not be surprised that $E(U(Y_{X=x}))$ is different from E(U(Y)|X=x). Let’s look at these values.

# %% colab={"base_uri": "https://localhost:8080/"} id="Hoi3CL11xBUD" outputId="5a59f726-98f4-49ee-9c2d-5c18bacfe44b"
def get_expectation(marginal):    #A
    u_values = marginal.state_names["U"]    #A
    probs = marginal.values    #A
    expectation = sum([x * p for x, p in zip(u_values, probs)])    #A
    return expectation    #A

infer = VariableElimination(model)    #B
marginal_u_given_debt = infer.query(
    variables=['U'], evidence={'X': 'debt'})    #B
marginal_u_given_equity = infer.query(
    variables=['U'], evidence={'X': 'equity'})    #B
e_u_given_x_debt = get_expectation(marginal_u_given_debt)    #B
e_u_given_x_equity = get_expectation(marginal_u_given_equity)    #B
print("E(U(Y)|X=debt)=", e_u_given_x_debt)    #B
print("E(U(Y)|X=equity)=", e_u_given_x_equity)    #B

int_model_x_debt = do(model, {"X": "debt"})    #C
infer_debt = VariableElimination(int_model_x_debt)    #C
marginal_u_given_debt = infer_debt.query(variables=['U'])    #C
expectation_u_given_debt = get_expectation(marginal_u_given_debt)    #C
print("E(U(Y_{X=debt}))=", expectation_u_given_debt)    #C
int_model_x_equity = do(model, {"X": "equity"})    #C
infer_equity = VariableElimination(int_model_x_equity)    #C
marginal_u_given_equity = infer_equity.query(variables=['U'])    #C
expectation_u_given_equity = get_expectation(marginal_u_given_equity)    #C
print("E(U(Y_{X=equity}))=", expectation_u_given_equity)    #C
#A A helper function for calculating the expected utility.
#B Set X by intervention to debt and equity and calculate the expectation of U under each intervention.
#C Condition on X = debt and X = equity and calculate the expectation of U.


# %% [markdown] id="PI7DpjNxxcy9"
# This gives us the following conditional expected utilities (I mark the highest with *):
#
# * $E(U(Y)|X=debt)$ = 57000.0 *
# *	$E(U(Y)|X=equity)$ = 37000.0
#
# And the following interventional expected utilities.
#
# *	$E(U(Y_{X=debt}))$ = 39000.0 *
# *	$E(U(Y_{X=equity}))$ = 34000.0
#
# So E(U(Y)|X=debt) is different from $E(U(Y_{X=\text{debt}}))$ and $E(U(Y)|X=equity)$ is different from $E(U(Y_{X=\text{equity}}))$. However, our goal is to optimize expected utility. And in this case, debt maximizes both $E(U(Y)|X=x)$ and $E(U(Y_{X=x}))$.
#
# $\text{argmax}_x E(U(Y_{X=x}))$
#
# = $\text{argmax}_x E(U(Y|X=x))$
#
# = "debt"
#
# If "debt" maximizes both queries, what is the point of causal decision theory? What does it matter if $E(U(Y)|X=x)$ and $E(U(Y_{X=x}))$ are different if the optimal action for both is the same?
#
# In decision problems, it is quite common that a causal formulation of the problem provides the same answer as more traditional non-causal formulations. This is especially true in higher dimensional problems common in reinforcement learning. Some observe this and wonder why the causal formulation is needed at all.
#
# ## Listing 12.6 Change a parameter in causal Markov kernel for Y
#
# To answer, watch what happens when we make a slight change to the parameters of Y in the model. Specifically, we'll change the parameter for P(Y=success|X=equity, C=bull) from .4 to .6. First, we'll rebuild the model with the parameter change.

# %% id="WQ8JdAIwyos4"
model2 = BayesianNetwork([    #A
    ('C', 'X'),    #A
    ('C', 'Y'),    #A
    ('X', 'Y'),    #A
    ('Y', 'U')    #A
])

cpd_y2 = TabularCPD(    #B
    variable='Y',
    variable_card=2,
    values=[[0.3, 0.9, 0.7, 0.4],  [0.7, 0.1, 0.3, 0.6]],    #C
    evidence=['X', 'C'],
    evidence_card=[2, 2],
    state_names={
        'Y': ['failure', 'success'],
        'X': ['debt', 'equity'],
        'C': ['bear', 'bull']
    }
)

model2.add_cpds(cpd_c, cpd_x, cpd_y2, cpd_u)    #D
#A Initialize a new model
#B Create a new conditional probability distribution for Y
#C Change the parameter P(Y=success|X=equity, C=bull) = 0.4 (the last parameter in the first list) to 0.6.
#D Add the causal Markov kernels to the model.

# %% [markdown] id="v_fLAfDuygbP"
# ## Listing 12.7 Compare outcomes with changed parameters
#
# Next, we rerun inference.

# %% colab={"base_uri": "https://localhost:8080/"} id="sDk0B4b5y1zv" outputId="dcfcf5a2-fe71-4a4b-f7cb-a046e18c8d7a"
infer = VariableElimination(model2)    #A
marginal_u_given_debt = infer.query(
    variables=['U'], evidence={'X': 'debt'})    #A
marginal_u_given_equity = infer.query(
    variables=['U'], evidence={'X': 'equity'})    #A
e_u_given_x_debt = get_expectation(marginal_u_given_debt)    #A
e_u_given_x_equity = get_expectation(marginal_u_given_equity)    #A
print("E(U(Y)|X=debt)=", e_u_given_x_debt)    #A
print("E(U(Y)|X=equity)=", e_u_given_x_equity)    #A

int_model_x_debt = do(model2, {"X": "debt"})    #B
infer_debt = VariableElimination(int_model_x_debt)    #B
marginal_u_given_debt = infer_debt.query(variables=['U'])    #B
expectation_u_given_debt = get_expectation(marginal_u_given_debt)    #B
print("E(U(Y_{X=debt}))=", expectation_u_given_debt)    #B
int_model_x_equity = do(model2, {"X": "equity"})    #B
infer_equity = VariableElimination(int_model_x_equity)    #B
marginal_u_given_equity = infer_equity.query(variables=['U'])    #B
expectation_u_given_equity = get_expectation(marginal_u_given_equity)    #B
print("E(U(Y_{X=equity}))=", expectation_u_given_equity)    #B
#A Set X by intervention to debt and equity and calculate the expectation of U under each intervention.
#B Condition on X = debt and X = equity and calculate the expectation of U.


# %% [markdown] id="Y0pBb58My9hO"
# This gives us the following conditional expectations (* indicates the optimal choice).
#
# * E(U(Y)|X=debt) = 57000.0 *
# * E(U(Y)|X=equity) = 53000.0
#
# And the following interventional expectations.
#
# *	$E(U(Y_{X=debt}))$ = 39000.0
# *	$E(U(Y_{X=equity}))$ = 44000.0 *
#
# With that slight change in a single parameter, “debt” is still the optimal value of x in  $E(U(Y)|X=x)$ but now “equity” is the optimal value of x in $E(U(Y_{X=x}))$. This is a case where the causal answer and the answer from conditioning on evidence are different. And since we are trying to answer a level 2 query, the causal approach is the right approach.
#
# This means that while simply optimizing a conditional expectation often gets you the right answer, you are vulnerable to getting the wrong answer in certain circumstances. Compare this to our discussion of semi-supervised learning in Chapter 4; often the unlabeled data can help with learning, but in specific causal circumstances, the unlabeled data adds no value. Similarly, in this case, there are specific causal scenarios where the causal formulation of the problem will provide a different and more correct result relative to the traditional non-causal formulation. Even the most popular decision-optimization algorithms, including the deep learning-based approaches used in deep reinforcement learning, can get the answer wrong if they fail to account for the causal structure of the problem.
#

# %% [markdown] id="W7OZKOV_KItL"
# # Newcomb's Paradox Example
#
# A famous thought experiment called Newcomb's paradox contrasts the causal approach to decision theory, maximizing utility under intervention, with the conventional approach of maximizing utility conditional on some action. First, I present an AI-inspired version of this thought experiment, and show how to approach it with a formal causal model.
#
# There are two boxes designated A and B. Box A always contains \$1,000. Box B contains either \$1,000,000 or \$0. The decision-making agent must choose between taking only box B or both boxes. The agent does not know what is in box B until they decide. Given this information, it is obvious the agent should take both boxes – choosing both yields either \$1,000 or \$1,001,000, while choosing only B yields either \$0 or \$1,000,000.
#
# Now, suppose there is an AI that can predict with high accuracy what choice the agent intends to make. If the AI predicts the agent intends to take both boxes, it will put no money in box B. If the AI is correct, the agent only gets \$1,000. If the AI predicts that the agent intends to take only box B, it will put \$1,000,000 in box B. If the AI predicts correctly, the agent gets the \$1,000,000 in box B but not the \$1,000 in box A. The agent does not know for sure what the AI predicted or what box B contains until they make their choice.
#
# ![Newcomb](https://raw.githubusercontent.com/altdeep/causalAI/master/book/chapter%2012/images/newcomb.png)
#
#
# The paradox arises as follows. If the agent goes with the conventional approach of maximizing expected utility conditional on its choice, the optimal choice is different. We can see that by enumerating the possible outcomes and their probabilities. Let's assume the AI's predictions are 95% accurate. If the agent chooses both boxes, there is a 95% chance the AI will have guessed the agent's choice and put no money in B, in which case the agent only gets the \$1,000. There is a 5% chance the algorithm will guess wrong, in which case it puts \$1,000,000 in box B, and the agent wins \$1,001,000. If the agent chooses only box B, there is a 95% chance the AI will have predicted the choice and placed \$1,000,000 in box B, given the agent \$1,000,000 in winnings. There is a 5% chance it will not, and the agent will take home nothing.
#
# In the traditional formulation of Newcomb's paradox, the assumption is that the agent using causal decision theory only attends to the consequences of their actions, i.e., they are reasoning on the following causal DAG.
#
# ![Newcomb naive DAG](https://raw.githubusercontent.com/altdeep/causalAI/master/book/chapter%2012/images/newcomb%20dag.png)
#
# But the true data generating process is better captured by the following DAG:
#
# ![Necomb complete DAG](https://raw.githubusercontent.com/altdeep/causalAI/master/book/chapter%2012/images/newcomb%20DAG%202.png)
#
#
#

# %% [markdown] id="E2eYtFKmQCao"
# ## Listing 12.8 Create the DAG
#
# First, we build the DAG.

# %% id="uPPJej3mELa0"
model = BayesianNetwork(    #A
    [    #A
        ('intent', 'AI prediction'),    #A
        ('intent', 'choice'),    #A
        ('AI prediction', 'box B'),    #A
        ('choice', 'U'),    #A
        ('box B', 'U'),    #A
    ]    #A
)    #A
#A Create the DAG

# %% [markdown] id="rhbXwXmQQRVV"
# ## Listing 12.9 Create causal Markov kernels for intent and choice
#
# Next, we create causal Markov kernels for intent and choice.

# %% colab={"base_uri": "https://localhost:8080/"} id="QlvR1LSNEfpG" outputId="da4d9b6f-9f9f-4337-9d8a-2bcebd834772"
cpd_intent = TabularCPD(    #A
    'intent', 2, [[0.5], [.5]],    #A
    state_names={'intent': ['B', 'both']}    #A
)    #A
print(cpd_intent)

cpd_choice = TabularCPD(    #B
    'choice', 2, [[1, 0], [0, 1]],    #B
    evidence=['intent'],     #B
    evidence_card=[2],    #B
    state_names={    #B
        'choice': ['B', 'both'],    #B
        'intent': ['B', 'both']    #B
    }    #B
)    #B
print(cpd_choice)
#A We assume a 50-50 chance the agent will prefer both boxes vs box B.
#B We assume the agent's choice is deterministically driven by their intent.

# %% [markdown] id="CvLYtad7QjfS"
# ## Listing 12.10 Create causal Markov kernels for AI prediction and box B content
#
# Similarly, we create the causal Markov kernels for the AI's decision and the content of box B.
#

# %% colab={"base_uri": "https://localhost:8080/"} id="6_gaDsDCE8RI" outputId="e9198c45-9498-4c06-f70a-3967f0898014"
cpd_AI = TabularCPD(    #A
    'AI prediction', 2, [[.95, 0.05], [.05, .95]],    #A
    evidence=['intent'],    #A
    evidence_card=[2],    #A
    state_names={    #A
        'AI prediction': ['B', 'both'],    #A
        'intent': ['B', 'both']    #A
    }    #A
)    #A
print(cpd_AI)

cpd_box_b_content = TabularCPD(    #B
    'box B', 2, [[0, 1], [1, 0]],    #B
    evidence=['AI prediction'],    #B
    evidence_card=[2],    #B
    state_names={    #B
        'box B': [0, 1000000],    #B
        'AI prediction': ['B', 'both']    #B
    }    #B
)    #B
print(cpd_box_b_content)
#A The AI's prediction is 95% accurate
#B Box B contents are set deterministically by the AI's prediction.

# %% [markdown] id="RvD_qre3Q7-8"
# ## Listing 12.11 Create utility kernel and build the model
#

# %% colab={"base_uri": "https://localhost:8080/"} id="qxIK0UgoEin9" outputId="031288c0-a205-4787-8998-e4810c22bc8f"
cpd_u = TabularCPD(    #A
    'U', 4,    #A
    [    #A
        [1, 0, 0, 0],    #A
        [0, 1, 0, 0],    #A
        [0, 0, 1, 0],    #A
        [0, 0, 0, 1],    #A
    ],    #A
    evidence=['box B', 'choice'],    #A
    evidence_card=[2, 2],    #A
    state_names={    #A
        'U': [0, 1000, 1000000, 1001000],    #A
        'box B': [0, 1000000],    #A
        'choice': ['B', 'both']    #A
    }    #A
)    #A
print(cpd_u)

model.add_cpds(cpd_intent, cpd_choice, cpd_AI)     #B
model.add_cpds(cpd_box_b_content, cpd_u)     #B
#A Setup the utility node.
#B Build the model.

# %% [markdown] id="MEFAuKo7RMJb"
# ## Listing 12.12 Infer optimal choice using intervention and conditioning on intent
#
# The choice of the agent can't cause the AI's prediction, because the prediction happens first. Thus, we assume the AI agent is inferring the agents intent, and thus the intent of the agent is the cause of the AI's prediction.
# The causal decision-making agent would prefer the latter graph because it is a better representation of the data generating process. The clever agent wouldn't focus on maximizing $E(U_{choice=x})$. The clever agent is aware of its own intention. Knowing that this intention is a cause of the contents of box B, it focuses on optimizing $E(U_{choice=x}|intent=i)$, where i is their original intention of which box to pick.
#
# $\text{argmax}_x E(U_{choice=x}|intent=i)$
#
# We'll assume the agent's initial intention is an impulse it cannot control. But while they can't control their initial intent, they can do some introspection and become aware of this intent. Further, we'll assume that upon doing so they have the ability to change its choice to something different from what it initially intended after the AI has made its prediction and set the contents of box B. Let's model this system in pgmpy and evaluate maximizing $E(U_{choice=x}|intent=i)$.

# %% colab={"base_uri": "https://localhost:8080/"} id="llhzdyMU2l6O" outputId="037bbafa-9b1d-416c-8c09-aa6012ae970a"
int_model_x_both = do(model, {"choice": "both"})    #A
infer_both = VariableElimination(int_model_x_both)    #A
marginal_u_given_both = infer_both.query(    #A
    variables=['U'], evidence={'intent': 'both'})    #A
expectation_u_given_both = get_expectation(marginal_u_given_both)    #A
print("E(U(Y_{choice=both}|intent=both))=", expectation_u_given_both)    #A
int_model_x_box_B = do(model, {"choice": "B"})    #B
infer_box_B = VariableElimination(int_model_x_box_B)    #B
marginal_u_given_box_B = infer_box_B.query(    #B
    variables=['U'], evidence={'intent': 'both'})    #B
expectation_u_given_box_B = get_expectation(marginal_u_given_box_B)    #B
print("E(U(Y_{choice=box B}|intent=both))=", expectation_u_given_box_B)    #B
int_model_x_both = do(model, {"choice": "both"})    #C
infer_both = VariableElimination(int_model_x_both)    #C
marginal_u_given_both = infer_both.query(    #C
    variables=['U'], evidence={'intent': 'B'})    #C
expectation_u_given_both = get_expectation(marginal_u_given_both)    #C
print("E(U(Y_{choice=both}|intent=B))=", expectation_u_given_both)    #C
int_model_x_box_B = do(model, {"choice": "B"})    #D
infer_box_B = VariableElimination(int_model_x_box_B)    #D
marginal_u_given_box_B = infer_box_B.query(    #D
    variables=['U'], evidence={'intent': 'B'})    #D
expectation_u_given_box_B = get_expectation(marginal_u_given_box_B)    #D
print("E(U(Y_{choice=box B}|intent=B))=", expectation_u_given_box_B)    #D
#A Infer E(U(Ychoice=both|intent=both))
#B Infer E(U(Ychoice=box B|intent=both))
#C Infer E(U(Ychoice=both|intent=B))
#D Infer E(U(Ychoice=box B|intent=B))

# %% [markdown] id="rsZm6_wy2m2U"
# This code produces the following results (* indicates the optimal choice for a given intent):
#
# *	$E(U(Y_{choice=both}|intent=both))$= 51000.0 *
# *	$E(U(Y_{choice=box B}|intent=both))$= 50000.0
# *	$E(U(Y_{choice=both}|intent=B))$= 951000.0 *
# *	$E(U(Y_{choice=box B}|intent=B))$= 950000.0
#
# When the agent's initial intention is to select both, the best choice is to select both. When the agent intends to choose only box B, the best choice is to ignore those intentions and choose both. Either way, the agent should choose both. Note that when the agent happens to initially intends to choose only box B, switching to both boxes gives them an expected utility of 951000.0, which is greater than the optimal choice utility of 950000 in the non-causal approach.
#
