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

# %% [markdown] id="Th3tUn8N5n4q"
# # Chapter 3 - Building a Causal Graph
#
# The notebook is a code companion to chapter 3 of the book [Causal AI](https://www.manning.com/books/causal-ai) by [Robert Osazuwa Ness](https://www.linkedin.com/in/osazuwa/).
#
# <a href="https://colab.research.google.com/github/altdeep/causalML/blob/master/book/chapter%203/Chapter_3_Building_a_Causal_Graph.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# %% id="APJ2_62w5oLu" outputId="87f77a7b-f3cb-452e-a51e-dabceb73e0f6" colab={"base_uri": "https://localhost:8080/"}
# For stability of this notebook, I install the versions of the libraries that were current at the time of writing.
# !pip install pgmpy==0.1.24
# !pip install pyro-ppl==1.8.6
# !pip install graphviz==0.20.1
# !pip install dowhy==0.11.1

# %% [markdown] id="l_ioUCQs6zzQ"
# ## Listing 3.1: DAG Rock-throwing example
#
# In the rock-throwing DGP from Chapter 2. In that example, Jenny and Bryan are have a certain amount of desire or “inclination” to throw rocks at a windowpane that has a certain amount of strength. If either person's inclination to throw surpasses a threshold, they throw. The window breaks depending on if either or both of them throw and the strength of the window. As a Python function, the DGP is as follows:

# %% id="vurkTE0s5oUK"
def true_dgp(jenny_inclination, brian_inclination, window_strength):    #A
    jenny_throws_rock = jenny_inclination > 0.5    #B
    brian_throws_rock = brian_inclination > 0.5    #B
    if jenny_throws_rock and brian_throws_rock:    #C
        strength_of_impact = 0.8    #C
    elif jenny_throws_rock or brian_throws_rock:    #D
        strength_of_impact = 0.6    #D
    else:    #E
        strength_of_impact = 0.0    #E
    window_breaks = window_strength < strength_of_impact    #F
    return jenny_throws_rock, brian_throws_rock, window_breaks

#A Input variables are numbers between 0 and 1
#B Jenny and Brian throw the rock if so inclined
#C If both throw the rock the strength of impact is .8
#D If one of them throws the strength of impact is .6
#E If neither throws the strength of impact is 0
#F The window breaks if strength of imact is greater than window strength


# %% [markdown] id="t7ncjfFO4W-K"
# This data generating process implies the following DAG.
#
# ![Window breaking DGP](https://github.com/altdeep/causalML/blob/master/book/chapter%203/images/window_breaking_DAG.png?raw=1)
#

# %% [markdown] id="kXlxbId87B20"
# ## Listing 3.2: Building the transportation DAG in pgmpy
#
# We can build the causal DAG with the following code.

# %% id="7Xvl8R9j7CBk"
from pgmpy.models import BayesianNetwork
model = BayesianNetwork(    #A
       [
        ('A', 'E'),    #B
        ('S', 'E'),    #B
        ('E', 'O'),    #B
        ('E', 'R'),    #B
        ('O', 'T'),    #B
        ('R', 'T')     #B
     ]
)
#A pgmpy provides a BayesianNetwork class where we add the edges to the model.
#B Inut the DAG as a list of edges (tuples).

# %% [markdown] id="c101BY4g4W-L"
# ![transportation DAG](https://github.com/altdeep/causalML/blob/master/book/chapter%203/images/transportation_DAG.png?raw=1)

# %% [markdown] id="FFvQMs-f7GTa"
# ## Listing 3.3: Loading transportation data
#
# The variables in the transportation data are all categorical variables. In this simple categorical case, we can rely on a graphical modeling library like pgmpy.
#

# %% colab={"base_uri": "https://localhost:8080/", "height": 424} id="amScTPVM7Gl9" outputId="9b79f394-80ba-492b-968d-b935454e6377"
import pandas as pd
url='https://raw.githubusercontent.com/altdeep/causalML/master/datasets/transportation_survey.csv'    #A
data = pd.read_csv(url)
data

# %% [markdown] id="bvr9LhbW7SEd"
# ## Listing 3.4: Learning parameters for the causal Markov kernels in the transportation model
#
# The `BayesianNetwork` class we initialized has a fit method that will learn the parameters of our causal Markov kernels. Since our variables are categorical, our causal Markov kernels will be in the form of conditional probability tables represented by pgmpy's TabularCPD class. The fit method will fit (“learn”) estimates of the parameters of those conditional probability tables using the data.
#

# %% colab={"base_uri": "https://localhost:8080/"} id="9rzqtLUm7Tib" outputId="aa6b4389-21d3-4d1b-b6d1-c62d852c323e"
from pgmpy.models import BayesianNetwork
model = BayesianNetwork(
      [
        ('A', 'E'),
        ('S', 'E'),
        ('E', 'O'),
        ('E', 'R'),
        ('O', 'T'),
        ('R', 'T')
     ]
)

# The fit method on the BayesianNetwork object will estimate parameters from data (a pandas DataFrame).
model.fit(data)

# Retrieve and view the causal Markov kernels learned by fit.
causal_markov_kernels = model.get_cpds()
print(causal_markov_kernels)

# %% [markdown] id="26h6WRcB4W-M"
# Let's look at the structure of the causal Markov kernel for the transportation variable T and gender S. Again, these are conditional probability tables.

# %% id="1WjTCbYc4W-N" outputId="727b10e1-ae9e-4653-f778-83790ba5ad2f" colab={"base_uri": "https://localhost:8080/"}
cmk_T = causal_markov_kernels[-1]
print(cmk_T)

print(causal_markov_kernels[2])

# %% [markdown] id="OKkTwKKF7ach"
# ## Listing 3.5: Bayesian point estimation with a Dirichlet conjugate prior
#
# We can also use a Bayesian technique called conjugate priors to estimate the parameters. Note, how we estimate the parameters of the causal Markov kernel is a statistical and computational question that arises after we select the DAG.

# %% colab={"base_uri": "https://localhost:8080/"} id="7do-dizG7p0-" outputId="1007d7c4-7048-47f7-be04-15ed495b82fb"
from pgmpy.estimators import BayesianEstimator    #A
estimator = BayesianEstimator(model, data)    #A
model.fit(
    data,
    estimator=BayesianEstimator,    #B
    prior_type="dirichlet",
    pseudo_counts=1    #C
)
causal_markov_kernels = model.get_cpds()     #D
cmk_T = causal_markov_kernels[-1]    #D
print(cmk_T)    #D
#A Import BayesianEstimator and initialize it on the model and data.
#B Pass the estimator object to the fit method.
#C pseudo_counts refers to the parameters of the Dirichlet prior.
#D Extract the causal Markov kernels (the last element on the list) and view P(T|O,R).


# %% [markdown] id="TBwJbZvv7xym"
# ## Listing 3.6: Training a causal graphical model with a latent variable.
#
# This is a causal model, but it is also a probabilistic graphical model. Thus we can use techniques in probabilistic graphical modeling, such as inferring parameters when some variables in the DAG are latent (unobserved in the data).
#
# To illustrate, suppose the education variable "E" in the transformation survey data were not recorded. pgmpy gives us a utility for learning the parameters of the causal Markov kernel for latent E using an algorithm called structural expectation maximization, which is a variant of parameter learning with maximum likelihood.

# %% colab={"base_uri": "https://localhost:8080/", "height": 531, "referenced_widgets": ["a3864206db4b4ff3ada29549349e82b9", "9d98fa142ecf44e283095ef1e8fe638f", "c286ca4e47bc40c2951389b2fc8313d3", "1d0a9c732e3749528c3c587bb7da39d4", "fce395ebd7224e2e9afd8d6f78346e42", "56dd7e0ea9724fac8d62956e999e5d71", "87dd0843125449c9ab8ad86021036dfa", "0f15f3fe25e34ee290142bcf87336046", "9cc4e346eb2a4d2eb87fc59338724932", "f2dfdd5542904a5992129577f260cc40", "d8cdf919c24c44a7bafac38ed0fc6ae2"]} id="VDWOW-ll7yED" outputId="8ac33b03-b9f8-431d-e174-8a30e2d04b3d"
import pandas as pd
from pgmpy.models import BayesianNetwork
from pgmpy.estimators import ExpectationMaximization as EM
url='https://raw.githubusercontent.com/altdeep/causalML/master/datasets/transportation_survey.csv'    #A
data = pd.read_csv(url)    #A
data_sans_E = data[['A', 'S', 'O', 'R', 'T']]    #B
model_with_latent = BayesianNetwork(
       [
        ('A', 'E'),
        ('S', 'E'),
        ('E', 'O'),
        ('E', 'R'),
        ('O', 'T'),
        ('R', 'T')
     ],
     latents={"E"}    #C
)
estimator = EM(model_with_latent, data_sans_E)    #D
cmks_with_latent = estimator.get_parameters(latent_card={'E': 2})    #D
print(cmks_with_latent[1].to_factor())    #E
#A Download the data and convert to a pandas data frame.
#B Keep all the columns except education E.
#C Indicate which variables are latent when training the model.
#D Run the structural expectation maximization algorithm to learn the causal Markov kernel for E.  You have to indicated the cardinality of the latent variable.
#E Print out the learned causal Markov kernel for E.  Print it as a factor object for legibility.


# %% [markdown] id="pps5wZki4W-N"
# One has to be careful about making causal inferences when some variables are unobserved in the data (e.g., "latent confounders"). We address this later chapters.

# %% [markdown] id="TT4kdobC749Y"
# ## Listing 3.7: Inference on the trained causal graphical model
#
# The benefit of implementing a probabilistic graphical model (PGM) on the causal DAG is that we can use the inference tools in PGM libraries. For example, we can use the inference algorithms such as [variable elimination](https://en.wikipedia.org/wiki/Variable_elimination) to infer any conditional and marginal probability of interest. These algorithms more or less just "work" without our having to understand their mechanics. This contrasts with more free-form probabilistic ML libraries such as Pyro, where we typically have to write the inference procedure explicitly.

# %% colab={"base_uri": "https://localhost:8080/"} id="STcTW75t75FM" outputId="530acc1a-3adb-4b15-843a-26657ae46545"
from pgmpy.inference import VariableElimination     #A
inference = VariableElimination(model)
query1 = inference.query(['E'], evidence={"T": "train"})
query2 = inference.query(['E'], evidence={"T": "car"})
print("train")
print(query1)
print("car")
print(query2)
#A VariableElimination is an inference algorithm specific to graphical models.


# %% [markdown] id="cyz8hkgn8An5"
# ## Listing 3.8: Implementing the trained causal model in Pyro
#
# In a tool like pyro, you have to be a bit more hands-on with the inference algorithm. The following illustrates the inference of P(E|T=”train”) using a probabilistic inference algorithm called importance sampling. First, we’ll implement the model in Pyro. Rather than fit the parameters, we’ll explicitly specify the parameter values we fit with pgmpy.
#

# %% colab={"base_uri": "https://localhost:8080/", "height": 368} id="dkyLvwtj8Avu" outputId="5e23b1cb-cd69-4c63-b197-6bcf4cf9082e"
import torch
import pyro
from pyro.distributions import Categorical

A_alias = ['young', 'adult', 'old']    #B
S_alias = ['M', 'F']    #B
E_alias = ['high', 'uni']    #B
O_alias = ['emp', 'self']    #B
R_alias = ['small', 'big']    #B
T_alias = ['car', 'train', 'other']    #B

A_prob = torch.tensor([0.3,0.5,0.2])    #C
S_prob = torch.tensor([0.6,0.4])    #C
E_prob = torch.tensor([[[0.75,0.25], [0.72,0.28], [0.88,0.12]],    #C
                     [[0.64,0.36], [0.7,0.3], [0.9,0.1]]])    #C
O_prob = torch.tensor([[0.96,0.04], [0.92,0.08]])    #C
R_prob = torch.tensor([[0.25,0.75], [0.2,0.8]])    #C
T_prob = torch.tensor([[[0.48,0.42,0.1], [0.56,0.36,0.08]],    #C
                     [[0.58,0.24,0.18], [0.7,0.21,0.09]]])    #C

def model():    #D
   A = pyro.sample("age", Categorical(probs=A_prob))    #D
   S = pyro.sample("gender", Categorical(probs=S_prob))    #D
   E = pyro.sample("education", Categorical(probs=E_prob[S][A]))    #D
   O = pyro.sample("occupation", Categorical(probs=O_prob[E]))    #D
   R = pyro.sample("residence", Categorical(probs=R_prob[E]))    #D
   T = pyro.sample("transportation", Categorical(probs=T_prob[R][O]))    #D
   return{'A': A,'S': S,'E': E,'O': O,'R': R,'T': T}    #D

pyro.render_model(model)    #E

#B The categorical distribution only returns integers, so it’s useful to write the integers’ mapping to categorical outcome names.
#C For simplicity I’ll use rounded versions of parameters learned with the “fit” method in pgmpy (Listing 3-4), though I could have learned the parameters in a training procedure.
#D When you implement the model in pyro, you specify the causal DAG implicitly using code logic.
#E You can then generate a figure of the implied DAG using pyro.render_model().


# %% [markdown] id="SmL3KOSd8G9o"
# ## Listing 3.9: Inference on the causal model in pyro
#
# In contrast to using variable elimination in pgmpy, we have to explicitly write an inference algorithm to infer probability queries of interest. In this code, we use importance sampling to infer P(E|T=1).

# %% colab={"base_uri": "https://localhost:8080/", "height": 490} id="jw2WgGZ08HFK" outputId="1cc3180f-2ab5-4b3b-b90b-b1922a8decaa"
import numpy as np
import pyro
from pyro.distributions import Categorical
from pyro.infer import Importance, EmpiricalMarginal    #A
import matplotlib.pyplot as plt

conditioned_model = pyro.condition(    #B
    model,    #C
    data={'transportation':torch.tensor(1.)}    #D
)

m = 5000    #E
posterior = pyro.infer.Importance(    #F
    conditioned_model,    #G
    num_samples=m    #H
).run()    #I
E_marginal = EmpiricalMarginal(posterior, "education")    #J
E_samples = [E_marginal().item() for _ in range(m)]    #J
E_unique, E_counts = np.unique(E_samples, return_counts=True)    #K
E_probs = E_counts / m    #K

plt.bar(E_unique, E_probs, align='center', alpha=0.5)    #L
plt.xticks(E_unique, E_alias)    #L
plt.ylabel('probability')    #L
plt.xlabel('education')    #L
plt.title('P(E | T = "train") - Importance Sampling')    #L

#A We’ll use two inference related classes, ImportanceSampling and EmpiricalMarginal.
#B pyro.condition is a conditioning operation on the model.
#C It takes in the model,
#D and evidence for conditioning on.  The evidence is a dictionary that maps variable names to values. The need to specify variable names during inference is why we have the name argument in the calls to pyro.sample.  Here we condition on T=”train”.
#E I’ll run a inference algorithm that will generate m samples.
#F Namely, I use importance sampling. The Importance class constructs this inference algorithm.
#G It takes the conditioned model and the number of samples.
#I Run the random process algorithm with the run method.  The inference algorithm will generate from the joint probability of the variables we didn’t condition on (everything but T) given the variables we conditioned on (T).
#J However, we only care about T, so EmpiricalMarginal operates on the output of algorithm so we obtain only samples of T.
#K Based on these samples, I produce a Monte Carlo estimation of the probabilities in P(E|T=”train”).
#L Plot a visualization of the learned probabilities.


# %% [markdown] id="CRjDV6Hp8POy"
# ## Listing 3.10: Creating a DAG based on roles in causal effect inference
#
# Some approaches to causal inference avoid specifying a causal DAG in favor of specifying the variables according to their role in a causal inference query. For example, in a causal effect inference query, a variable could be a "treatment" or "effect" or "instrumental variable" or "confounder" or a "mediator". But we can see this approach as merely another way of specifying a causal DAG. We can see this with the library DoWhy, which allows us to specify a model in terms of these roles. Once we do that, we can extract and visualize the DAG.

# %% colab={"base_uri": "https://localhost:8080/", "height": 406} id="NWoJFO8C8PVZ" outputId="edcd5e8c-4f28-4e99-8877-3267afb2d764"
from dowhy import datasets

import networkx as nx
import matplotlib.pyplot as plt

sim_data = datasets.linear_dataset(    #A
    beta=10.0,
    num_treatments=1,    #B
    num_instruments=2,    #C
    num_effect_modifiers=2,    #D
    num_common_causes=5,    #E
    num_frontdoor_variables=1,    #F
    num_samples=100,

)
dag = nx.parse_gml(sim_data['gml_graph'])    #G
pos = {    #G
 'X0': (600, 350),    #G
 'X1': (600, 250),    #G
 'FD0': (300, 300),    #G
 'W0': (0, 400),    #G
 'W1': (150, 400),    #G
 'W2': (300, 400),    #G
 'W3': (450, 400),    #G
 'W4': (600, 400),    #G
 'Z0': (10, 250),    #G
 'Z1': (10, 350),    #G
 'v0': (100, 300),    #G
 'y': (500, 300)    #G
}    #G
options = {    #G
    "font_size": 12,    #G
    "node_size": 800,    #G
    "node_color": "white",    #G
    "edgecolors": "black",    #G
    "linewidths": 1,    #G
    "width": 1,    #G
}    #G
nx.draw_networkx(dag, pos, **options)    #G
ax = plt.gca()    #G
ax.margins(x=0.40)    #G
plt.axis("off")    #G
plt.show()    #G

#A datasets.linear_dataset generates a DAG from the specified variables.
#B I add one treatment variable, like V in Figure 3 - 18.
#C Z in Figure 3 - 18 is an example of an instrumental variable; a variable that is a cause of the treatment but its only causal path to the outcome is through the treatment.  Here I create two instruments.
#D X0 and X1 are in Figure 3 - 18 are examples of “effect modifiers” that help model heterogeneity in the causal effect.  Dowhy defines these as other causes of the outcome (though they needn’t be).  Here I create two effect modifiers.
#E I add 5 common causes, like the three W0, W1, and W2 in Figure 3 - 18.  Unlike the nuanced structure between in Figure 3 - 18, the structure here will be simple.
#F Front door variables are on the path between the treatment and the effect, like U in Figure 3 - 18.  Here I add one.
#G This code extracts the graph, creates a plotting layout, and plots the graph.
