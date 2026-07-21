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

# %% [markdown] id="0E34xvN8et9z"
# # Chapter 2 - Primer on Probability Modeling
#
# The notebook is a code companion to chapter 2 of the book [Causal AI](https://www.manning.com/books/causal-ai) by [Robert Osazuwa Ness](https://www.linkedin.com/in/osazuwa/).
#
# <a href="https://colab.research.google.com/github/altdeep/causalAI/blob/master/book/chapter%202/Chapter_2_Primer_on_Probability_Modeling.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# %% [markdown] id="MjO5TOkYgm53"
# This code was written with pgmpy version 0.1.24 and pyro version 1.8.6. Install from within a Python environment using:

# %% id="98ENU9Vy1KK7" colab={"base_uri": "https://localhost:8080/"} outputId="6cd65485-0d4c-4116-807a-92194847ea65"
# !pip install pgmpy==0.1.24
# !pip install pyro-ppl==1.8.6

# %% [markdown] id="vAPYaH_xku_W"
# pgmpy depends on pandas. The version of pandas used was 1.5.3. Check your version with:

# %% colab={"base_uri": "https://localhost:8080/"} id="DX7q6rCtk1zj" outputId="073c1b75-dad1-4853-f2d6-4a11b5629bf8"
import pandas as pd
print(pd.__version__)

# %% [markdown] id="HcycIJX81Pmm"
#
# ## Listing 2.1: Implementing a discrete distribution table in pgmpy
#

# %% colab={"base_uri": "https://localhost:8080/"} id="qzPPspx4eufQ" outputId="19d1243b-1976-4d01-8d19-a9920d2dfa95"
import pgmpy
from pgmpy.factors.discrete import DiscreteFactor
dist = DiscreteFactor(
    variables=["X"],    #A
    cardinality=[3],    #B
    values=[.45, .30, .25],    #C
    state_names= {'X': ['1', '2', '3']}    #D
)

#A A list of the names of the variables in the factor
#B The cardinality (number of possible outcomes of each variable in the factor
#C The values each variable in the factor can take
#D Dictionary where the key is the variable name and the value is a list of the names of that variable’s outcomes

print(dist)

# %% [markdown] id="A-D3GoX4fN52"
# “phi(X)” is the probability value assigned to each outcome of X.  One thing to note is that these phi values don’t need to sum to one.  For example, I can multiple each probability value by one hundred as follows.

# %% colab={"base_uri": "https://localhost:8080/"} id="G8bwqFF2fDYb" outputId="6c25ab42-b888-45fc-faa4-2b4df0a78e29"
dist = DiscreteFactor(
    variables=["X"],
    cardinality=[3],
    values=[45, 30, 25],
    state_names= {'X': ['1', '2', '3']}
)

print(dist)

# %% [markdown] id="tdAVY-hpfa_N"
# pgmpy relaxes the constraint to sum to one is because the relaxation is useful in some algorithms.  We can always normalize to obtain proper probability values.

# %% colab={"base_uri": "https://localhost:8080/"} id="gA1aBQajfZk7" outputId="52f38272-ac91-4772-b192-06ff9a6eb587"
# Normalize takes phi values for each outcome and divides them by their sum to get a probability.
dist.normalize()

print(dist)

# %% [markdown] id="HDS9Juwtf6tY"
# ## Listing 2.2 Modeling a joint distribution in pgmpy

# %% colab={"base_uri": "https://localhost:8080/"} id="TXRN08jSfqCl" outputId="eccecfdf-272b-47eb-8baa-73b82b12a152"
joint = DiscreteFactor(
    variables=['X', 'Y'],   #A
    cardinality=[3, 2],    #B
    values=[.25, .20, .20, .10, .15, .10],   #C
    state_names= {
        'X': ['1', '2', '3'],    #C
        'Y': ['0', '1']    #C
    }
)

#A Now we have two variables instead of one.
#B X has 3 outcomes, Y has 2.
#C You can look at the printed output to see how the values are ordered of values.
#D Now there are two variables, so we name the outcomes for both variables.

print(joint)

# %% [markdown] id="LmU5Mf5mgO-H"
# The marginalize method will accomplish summing over the specified variables for us.
#

# %% colab={"base_uri": "https://localhost:8080/"} id="5FQ1-LtqhoRq" outputId="ea80e541-2dc0-48c6-b881-22212defb849"
print(joint.marginalize(variables=['Y'], inplace=False))
print(joint.marginalize(variables=['X'], inplace=False))

# %% [markdown] id="3DQjnxI9ZFaZ"
# We can use a division operator on two factors to calculate conditional probabilities. Here, P(X, Y)/P(X) = P(Y|X).

# %% colab={"base_uri": "https://localhost:8080/"} id="6pHGwacShqn7" outputId="6b9cc471-5ebc-43d3-a58e-2b3c6b8a5eb6"
print(joint / dist)

# %% [markdown] id="lNje9Z9IZBY2"
# ## Listing 2.3 Canonical parameters in Pyro
#
# However, the more direct way in pgmpy to specify a conditional probability distribution table is with the TabularCPD class:

# %% colab={"base_uri": "https://localhost:8080/"} id="4eUC_8dBZPCW" outputId="59e0ea89-05be-47e4-c880-c86b36538f85"
from pgmpy.factors.discrete.CPD import TabularCPD
PYgivenX = TabularCPD(
    variable='Y',    #A
    variable_card=2,    #B
    values=[
        [.25/.45, .20/.30, .15/.25],    #C
        [.20/.45, .10/.30, .10/.25],    #C
    ],
    evidence=['X'],
    evidence_card=[3],
    state_names = {
        'X': ['1', '2', '3'],
        'Y': ['0', '1']
    })

#A Conditional distribution has one variable instead of DiscreteFactor’s list of variables.
#B variable_card is the cardinality of Y
#C Elements of the list correspond to outcomes for Y. Elements of each list correspond to elements of X.

print(PYgivenX)

# %% [markdown] id="FyDHZZDFZy80"
# ### Canonical parameters in Pyro

# %% colab={"base_uri": "https://localhost:8080/"} id="UG_fOPwiZokM" outputId="6f7980ab-7b5c-4ac6-f887-e7f56deac10e"
import torch
from pyro.distributions import Bernoulli, Categorical, Gamma, Normal

# The categorical distribution takes a list of probability values, each value corresponding to an outcome.
print(Categorical(probs=torch.tensor([.45, .30, .25])))
print(Normal(loc=0.0, scale=1.0))
print(Bernoulli(probs=0.4))
print(Gamma(concentration=1.0, rate=2.0))

bern = Bernoulli(0.4)
lprob = bern.log_prob(torch.tensor(1.0))

import math
print(math.exp(lprob))


# %% [markdown] id="4sKd5PbLbvtu"
# ## Listing 2.4: Simulating from DiscreteFactor in pgmpy and Pyro

# %% colab={"base_uri": "https://localhost:8080/", "height": 143} id="lOqDNVM8buUH" outputId="41f0d9e6-6ffc-489b-885b-419ed529c9ad"
from pgmpy.factors.discrete import DiscreteFactor
dist = DiscreteFactor(
    variables=["X"],
    cardinality=[3],
    values=[.45, .30, .25],
    state_names= {'X': ['1', '2', '3']}
)

#n is the number of instances you wish to generate
dist.sample(n=3)


# %% colab={"base_uri": "https://localhost:8080/", "height": 143} id="3gFe94BJb904" outputId="e6bd60fc-0224-4750-db9c-36aba79bdd6a"
joint = DiscreteFactor(
    variables=['X', 'Y'],
    cardinality=[3, 2],
    values=[.25, .20, .20, .10, .15, .10],
    state_names= {
        'X': ['1', '2', '3'],
        'Y': ['0', '1']
    }
)

joint.sample(n=3)


# %% [markdown] id="jL-oWJuRcNFI"
# Canonical distributions in pyro also use a method with sample method.

# %% colab={"base_uri": "https://localhost:8080/"} id="LvvQ_jv1cJLZ" outputId="82775bab-129d-4207-b048-9849d251178c"
import torch
from pyro.distributions import Categorical
# Generate 1 sample
one_sample = Categorical(probs=torch.tensor([.45, .30, .25])).sample()
print(one_sample)
# Generate 10 samples
samples = Categorical(probs=torch.tensor([.45, .30, .25])).sample(sample_shape=torch.Size([10]))
print(samples)

# %% [markdown] id="MtP49-FcdZ0G"
# ## Listing 2.5: Creating a random process in pgmpy and pyro.

# %% colab={"base_uri": "https://localhost:8080/", "height": 175, "referenced_widgets": ["9f11d0952a4f4bc88706b3bcbd88f81d", "aa451de21abe4301aeb5b84c116fa3aa", "b10785a458f940ba9427a4005c4326a0", "6cfc8555cbb24fd887b9e2781d88ef00", "d5434fbd6fd3469e92e404c95486ee94", "bc85f56ee02a412ca95314fbc5ff6af3", "155589f0ef68409a8bc1e02c71debe80", "f60c520592344c4a9806592b0de1cbeb", "702c9c6fbfb844eb879a7d0e17d65efd", "9512dedfb8204dc580dc9401cd4a4c79", "45feef2e893e45d2b6b445b6829d83ef"]} id="ix9uF77qcW-D" outputId="57eaa10e-94ba-4d60-9112-ff24602df2a3"
from pgmpy.factors.discrete.CPD import TabularCPD
from pgmpy.models import BayesianNetwork
from pgmpy.sampling import BayesianModelSampling

# Values is the probability value assigned to each outcome. The
PZ = TabularCPD(
    variable='Z',
    variable_card=2,
    values=[[.65], [.35]],
    state_names = {
        'Z': ['0', '1']
    })

# The probability values map to the outcomes specified in the state_names argument.
PXgivenZ = TabularCPD(
    variable='X',
    variable_card=2,
    values=[
        [.8, .6],
        [.2, .4],
    ],
    evidence=['Z'],
    evidence_card=[2],
    state_names = {
        'X': ['0', '1'],
        'Z': ['0', '1']
    })

PYgivenX = TabularCPD(
    variable='Y',
    variable_card=3,
    values=[
        [.1, .8],
        [.2, .1],
        [.7, .1],
    ],
    evidence=['X'],
    evidence_card=[2],
    state_names = {
        'Y': ['1', '2', '3'],
        'X': ['0', '1']
    })

model = BayesianNetwork([('Z', 'X'), ('X', 'Y')])
model.add_cpds(PZ, PXgivenZ, PYgivenX)

generator = BayesianModelSampling(model)
generator.forward_sample(size=3)


# %% [markdown] id="Mx1_KI7AdOsm"
# ## Listing 2.6 Working with combinations of canonical distributions in Pyro

# %% colab={"base_uri": "https://localhost:8080/"} id="92aWaC4qhcv0" outputId="5ec7b397-c918-47f4-fe08-88624264e7c5"
import torch
from pyro.distributions import Bernoulli, Poisson, Gamma

# Represent P(Z) with a Gamma distribution, and sample z.
z = Gamma(7.5, 1.0).sample()    #A
# Represent P(X|Z=z) with a Poisson distribution with location parameter z, and sample x.
x = Poisson(z).sample()    #B
# Represent P(Y|X=x) with a Bernoulli distribution. The probability parameter is a function of y.
y = Bernoulli(x / (5+x)).sample()    #C

print(z, x, y)


# %% [markdown] id="4eYUHJFUj7cp"
# ## Listing 2.7: Random processes with nuanced control flow in Pyro

# %% id="IrGBg5gYj80N" colab={"base_uri": "https://localhost:8080/"} outputId="403be5e8-b6c3-4b8d-82fe-c0ad7f111ce3"
import torch
from pyro.distributions import Bernoulli, Poisson, Gamma
z = Gamma(7.5, 1.0).sample()
x = Poisson(z).sample()
# y is calculated as a sum of random coin flips.
# y is generated from P(Y|X=x) because the number of flips depends on x.
y = torch.tensor(0.0)
for i in range(int(x)):
    y += Bernoulli(.5).sample()
print(z, x, y)

# %% [markdown] id="t9apo5zPlsz-"
# ## Listing 2.8: Using functions for random processes and pyro.sample
#
# In the code below, we generate multiple samples and then use the `mean` method to calculate an average of those samples.

# %% colab={"base_uri": "https://localhost:8080/", "height": 66, "referenced_widgets": ["71c5858bd1484592a979029f2895372a", "cdd2fbd60bb64a1a94ef92b9e462687f", "007075b2e88741ccb5ce36d54629d941", "621325d1cffe4b64bb9f5b2091b24f72", "339551fb3ab54f4fa4b8cf2a570874b9", "ef22f67a1aed4acd88188df4a1c51af2", "9ca1040c5670414e80b7d54f66ec2f37", "d48dc4d414e74563b6c86f509ae6b029", "d96051cbe7ee4b1987cf0bf55cfbe308", "cb874634568247f8b2685531cc594f1a", "3fe2bf351ce74c4db1dcce24399410a0"]} id="WaaIKa8dlt5q" outputId="8a7c5e7f-9dfe-407d-8904-c32b08332c42"
import torch
import pyro
def random_process():
    z = pyro.sample("z", Gamma(7.5, 1.0))
    x = pyro.sample("x", Poisson(z))
    y = torch.tensor(0.0)
    for i in range(int(x)):
        y += pyro.sample(f"y{i}", Bernoulli(.5))    #A
    return y

generated_samples = generator.forward_sample(size=100)
generated_samples['Y'].apply(int).mean()


# %% colab={"base_uri": "https://localhost:8080/"} id="NXfZ1J87mFxn" outputId="a41a47f8-407b-4542-d987-12fe35e2b7ec"
generated_samples_ = torch.stack([random_process() for _ in range(100)])
generated_samples_.mean()

# %% colab={"base_uri": "https://localhost:8080/"} id="TZpCf8eFmQh6" outputId="4347cd19-b119-41fe-c8cd-413e0dcf5999"
torch.square(generated_samples_).mean()

# %% [markdown] id="lfVI_FdUoBUI"
# ## Listing 2.9: Monte Carlo estimation in Pyro

# %% colab={"base_uri": "https://localhost:8080/"} id="U1pve2xmoYAS" outputId="9c4fc206-a2f8-4705-ea56-58a61a2570a6"
import torch
import pyro
from pyro.distributions import Bernoulli, Gamma, Poisson
# This new version of random_process now returns both z and y.
def random_process():
    z = pyro.sample("z", Gamma(7.5, 1.0))
    x = pyro.sample("x", Poisson(z))
    y = torch.tensor(0.0)
    for i in range(int(x)):
        y += pyro.sample(f"{i}", Bernoulli(.5))
    return z, y

# Generate 1000 instances of z and y using a list comprehension.
generated_samples = [random_process() for _ in range(1000)]
# Turn the individual z tensors into a single tensor,
# then calculate the Monte Carlo estimate via the mean method.
z_mean = torch.stack([z for z, _ in generated_samples]).mean()

print(z_mean)

z_given_y = torch.stack([z for z, y in generated_samples if y == 3])

print(z_given_y.mean())


# %% colab={"base_uri": "https://localhost:8080/", "height": 395, "referenced_widgets": ["e3e2c51b2c9c4bea8f4cfa30c74c315f", "d753155e1bce4edda4da7d74d26aa9bb", "6274422394d5409d8bc1d9d5948dd91e", "f59864d71e1040aebfdec73c9d343b8e", "e6d89b089e3a4727aa0b692e8a60152c", "afdd02e766974a8ca9b686d732bf6111", "c7134ac8f2b04fb181909c6044317b59", "d0090718d9394033b4cdbf215fa4b6ff", "e123e27adbbf4c0d9b5748a47575bcad", "7a43be8e468c4b5d95eb4876fdf3773c", "e54e0af3593a473787922cef7b9af0f7"]} id="sKz_hqByoxHf" outputId="527de1f4-8fdf-41e1-f5a3-6b988022cbc5"
generator.forward_sample(size=10)

# %% [markdown] id="aajo3IGGo0Gq"
# ## Listing 2.10: Generating IID samples in Pyro

# %% colab={"base_uri": "https://localhost:8080/"} id="x_v3os5No1Mv" outputId="842f455f-2de2-4200-f0a9-44099f2dfdc7"
import pyro
from pyro.distributions import Bernoulli, Poisson, Gamma

def model():
    z = pyro.sample("z", Gamma(7.5, 1.0))
    x = pyro.sample("x", Poisson(z))
    # pyro.plate is a context manager for generating conditional independent
    # samples. This instance of pyro.plate will generate 10 IID samples.
    with pyro.plate('IID', 10):
        # Calling pyro.sample to generates a single outcome y, where y is a
        # tensor of 10 IID samples.
        y = pyro.sample("y", Bernoulli(x / (5+x)))
    return y

model()


# %% [markdown] id="EZliAGAGpOMO"
# ## Listing 2.11: An example of a data generating process in code form

# %% colab={"base_uri": "https://localhost:8080/"} id="RpuftchRpPUc" outputId="38f0e857-e1ce-4bd0-8ffe-1fc860b14f50"
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

#A Input variables reflect Jenny and Brian’s inclination to throw and the window strength.
#B Jenny and Brian throw the rock if so inclined.
#C If both Jenny and Brian throw the rock, the total strength of the impact is .8.
#D If either Jenny or Brian throws the rock, the total strength of the impact is .6.
#E Otherwise, no one throws and the strength of impact is 0.
#F If the strength of impact is greater than the strength of the window, the window breaks.

initials = [
    (0.6, 0.31, 0.83),
    (0.48, 0.53, 0.33),
    (0.66, 0.63, 0.75),
    (0.65, 0.66, 0.8),
    (0.48, 0.16, 0.27)
]

data_points = []
# Now we pass each tuple in the initials list as an input to the true_dgp function, generate a data point, and add this to the data_points list.
for jenny_inclination, brian_inclination, window_strength in initials:
    data_points.append(
        true_dgp(
            jenny_inclination, brian_inclination, window_strength
        )
    )

print(data_points)
