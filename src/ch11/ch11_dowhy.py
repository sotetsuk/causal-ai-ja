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
# # Chapter 11, Part 1 - Building a Causal Effect Inference Workflow with DoWhy
#
# The notebook is a code companion to chapter 11 of the book [Causal AI](https://www.manning.com/books/causal-ai) by [Robert Osazuwa Ness](https://www.linkedin.com/in/osazuwa/).  View the [book resources](https://www.altdeep.ai/causalaibook) to see other tutorials and book-related links.
#
# <a href="https://colab.research.google.com/github/altdeep/causalML/blob/master/book/chapter%2011/Chapter_11_DoWhy_Causal_Effect_Workflow.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# %% colab={"base_uri": "https://localhost:8080/"} id="cvkoqysCeZpS" outputId="03ebfcca-cdad-4d72-b68b-5017dcbe6f69"
# !apt-get install graphviz libgraphviz-dev pkg-config    #A
# !pip install pygraphviz==1.12    #A
# !pip install dowhy==0.11
# !pip install econml==0.15

#A DoWHy uses pygraphviz to visualize the DAG. This uses apt-get to first install the underlying graphviz package in Debian-based Linux environment like Google Colab. Try using  Chocolatey to install graphviz for Windows and Brew for MacOS.

# %% [markdown] id="DCBawB0QUz-r"
# ## Listing 11.1: Build the Causal DAG
#
# We return to the online gaming DAG. We expand it with a few new variables.
#
# * Side-quest Group Assignment: 1 if a player was exposed to the mechanics that encouraged more side-quest engagement in the randomized experiment, 0 otherwise.
# * Customization Level: A score quantifying the player’s customizations of their character and the game environment.
# * Time Spent Playing (T): How much time the player has spent playing.
# * Prior Experience (Y): How much experience the player had prior to when they started playing the game.
# * Player Skill Level (S): A score of how well the player performs in game tasks.
# * Total inventory (V): The amount of game items the player has accumulated.
#
# ![causal DAG with latent](https://github.com/altdeep/causalML/blob/master/book/chapter%2011/images/gamingDAG.png?raw=true)
#
# We are interested in finding the average treatment effect between Side-quest Engagement and In-game Purchases.
#
# First, let’s build the DAG and visualize the graph with the PyGraphviz library.

# %% colab={"base_uri": "https://localhost:8080/", "height": 652} id="8ADuBXs015qN" outputId="f6466d2d-896f-4cff-9d72-046b6b27c9f8"
import pygraphviz as pgv    #A
from IPython.display import Image   #B

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
"""     #C
G = pgv.AGraph(string=causal_graph)    #C
G.draw('/tmp/causal_graph.png', prog='dot')    #D
Image('/tmp/causal_graph.png')    #E

#A Download pygraphviz and related libraries.
#B Optional import for visualizing the DAG in a Jupyter notebook
#C Specify the DAG as a DOT language string and load a pygraphviz AGraph object from the string.
#D Render the graph to a PNG file.
#E Display the graph.

# %% [markdown] id="KgEsn612b573"
# ## Listing 11.2: Download and display the data
#
# Let's download our data and see what variables are in our observational distribution.

# %% colab={"base_uri": "https://localhost:8080/"} id="RHvs5CLMIOsf" outputId="50b75f0e-0425-45d0-9005-23f2c38bb484"
import pandas as pd
data = pd.read_csv(
    "https://raw.githubusercontent.com/altdeep/causalML/master/datasets/online_game_example_do_why.csv"    #A
)
print(data.columns)    #B

#A Download an online gaming dataset.
#B Print the variables

# %% [markdown] id="XyTgNvyIcQEe"
# ## Listing 11.3: Instantiate an instance of DoWhy's CausalModel
#
# In our data, Prior Experience (Y) is unobserved.
#
# ![causal DAG with latent](https://github.com/altdeep/causalML/blob/master/book/chapter%2011/images/gamingDAGLatent.png?raw=true)
#
# We use the CausalModel class to represent the DAG and tell us if the ATE is identified.

# %% colab={"base_uri": "https://localhost:8080/"} id="XxGQ9hbYzUfR" outputId="421cd43c-cc2d-48b6-8e21-8ed4960f9336"
from dowhy import CausalModel    #A

model = CausalModel(
    data=data,    #B
    treatment='Side-quest Engagement',    #C
    outcome='In-game Purchases',    #C
    graph=causal_graph    #D
)

#A Install DoWhy and load the CausalModel class
#B Instantiate the CausalModel object with the data, which represents the level 1 observational distribution from which we derive the estimands.
#C We also specify the target causal query we wish to estimate, namely the causal effect of the "treatment" on the "outcome."
#D And of course we provide the causal DAG.

# %% [markdown] id="S-dLarPDiCBU"
# ## Listing 11.4: Run identification in DoWhy
#
# The `identify_effect` methods show us possible estimands we can target given our causal model and observed variables. The `identified_estimand` object is an object of the class `IdentifiedEstimand`. Printing it will list out the estimands, if any, and the assumptions they entail. In our case we have three estimands we can target:
# * The backdoor adjustment estimand through the adjustment set Player Skill Level, Guild Membership, and Time Spent Playing
# * The front-door adjustment estimand through the mediator Won Items
# * Instrumental variable estimands through Side-quest Group Assignment and Customization Level
#
#

# %% colab={"base_uri": "https://localhost:8080/"} id="u8TzWYPeqHxk" outputId="552cf1d1-ff77-438c-ade8-f131c86cea03"
identified_estimand = model.identify_effect()    #A
print(identified_estimand)

#A The identify_effect method of the CausalModel class lists identified estimands.

# %% [markdown] id="1oPk684Fi433"
# ## Listing 11.5 and 11.6. Estimating the backdoor estimand with linear regression (and printing the results)
#
# In DoWhy, we do estimation using a method in the CausalModel class called `estimate_effect`.

# %% colab={"base_uri": "https://localhost:8080/"} id="HPn-BWlOsotr" outputId="9f2651dc-453b-4562-e07c-4b3a14145163"
causal_estimate_reg = model.estimate_effect(
    identified_estimand,    #A
    method_name="backdoor.linear_regression",    #B
    confidence_intervals=True    #C
)
print(causal_estimate_reg)

#A The estimate_effect method takes an estimand as input.
#B method_name is of the form "[estimand].[estimator]". Here we use linear regression estimator to estimate the backdoor estimand.
#C Return confidence intervals.

# %% [markdown] id="dnJb-6J2MjR5"
# The "Causal Estimate" is the estimated coefficient for the treatment variable $v_0$.  The p-value is the significance test statistic for the null hypothesis that the true value of the coefficient is 0.  
#
# Note that once we have established that we want to use regression as the estimation method, all of the model evaluation methods for regression become relevant.  For example, if there were issues with the fit, such as residuals that had unstable variance or correlated with regression variables, it should cause us to question the linear assumption.  We also care about sparsity.  In this case, we had 30 data points to calculate seven coefficient estimates (one for the $v_0$, one each for the common cause confounders $W$'s and an intercept term.  If there were more potential common causes, we might need more data for this method to work.

# %% [markdown] id="bP38voZAp0cn"
# ## Listing 11.7: Propensity Score Stratification
#
# Propensity score methods are a collection of estimation methods for the backdoor estimand that use a quantity called the propensity score. The traditional definition of a propensity score is the probability of being exposed to the treatment conditional on the confounders. In the context of the online gaming example, this is the probability a player has high side-quest engagement given their guild membership (G), time spent playing (T), and player skill level (S), i.e. P(E=1|T=t, G=g, S=s) where t, i, and s are that player’s values for T, I, and S. In other words, it quantifies the player’s “propensity” of being exposed to the treatment (E=1). Typically P(E=1|T=t, G=g, S=s) is fit by logistic regression.
# But we can take a more expansive, machine learning friendly view of the propensity score. You can learn a propensity score function λ(...) of the backdoor adjustment set of confounders that renders those confounders conditionally independent of the treatment, as in the following figure.
#
# ![DAG with propensity score node](https://github.com/altdeep/causalML/blob/master/book/chapter%2011/images/propensity_score_node.png?raw=true)
#

# %% id="eb_5Ny5Lp2XI" colab={"base_uri": "https://localhost:8080/"} outputId="b7a4c54e-247d-40de-ba32-eb691db1d26b"
causal_estimate_strat = model.estimate_effect(
    identified_estimand,
    method_name="backdoor.propensity_score_stratification",     #A
    target_units="ate",
    confidence_intervals=True
)

print(causal_estimate_strat)

#A Propensity score stratification

# %% [markdown] id="mcGNxs5hzaIx"
# ## Listing 11.8: Propensity score matching
#
# Propensity score weighting methods use the propensity score to calculate a weight in a class of inference algorithms called inverse probability weighting. We implement this method in DoWhy as follows.

# %% id="t6gIZuKEvaoL" colab={"base_uri": "https://localhost:8080/"} outputId="824f608c-2f55-4a29-ee89-8c68cbb7d595"
causal_estimate_match = model.estimate_effect(
    identified_estimand,
    method_name="backdoor.propensity_score_matching",    #A
    target_units="ate",
    confidence_intervals=True
)
print(causal_estimate_match)

#A Propensity score matching

# %% [markdown] id="pYFIkPMvvZ1l"
# ## Listing 11.9: Propensity score weighting
# Propensity score weighting methods use the propensity score to calculate a weight in a class of inference algorithms called inverse probability weighting. DoWhy supports [a few different weighting schemes](https://microsoft.github.io/dowhy/dowhy.causal_estimators.html#module-dowhy.causal_estimators.propensity_score_weighting_estimator
# ):
# 1. Vanilla Inverse Propensity Score weighting (IPS) (`weighting_scheme="ips_weight"`)
# 2. Self-normalized IPS weighting (also known as the Hajek estimator) (`weighting_scheme="ips_normalized_weight"`)
# 3. Stabilized IPS weighting (`weighting_scheme = "ips_stabilized_weight"`)
#

# %% colab={"base_uri": "https://localhost:8080/"} id="yVXvBf97vtj-" outputId="b9a21e2f-5fbf-45ee-ad51-c22dc6496c51"
causal_estimate_ipw = model.estimate_effect(
    identified_estimand,
    method_name="backdoor.propensity_score_weighting",    #A
    target_units = "ate",
    method_params={"weighting_scheme":"ips_weight"},    #B
    confidence_intervals=True
)
print(causal_estimate_ipw)

#A Inverse probability weighting
#B Parameters used to set the IPS algorithm.

# %% [markdown] id="8lT48_Sn0Mh4"
# The fact that this estimator’s result differs so dramatically from the others suggest that it is relying on statistical assumptions that don’t hold in this data.

# %% [markdown] id="81JeFFJhqkoW"
# ## Listing 11.10: Double ML with DoWhy, EconML, and sklearn
#
# Recent developments in causal effect estimation focus on leveraging machine learning models. Most of these target the backdoor estimand. These approaches to causal effect estimation scale to large datasets and allow us to relax parametric assumptions such as linearity. The following DoWhy code uses the sklearn and EconML libraries for these machine learning methods. DoWhy’s `estimate_effects` provides a wrapper to the EconML’s implementation of these methods.
#
# Double machine learning is a backdoor estimator that uses machine learning methods to fit two predictive models: a model of the outcome given the adjustment set of confounders and a model of the treatment given the adjustment set. The approach then combines these two predictive models in a final-stage estimation to create a model of the target causal effect query.
# The following code performs double ML using a gradient boosting model and regularized regression model (LassoCV) from sklearn.
#

# %% colab={"base_uri": "https://localhost:8080/"} id="BvtkvQVtqjoQ" outputId="52f678df-d215-426c-f68b-870829976f54"
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LassoCV
from sklearn.ensemble import GradientBoostingRegressor

gb_estimate = model.estimate_effect(
    identified_estimand,
    method_name="backdoor.econml.dml.DML",    #A
    control_value = 0,
    treatment_value = 1,
    method_params={
        "init_params":{
            'model_y':GradientBoostingRegressor(),    #B
            'model_t': GradientBoostingRegressor(),    #C
            "model_final":LassoCV(fit_intercept=False),    #D
            'featurizer':PolynomialFeatures(degree=1, include_bias=False)
        },
        "fit_params":{}
    }
)
print(gb_estimate)
#A Select the double ML estimator
#B Use a gradient boosting model to model the outcome given the confounders.
#C Use a gradient boosting model to model the treatment given the confounders.
#D Use linear regression with L1 regularization (LASSO) as the final model.

# %% [markdown] id="SvBntf_L1T6x"
# ## Listing 11.11: Backdoor estimation with a meta-learner
#
# Meta learners are another machine learning method for backdoor estimation. Broadly speaking, meta learners train a model (or models) of the outcome given the treatment variable and the confounders, then account for the difference in prediction across treatment and control values of the treatment variable. They are particularly focused on highlighting heterogeneity of treatment effects across the data. The code below shows a meta learner example called a T learner that uses a random forest predictor.
#

# %% colab={"base_uri": "https://localhost:8080/"} id="0hoVsyTm0aLK" outputId="a3987750-6789-49a9-b0f7-9735b6260ba1"
from sklearn.ensemble import RandomForestRegressor    #A
metalearner_estimate = model.estimate_effect(    #A
    identified_estimand,    #A
    method_name="backdoor.econml.metalearners.TLearner",
    method_params={    #A
        "init_params": {'models': RandomForestRegressor()},    #A
        "fit_params": {}    #A
    }    #A
)    #A

print(metalearner_estimate)

# %% [markdown] id="M-gGZxJ7UotG"
# ## Listing 11.12: Front-door estimation with DoWhy
#
# Front-door estimation will run a two-stage linear model estimation procedure targeting the front-door estimand.

# %% colab={"base_uri": "https://localhost:8080/"} id="bqNQ3sDIU5Iu" outputId="ff51e7c6-7ace-4641-d1d7-f8836ffb6530"
causal_estimate_fd = model.estimate_effect(
    identified_estimand,
    method_name="frontdoor.two_stage_regression",    #A
    target_units = "ate",
    method_params={"weighting_scheme": "ips_weight"},    #B
    confidence_intervals=True
)
print(causal_estimate_fd)

#A Select two stage regression for the front door estimand.
#B Specify estimator hyperparameters.

# %% [markdown] id="tsLoiBD4vxuW"
# ## 11.13: Instrumental variable estimation and regression discontinuity
#
# Instrumental variable (IV) estimation uses the instrument Side-quest Group Assignment to estimate an estimand that doesn't depend on any of the common causes in the estimation.  Whether or not IV estimation is possible depends on the causal DAG.  Like regression estimation, it relies on certain parametric assumptions as well.

# %% colab={"base_uri": "https://localhost:8080/"} id="FZD_S6u_v1Xl" outputId="cedcce8f-e67e-4e29-f113-3b8e3fb1bc0b"
causal_estimate_iv = model.estimate_effect(
    identified_estimand,
    method_name="iv.instrumental_variable",    #A
    method_params = {
        "iv_instrument_name": "Side-quest Group Assignment"    #B
    },
    confidence_intervals=True
)
print(causal_estimate_iv)

#A Select intrumental variable estimation.
#B Select side-quest engagement as the instrument.

# %% [markdown] id="pB_OIrW75Sog"
# The large confidence interval suggests this estimator has too much variance to be useful.

# %% [markdown] id="qtPvwg7sv4mo"
# ## Listing 11.14 Regression discontinuity estimation with DoWhy
#
# Regression discontinuity is an estimation method popular in econometrics.  Regression discontinuity tries to model the "do" intervention by finding a threshold on a continuous variable that partitions the data into two parts corresponding to two different "do" values (e.g., "treatment" and "control").  It compares observations lying closely on either side of the threshold because data points close to that threshold will have similar values for the confounders.
#
# DoWhy treats it as a special type of [IV approach](https://microsoft.github.io/dowhy/dowhy.causal_estimators.html#dowhy.causal_estimators.regression_discontinuity_estimator.RegressionDiscontinuityEstimator
# ). The argument `rd_variable_name` name of the variable on where the thresholding occurs. This variable is analogous to the instrument.
# `rd_threshold_value` is the threshold value where the split (or "discontinuity") occurs.  `rd_bandwidth` is the distance from the threshold within which confounders can be considered the same between treatment and control.
# *italicized text*
#

# %% id="yC9t3Va7v8yW"
# %%capture
causal_estimate_regdist = model.estimate_effect(
    identified_estimand,
    method_name="iv.regression_discontinuity",    #A
    method_params={
        'rd_variable_name':'Customization Level',    #B
        'rd_threshold_value':0.5,    #C
        'rd_bandwidth': 0.15    #D
    },
    confidence_intervals=True,
)

#A DoWhy treats it as a special type of IV estimator.
#B Using Customization Level as our instrument.
#C The threshold value for the split ("discontinuity").
#D The distance from the threshold within which confounders are considered the same between treatment and control values of the treatment variable.

# %% colab={"base_uri": "https://localhost:8080/"} id="9Dw3cAq-8Rdi" outputId="d57101af-6f83-4385-f5da-b3d6bd971183"
print(causal_estimate_regdist)

# %% [markdown] id="rVCPseen6GOj"
# Again, though the estimate is not too far off the those of the other estimators, the confidence intervals are a bit too large for comfort, suggesting either the instrument is weak or that we need to tune the arguments passed to the estimator.
#

# %% [markdown] id="WYi6yUov1M8q"
# ## Listing 11.15: Refutating the assumption of sufficient data
#
# Refutation sensitivity tests enable us to test the sensitivity of our estimations to violations of our model assumptions.
#
# ### Data size reduction
# One way to test the robustness of the analysis is to reduce the size of the data and see if you obtain similar results.  If the size of your data small relative to your estimator, then removal of that already sparse data would have an impact on your results.

# %% colab={"base_uri": "https://localhost:8080/"} id="WL9wSZVjzS45" outputId="977e9f93-e283-4f57-dce4-81c31fc157b9"
identified_estimand.set_identifier_method("frontdoor")    #A
res_subset = model.refute_estimate(
    identified_estimand,    #B
    causal_estimate_fd,    #C
    method_name="data_subset_refuter",    #D
    subset_fraction=0.8,    #E
    num_simulations=100    #F
)
print(res_subset)

#A Not strictly necessary, but in some cases clarifying the estimand when working with refuters helps avoid errors.
#B The refute_estimate function takes in the identified estimand...
#C ... and the estimator that targets the estimand.
#D We select "data_subset_refuter", which tests if the causal estimate is different when we run the analysis on a subset of the data.
#E Set the size of the subset to 80% the size of the original data.
#F Set the number of simulations to 30.

# %% [markdown] id="96fO49Pt2h3P"
# “Estimand Effect” is the effect from our original analysis.  “New effect” is the average ATE across the simulations.  We hope these two effects to be similar.  The p-value is calculated under the null hypothesis that the two are the same.  P-values falling below a traditional significance threshold (e.g., .1, .05) indicate evidence that the two effects are different, which would mean our analysis is sensitive to the size of our data.  This is not the case with our analysis, since our p-value is high.

# %% [markdown] id="DAWEyzzM2H2B"
# ## Listing 11.16: Adding a dummy variable
#
# One way to test our models is to add dummy variables – variables that have no causal bearing on our problem, but where the additional noise can impact the performance of the statistical estimator.  One way to test this is by adding a dummy confounder and re-running the analysis.  The dummy confounder is a random variable that is not actually a confounder, it is completely independent of the treatment and outcome variables, but the analysis treats it as a confounder.  By doing this many times, we can evaluate how sensitive your results are to the random common cause.

# %% id="4zyNBy4v2Bwo" colab={"base_uri": "https://localhost:8080/"} outputId="0c1f85e7-0190-4731-9ef3-f035809e1639"
identified_estimand.set_identifier_method("backdoor")

res_random = model.refute_estimate(    #A
    identified_estimand,    #A
    gb_estimate,    #A
    method_name="random_common_cause",    #A
    num_simulations=100,    #A
    n_jobs=2    #A
)    #A
print(res_random)


# %% [markdown] id="fRZZKH9p2oDU"
# “Estimated effect” is the original causal effect estimate that we obtained from your model before running the refutation test.  “New effect” is the new causal effect estimate obtained after adding a random common cause to your data and re-running the analysis.  “p value” is the p-value of the test comparing the original and new estimated effects. A small p-value suggests that the two effects are statistically different, indicating that your original estimate might be sensitive to potential unobserved confounding.
# Given ourresults, the original estimated effect and the new effect after adding a random common cause are quite similar, and the p-value is greater than typical significance cutoffs (e.g., 0. or 0.05). This means there's no strong statistical evidence to suggest that the estimated effect changes significantly with the addition of a random common cause. So, the original estimated causal effect appears to be quite robust, at least to the addition of random common causes, according to this specific refutation test.

# %% [markdown] id="_rstWy7V6upL"
# ## Listing 11:17: Replacing the treatment variable with a dummy variable
#
# We can also experiment with replacing the treatment variable with a dummy (placebo) variable.  Here, we expect the ATE to be close to 0.

# %% id="kqCyNguG2zCj" colab={"base_uri": "https://localhost:8080/"} outputId="ba703f77-369b-4d8d-e62a-4540a55ab965"
identified_estimand.set_identifier_method("backdoor")

res_placebo = model.refute_estimate(
    identified_estimand,    #A
    causal_estimate_ipw,    #A
    method_name="placebo_treatment_refuter",    #A
    placebo_type="permute",    #A
    num_simulations=100    #A,
)

print(res_placebo)

#A This refuter replaces the treatment variable with a dummy (placebo) variable.

# %% [markdown] id="5eIf2rCn3PZU"
# “New Effect” is the ATE generated when the treatment variable (in our case Tech Support) is replaced in the analysis by a random dummy variable.  We expect the new ATE to be 0, and the p-value here compares New Effect value to 0.  Values that are significantly different from 0 will have a low p-value.  When the p-value falls below typical significance thresholds (e.g., .05), the test shows evidence that your analysis might be overly sensitive and that you should be suspicious of your ATE estimates.  In our case, the high p-value suggests that this is not the case.
#

# %% [markdown] id="5PfiXOTF7IYU"
# ## Listing 11.18: Replacing the outcome variable with a dummy variable
#
# We can substitute the outcome variable with a dummy variable. The ATE in this case should be 0 because the treatment has no effect on this dummy. We’ll simulate it as a linear function of some of the confounders so the outcome still has a meaningful relationship with some of the covariates. I’ll try this with the front door estimator.

# %% id="Mms5qfR63SpC" colab={"base_uri": "https://localhost:8080/"} outputId="82a0ed7b-88a1-45d1-ba99-bdd24f01bf21"
import numpy as np

coefficients = np.array([100.0, 50.0])
bias = 50.0
def linear_gen(df):     #A
    subset = df[['guild_membership','player_skill_level']]     #A
    y_new = np.dot(subset.values, coefficients) + bias     #A
    return y_new     #A

identified_estimand.set_identifier_method("frontdoor")
ref = model.refute_estimate(    #B
    identified_estimand,    #B
    causal_estimate_fd,    #B
    method_name="dummy_outcome_refuter",    #B
    outcome_function=linear_gen    #B
)    #B

res_dummy_outcome = ref[0]
print(res_dummy_outcome)

#A Create a function that generates a new dummy outcome variable as a linear function of the covariates.
#B Runs refute_estimate with a dummy outcome refuter.


# %% [markdown] id="Stidih4h7ruC"
# Again, the p-value is calculated under the null hypothesis that New effect equals 0, and a low p-value refutes our assumptions. In this case, our assumptions are not refuted.
#

# %% [markdown] id="tmtk16EI4cGr"
# ## Listing 11.19: Adding an unobserved confounder
#
# Since we used backdoor adjustment, we assume that the adjustment set blocks all backdoor paths.  If there were an unobserved confounder that we failed to adjust for, that assumption is violated and our estimate would have confounder bias.  That is not necessarily the worst thing; if we adjust for all major confounders, any bias from unobserved confounders could be small and not impact our results by much.  On the other hand, missing a major confounder could lead us to conclude there is a nonzero ATE when there is none, or conclude a positive ATE when the true ATE is negative, or vice versa.  We therefore test our robust our analysis is to introducing unobserved confounders.

# %% id="bqTWKg1f3ukC" colab={"base_uri": "https://localhost:8080/", "height": 612} outputId="e7f13298-a318-4d35-9915-0bcfcd7cd28e"
identified_estimand.set_identifier_method("backdoor")
res_unobserved = model.refute_estimate(    #A
    identified_estimand,    #A
    causal_estimate_strat,    #A
    method_name="add_unobserved_common_cause"    #A
)    #A

print(res_unobserved)

#A Setting up a refuter that adds an unobserved common cause

# %% [markdown] id="wBSKEYR379S4"
# This analysis does not return a p-value. It produces the above heat map, which shows how quickly the estimate changes when the no unobserved confounder assumption is violated.  The horizontal axis in the heat map shows the various levels of influence the unobserved confounder has on the outcome, and the vertical axis shows the various levels of influence the confounder can have on the treatment. The color corresponds to the new effect estimates that result at different levels of influence. Here, we see the ATE is quite sensitive to the effect the confounder has on the treatment. Note that you can change the default parameters of the refuter to experiment with different impacts the confounder could have on the treatment and outcome.
