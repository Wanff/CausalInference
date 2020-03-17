import dowhy
from dowhy import CausalModel
import pandas as pd
import numpy as np
from IPython.display import Image, display

#get the data
data= pd.read_csv("https://raw.githubusercontent.com/AMLab-Amsterdam/CEVAE/master/datasets/IHDP/csv/ihdp_npci_1.csv", header = None)
col =  ["treatment", "y_factual", "y_cfactual", "mu0", "mu1" ,]
for i in range(1,26):
    col.append("x"+str(i))
data.columns = col
data = data.astype({"treatment":'bool'}, copy=False)
print(data.head())


# Create a causal model from the data and given common causes.
xs = ""
for i in range(1,26):
    xs += ("x"+str(i)+"+")

model=CausalModel(
        data = data,
        treatment='treatment',
        outcome='y_factual',
        common_causes=xs.split('+'),
        )
#save the model as a png
model.view_model()
display(Image(filename="causal_model.png"))

#Identify the causal effect
identified_estimand = model.identify_effect()
print(identified_estimand)


# Estimate the causal effect and compare it with Average Treatment Effect
estimate = model.estimate_effect(identified_estimand,
        method_name="backdoor.linear_regression", test_significance=True
)

print(estimate)

print("Causal Estimate is " + str(estimate.value))
data_1 = data[data["treatment"]==1]
data_0 = data[data["treatment"]==0]

print("ATE", np.mean(data_1["y_factual"])- np.mean(data_0["y_factual"]))

refute_results=model.refute_estimate(identified_estimand, estimate,
        method_name="random_common_cause")
print(refute_results)
