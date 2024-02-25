# ImpactFlow
ImpactFlow is a Python library designed for advanced decision modeling, leveraging the principles of causal decision models. It differentiates itself by focusing on the dynamic interplay between levers, external factors, and outcomes within a decision-making context. Unlike traditional causal modeling approaches, such as those found in DoWhy's GCM package, ImpactFlow tailors its functionality to meet the specific needs of decision support, emphasizing the simulation and analysis of decision impacts under varying conditions.

## Key Features

ImpactFlow will be engineered with a suite of powerful features aimed at enhancing decision-making processes:

- **Graph-Based Framework**: The model uses a directed graph (DiGraph) structure to represent decision elements, including levers, outcomes, and external factors. This structure captures the causal relationships and dependencies between different elements, facilitating an intuitive representation of complex decision-making environments.

- **Outcome Prediction**: ImpactFlow dynamically evaluates the effects of adjustable factors and external influences on specified outcomes. It utilizes predefined predictive functions or models tied to decision elements, allowing for the exploration of various scenarios and their impacts. 

- **Scenario Analysis**: ImpactFlow allows users to assess the impacts of various lever configurations on outcomes. This analysis aids in understanding the potential effects of different strategies, enabling stakeholders to make informed decisions by exploring a wide range of scenarios and their implications.

- **Optimization**: The model provides robust tools for optimizing decision outcomes, accommodating both single-objective and multi-objective optimization strategies. This capability enables the identification of optimal or Pareto-optimal sets of lever adjustments to achieve desired objectives, facilitating strategic decision-making in complex environments.

- **Sensitivity Analysis**: ImpactFlow includes functionality for sensitivity analysis, leveraging the SALib library to understand how variations in lever values influence outcomes. This feature is crucial for identifying critical levers and assessing the robustness of decision strategies.

- **User-defined Flexibility**: The framework is designed to be flexible, allowing users to define their own decision elements, relationships, and optimization criteria based on specific needs and objectives.

ImpactFlow positions itself as a pivotal tool in the realm of decision modeling, offering nuanced insights and strategic foresight into the outcomes of various decision-making scenarios.

## Usage

Begin by first constructing a Causal Decision Model.

First construct levers by including the name, the default value, and bounds of the levers:
```angular2html
from impactflow import Lever
lever1 = Lever("Marketing Spend", 2750000.00, [0.0, 6000000.00])
lever2 = Lever("Usability", 75.0, [0.0, 100.0])
lever3 = Lever("Product Price", 10.00, [1.00, 20.00])
```
Externals are factors a decision maker cannot control. Adding externals is similar to adding levers. They primarily differ later on in that they aren't optimized over.
```angular2html
from impactflow import External
external1 = External("CPC", 1.16, [1.00, 2.00])
```
The next step are adding intermediates, which are decision elements that act in between levers/externals and outcomes.
They don't have default values or bounds, but rather are linked to earlier elements.
```angular2html
from impactflow import Intermediate
itm1 = Intermediate("Conversion Rate", ["Usability", "Product Price"])
itm2 = Intermediate("Loyalty", ["Marketing Spend", "Usability", "Product Price"]
itm3 = Intermediate("New Visitors", ["Marketing Spend"])
itm4 = Intermediate("Returning Visitors", ["Loyalty"])
itm5 = Intermediate("Total Visitors", ["New Visitors", "Returning Visitors"])
itm6 = Intermediate("Orders", ["Conversion Rate", "Total Visitors"])
```

Intermediates need to have a function attached to them. This is done with the `fit` function. To begin with, `fit` will take regular callable functions but will down the road take a number of different kinds of models.
The basic idea is that the function has the arguments in the same order that have been passed when the intermediate was created.
```angular2html
# itm5 was Total Visitors = this should be simple addition of new and returning visitors
itm5 = itm5.fit(lambda x, y: x + y)
```
Finally, the model should have one or more outcomes. The outcome should be fit in the same fashion as intermediates.
```angular2html
from impactflow import Outcome
outcome = Outcome("Revenue", ["Orders", "Product Price"]).fit(lambda x, y: x * y)
```
Now that the elements have been created, they can be placed in the Causal Decision Model. The `CausalDecisionModel` class takes the levers as inputs.
```angular2html
from impactflow import CausalDecisionModel
# Create the CDM and feed it the levers
model = CausalDecisionModel([lever1, lever2, lever3])
# add a single eleemnt
model.add_element(external1)
# add multiple elements
model.add_elements([itm1, itm2, itm3, itm4, itm5, itm6])
# add the outcome
model.add_element(outcome)
```

An element can be called with the `model.element` method. This can be used to set the value of a lever or external.
```angular2html
lever2 = model.element("Usability")
model.element("Usability").value = 80.0 # value of lever changed! This will propogate.
```
You can `predict` a specific intermediate or outcome to get the value of the element relative to an input. Meanwhile, you can `call` an intermediate or outcome to do the same, but the inputs are the levers of the model rather than the immediate inputs to the element.
```angular2html
revenue = model.element("Revenue").predict({"Orders": 10000.0, "Product Price": 10.0})
revenue = model.element("Revenue").call([2750000.00, 75.0, 10.00]}
```
You can optimize over a single or multiple outcomes:
```angular2html
# single outcome only
lever1_value, lever2_value, lever3_value = model.optimize("Revenue")
# single outcome but list shows that you can optimize over multiple outcomes at once
lever1_value, lever2_value, lever3_value = model.optimize(["Revenue"])
```


## Quick Start
ImpactFlow supports Python 3.9+. Once it's released, you would be able to use pip, poetry, or conda.


