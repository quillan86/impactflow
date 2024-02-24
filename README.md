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

## Quick Start
ImpactFlow supports Python 3.9+. Once it's released, you would be able to use pip, poetry, or conda.


