from typing import Union
import numpy as np
from pymoo.core.problem import ElementwiseProblem


class MultiObjectiveProblem(ElementwiseProblem):
    def __init__(self, model, outcome_names: list[str], bounds: list[int | float], types: Union[str, list[str]] = "min"):
        """
        :param model: Instance of CausalDecisionModel.
        :param outcome_names: List of names of outcomes to be optimized.
        :param bounds: Bounds for each lever in the form of (min, max) tuples.
        :param type: whether max or min.
        """
        self.model = model
        self.outcome_names: list[str] = outcome_names
        self.types: str = types
        super().__init__(n_var=len(bounds),
                         n_obj=len(outcome_names),
                         n_constr=0,
                         xl=np.array([b[0] for b in bounds]),
                         xu=np.array([b[1] for b in bounds]))

    def _evaluate(self, x, out, *args, **kwargs):
        def model_call(outcome_name, type_, x):
            if type_ == 'max':
                return lambda x: -self.model.call(outcome_name)(x)
            elif type_ == 'min:':
                return lambda x: self.model.call(outcome_name)(x)
            else:
                return lambda x: self.model.call(outcome_name)(x)

        # Update lever values based on x
        for i, lever_name in enumerate(self.model.lever_names):
            self.model.element(lever_name).value = x[i]



        # Calculate and return the outcomes
        if self.types == 'max':
            outcomes = [-self.model.call(outcome_name)(x) for outcome_name in self.outcome_names]
        elif self.types == 'min':
            outcomes = [self.model.call(outcome_name)(x) for outcome_name in self.outcome_names]
        elif isinstance(self.types, list):
            outcomes = [model_call(outcome_name, type_, x) for outcome_name, type_ in zip(self.outcome_names, self.types)]
        else:
            outcomes = [self.model.call(outcome_name)(x) for outcome_name in self.outcome_names]
        out["F"] = np.array(outcomes)
