from typing import Union, Callable
from sklearn.base import BaseEstimator, RegressorMixin, ClassifierMixin


class ElementModel:
    def __init__(self, obj: Union[Callable, BaseEstimator], kind: str = "function"):
        """

        :param obj: The model object to pass.
        :param kind:  one of "function", "sklearn", ...
        """
        if kind not in ["function", "sklearn"]:
            raise ValueError("Model does not have a valid kind (function, sklearn, ...)")
        self.kind = kind
        self.obj: Union[Callable, BaseEstimator] = obj
        self.fidelity: int | None = 0
        if isinstance(self.obj, BaseEstimator) and kind != "sklearn":
            raise ValueError("Object passed is a scikit-learn Estimator, but kind is not set to sklearn")
        elif isinstance(self.obj, BaseEstimator) and kind == "sklearn":
            self.fidelity = 3
        elif isinstance(self.obj, Callable) and kind != "function":
            raise ValueError("Object passed is a Callable, but kind is not set to function")
        elif isinstance(self.obj, Callable) and kind == "function":
            self.fidelity = 1

    def predict(self, values: list[int | float]):
        """
        Predict the outcome using the fitted predictor. The function handles both callable functions
        and scikit-learn models.
        """
        if isinstance(self.kind, BaseEstimator):
            values_reshaped = [values]
            prediction = self.obj.predict(values_reshaped)
            return prediction[0]
        else:
            # Calling the callable predictor with unpacked values
            return self.obj(*values)
