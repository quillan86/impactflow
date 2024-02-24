from typing import Callable, Optional, Union


class DecisionElement:
    _type = ""
    def __init__(self, name: str):
        self.name = name
        self.inputs = []  # empty!

    def predict(self, args):
        return NotImplementedError

    def __str__(self):
        return f"{self._type}({self.name})"


# need better names for these.
class DecisionHead(DecisionElement):
    def __init__(self, name: str, value: int | float, bounds: list[int | float]):
        super().__init__(name)
        self.value = value
        self.bounds = bounds[:2] # first two items...


class DecisionTail(DecisionElement):
    def __init__(self, name: str, inputs: list[str]):
        super().__init__(name)
        self.inputs = inputs
        self.predictor: Optional[Union[Callable]] = None # Initialized to None

    def fit(self, predictor: Union[Callable]):
        """
        Fit the DecisionTail with a predictor, which can be a callable function or a scikit-learn model.
        """
        # to do - get this working better?
        self.predictor = predictor
        return self

    def predict(self, args: dict[str, int | float]):
        """
        Predict the outcome using the fitted predictor. The function handles both callable functions
        and scikit-learn models.
        """
        if self.predictor is None:
            raise ValueError("Predictor function not fitted yet.")
        if not all(key in args for key in self.inputs):
            raise ValueError("Not all inputs are provided in args.")

        # Extracting values in the order of self.inputs
        values = [args[input_name] for input_name in self.inputs]

        # Calling the callable predictor with unpacked values
        return self.predictor(*values)


class Outcome(DecisionTail):
    _type = "Outcome"
    def __init__(self, name: str, inputs: list[str]):
        super().__init__(name, inputs)
        self.name = name


class Lever(DecisionHead):
    _type = "Lever"
    def __init__(self, name: str, value: int | float, bounds: list[int | float]):
        super().__init__(name, value, bounds) # first two items...

    def predict(self, *args):
        self.value = args[0]
        return self.value


class External(DecisionHead):
    _type = "External"
    def __init__(self, name: str, value: int | float, bounds: list[int | float]):
        super().__init__(name, value, bounds)

    def predict(self, *args):
        self.value = args[0]
        return self.value


class Intermediate(DecisionTail):
    _type = "Intermediate"
    def __init__(self, name: str, inputs: list[str]):
        super().__init__(name, inputs)
        self.name = name
