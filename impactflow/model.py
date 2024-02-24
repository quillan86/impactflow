import numpy as np
from scipy.optimize import basinhopping
import networkx as nx
from element import DecisionElement, DecisionHead, DecisionTail, Lever, Outcome


class CausalDecisionModel:
    def __init__(self, levers: list[Lever], random_seed=42):
        self.graph = nx.DiGraph()
        self.add_elements(levers)
        self.random_seed = random_seed

    def add_element(self, element: DecisionElement):
        self.graph.add_node(element.name, element=element)
        if isinstance(element, DecisionTail):
            for input in element.inputs:
                self.graph.add_edge(input, element.name)

    def add_elements(self, elements: list[DecisionElement]):
        for element in elements:
            self.add_element(element)

    def element(self, name: str):
        return self.graph.nodes[name]["element"]

    def call(self, outcome_name: str):
        # Check if the outcome exists in the model
        if outcome_name not in self.graph:
            raise ValueError(f"Outcome {outcome_name} does not exist in the model.")

        # Identify all levers that feed into the outcome, directly or indirectly
        levers = [attrs["element"] for node, attrs in self.graph.nodes(data=True) if isinstance(attrs["element"], Lever)]

        # Ensure the order of levers matches the order of arguments in the returned function
        lever_names = [lever.name for lever in levers]

        def outcome_function(args: np.ndarray):
            # Ensure args_array is a NumPy array
            args = np.array(args)

            # Check if the number of elements in args matches the number of levers
            if args.size != len(levers):
                raise ValueError("The size of the input array does not match the number of levers.")

            # Update lever values based on the elements in args
            for arg, lever_name in zip(args, lever_names):
                self.element(lever_name).value = arg

            # Compute the outcome value using the updated lever values
            def compute_value(node_name):
                node = self.element(node_name)
                if isinstance(node, DecisionHead):
                    return node.value
                elif isinstance(node, DecisionTail):
                    inputs = {inp: compute_value(inp) for inp in node.inputs}
                    return node.predict(inputs)
                else:
                    raise ValueError(f"Unsupported node type for {node_name}")

            # Compute and return the outcome
            return compute_value(outcome_name)

        # Return the callable function
        return outcome_function

    def optimize(self, name: str, sign="pos"):
        element = self.element(name)
        if not isinstance(element, Outcome):
            raise ValueError(f"Element {name} is not an Outcome - cannot optimize.")
        if sign == "pos" or sign == "+":
            outcome_func = lambda x: self.call(name)(x)
        elif sign == "neg" or sign == "-":
            outcome_func = lambda x: -self.call(name)(x)
        else:
            outcome_func = lambda x: self.call(name)(x)

        # Identify all levers that feed into the outcome, directly or indirectly
        levers = [attrs["element"] for node, attrs in self.graph.nodes(data=True) if isinstance(attrs["element"], Lever)]

        xmin = [lever.bounds[0] for lever in levers]
        xmax = [lever.bounds[1] for lever in levers]
        bounds = [(low, high) for low, high in zip(xmin, xmax)]
        x0 = np.array([lever.value for lever in levers])
        minimizer_kwargs = dict(method="nelder-mead", bounds=bounds)

        # find global maximum with bounds
        res = basinhopping(outcome_func, x0, niter=100, minimizer_kwargs=minimizer_kwargs, seed=self.random_seed)
        x = res.x
        # set value
        for value, lever in zip(x, levers):
            lever.value = value
        return self.call(name)(x)
