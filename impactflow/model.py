import numpy as np
from scipy.optimize import basinhopping
from SALib.sample import saltelli
from SALib.analyze import sobol
import networkx as nx
from element import DecisionElement, DecisionHead, DecisionTail, Lever, Outcome, External


class CausalDecisionModel:
    def __init__(self, levers: list[Lever]):
        """
        Initialize the causal decision model with a directed graph and add given levers as nodes.

        :param levers: A list of Lever objects to be added to the model.
        """
        self.graph = nx.DiGraph()
        self.add_elements(levers)

    @property
    def levers(self) -> list[Lever]:
        result = [attrs["element"] for node, attrs in self.graph.nodes(data=True) if
                  isinstance(attrs["element"], Lever)]
        return result

    @property
    def lever_names(self) -> list[str]:
        levers = self.levers
        return [lever.name for lever in levers]

    @property
    def externals(self):
        result = [attrs["element"] for node, attrs in self.graph.nodes(data=True) if
                  isinstance(attrs["element"], External)]
        return result

    @property
    def external_names(self) -> list[str]:
        levers = self.levers
        return [lever.name for lever in levers]

    def add_element(self, element: DecisionElement):
        """
        Add a single element (node) to the model's graph. If the element is a DecisionTail,
        add edges from its inputs to it, representing dependencies.

        :param element: A DecisionElement instance to add to the graph.
        """
        self.graph.add_node(element.name, element=element)
        if isinstance(element, DecisionTail):
            for input in element.inputs:
                self.graph.add_edge(input, element.name)

    def add_elements(self, elements: list[DecisionElement]):
        """
        Add multiple elements (nodes) to the model's graph by iterating through a list
        of DecisionElement instances and adding each one.

        :param elements: A list of DecisionElement instances to add to the graph.
        """
        for element in elements:
            self.add_element(element)

    def element(self, name: str):
        """
        Retrieve an element from the graph by its name.

        :param name: The name of the element to retrieve.
        :return: The DecisionElement instance associated with the given name.
        """
        return self.graph.nodes[name]["element"]

    def call(self, outcome_name: str):
        """
        Create and return a callable function that computes the value of a specified outcome
        based on the current values of levers, passing arguments as a numpy array.

        :param outcome_name: The name of the outcome whose value is to be computed.
        :return: A function that takes a numpy array of lever values as input and returns the outcome value.
        """
        # Check if the outcome exists in the model
        if outcome_name not in self.graph:
            raise ValueError(f"Outcome {outcome_name} does not exist in the model.")

        # Identify all levers that feed into the outcome, directly or indirectly
        levers = self.levers

        # Ensure the order of levers matches the order of arguments in the returned function
        lever_names = [lever.name for lever in levers]

        def outcome_function(args: np.ndarray):
            """
            The callable function to compute the outcome value. It updates lever values
            based on the provided numpy array before computation.

            :param args: A numpy array containing lever values.
            :return: The computed value of the specified outcome.
            """
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

    def optimize(self, name: str, sign: str = "pos", niter: int = 100, random_seed: int = 42):
        """
        Optimize the values of levers to maximize or minimize a specified outcome using
        the basinhopping algorithm.

        :param name: The name of the outcome to optimize.
        :param sign: The direction of optimization ('pos' for maximization, 'neg' for minimization).
        :param niter: The number of iterations for the basinhopping algorithm.
        :param random_seed: The seed for the random number generator used in basinhopping.
        :return: The optimized value of the specified outcome.
        """
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
        res = basinhopping(outcome_func, x0, niter=niter, minimizer_kwargs=minimizer_kwargs, seed=random_seed)
        x = res.x
        # set value
        for value, lever in zip(x, levers):
            lever.value = value
        return self.call(name)(x)

    def sensitivity(self, outcome_name: str, n_samples=1024):
        """
        Conducts a sensitivity analysis on the specified outcome using the SALib library. This
        method evaluates the first-order and total effect sensitivity indices for each lever in
        the model to understand their impact on the outcome.

        The sensitivity analysis is performed using the Sobol sensitivity analysis method, which
        is suitable for nonlinear, non-monotonic, and high-dimensional model spaces. It provides
        a quantitative measure of how much each input lever contributes to the output variability.

        Parameters:
        - outcome_name (str): The name of the outcome variable for which the sensitivity analysis
                              is to be performed.
        - n_samples (int): The number of samples to generate for the analysis. A higher number of
                           samples will lead to more accurate estimates but increase computation time.

        Returns:
        - S1 (numpy.ndarray): The first-order Sobol sensitivity indices for each lever, indicating
                              the direct contribution of each input to the variance of the output.
        - ST (numpy.ndarray): The total effect sensitivity indices for each lever, capturing both
                              the direct effects and all higher-order interactions involving the input.

        The method defines the problem for SALib, generates parameter values using the Saltelli
        sampling scheme, evaluates the model output for each sample, and then performs the Sobol
        sensitivity analysis on the collected outputs.

        Note:
        - The 'bounds' for each lever are extracted directly from the model elements.
        - The outcome function is assumed to require all lever values as inputs for the evaluation.
        - The Sobol analysis method is used to calculate both first-order and total effect indices,
          providing insights into the importance of each lever and their interactions.
        """

        lever_names = self.lever_names
        # Define the problem for SALib
        problem = {
            'num_vars': len(lever_names),
            'names': lever_names,
            'bounds': [self.element(lever_name).bounds for lever_name in lever_names]
        }

        # Generate samples
        param_values = saltelli.sample(problem, n_samples)

        # Placeholder for model outputs
        y = np.zeros(param_values.shape[0])

        # Evaluate the model for each sample
        for i, X in enumerate(param_values):
            temp_values = {name: value for name, value in zip(lever_names, X)}
            # Assume all levers' values are required for the outcome function
            outcome_function = self.call(outcome_name)
            y[i] = outcome_function(np.array([temp_values[lever_name] for lever_name in lever_names if lever_name in temp_values]))

        # Perform analysis
        Si = sobol.analyze(problem, y)

        # SI contains:
        # S1 - first-order Sobol sensitivity indices
        # ST - total effect sensitiity induces
        # S2 - second-order Sobol sensitivity indices
        # pull out the first order and second order effects
        S1 = Si['S1']
        ST = Si['ST']
        return S1, ST

