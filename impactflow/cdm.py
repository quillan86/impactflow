import numpy as np
from SALib.sample import saltelli
from SALib.analyze import sobol
from typing import Union
import networkx as nx
from pymoo.optimize import minimize
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from element import DecisionElement, DecisionHead, DecisionTail, Lever, Outcome, External
from optimize import MultiObjectiveProblem


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
    def outcomes(self):
        result = [attrs["element"] for _, attrs in self.graph.nodes(data=True) if
                    isinstance(attrs["element"], Outcome)]
        return result

    @property
    def external_names(self) -> list[str]:
        levers = self.levers
        return [lever.name for lever in levers]

    @property
    def fidelity(self):
        elements = [attrs["element"] for _, attrs in self.graph.nodes(data=True) if
                    isinstance(attrs["element"], DecisionTail)]
        result = []
        for element in elements:
            if element.predictor is None:
                result.append(0)
            else:
                result.append(element.predictor.fidelity)
        print(result)
        return np.mean(result)

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

    def call(self, name: str):
        """
        Create and return a callable function that computes the value of a specified outcome
        based on the current values of levers, passing arguments as a numpy array.

        :param outcome_name: The name of the outcome whose value is to be computed.
        :return: A function that takes a numpy array of lever values as input and returns the outcome value.
        """
        # Check if the outcome exists in the model
        if name not in self.graph:
            raise ValueError(f"Outcome {name} does not exist in the model.")

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
            return compute_value(name)

        # Return the callable function
        return outcome_function

    def optimize(self, outcome_name, type: str = 'max', n_generations=100):
        """
        Optimize the values of levers to maximize or minimize a specified outcome using
        the basinhopping algorithm.

        :param name: The name of the outcome to optimize.
        :param sign: The direction of optimization ('pos' for maximization, 'neg' for minimization).
        :param niter: The number of iterations for the basinhopping algorithm.
        :param random_seed: The seed for the random number generator used in basinhopping.
        :return: The optimized value of the specified outcome.
        """
        return self.multi_optimize([outcome_name], types=type, n_generations=n_generations)

    def multi_optimize(self, outcome_names, types: Union[str, list[str]] = 'max', n_generations=100):
        """
        Performs multi-objective optimization on specified outcomes.

        :param outcome_names: List of outcome names to optimize.
        :param type: Whether to maximize (max) or minimize the outcomes.
        :param n_generations: Number of generations for the optimization algorithm.
        """
        bounds = [self.element(lever_name).bounds for lever_name in self.lever_names]
        problem = MultiObjectiveProblem(self, outcome_names, bounds, types=types)

        algorithm = NSGA2(
            pop_size=100,
            n_offsprings=10,
            sampling=FloatRandomSampling(),
            crossover=SBX(prob=0.9, eta=15),
            mutation=PM(eta=20),
            eliminate_duplicates=True
        )

        res = minimize(problem,
                       algorithm,
                       ('n_gen', n_generations),
                       verbose=False)

        return res.X

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

    def validate(self):
        """
        Validates the Causal Decision Model to ensure its integrity and readiness for decision analysis.
        This method performs several critical checks:

        1. Checks that there is at least one Lever in the model. Levers are crucial as they represent
           the decision variables that can be manipulated to influence outcomes.

        2. Verifies the existence of at least one Outcome. Outcomes are the variables of interest that
           the decision model aims to influence or optimize, and their presence is essential for the model's purpose.

        3. Ensures all inputs to DecisionTail elements (which represent decision nodes with dependencies)
           come from existing elements within the model. This check prevents references to undefined elements,
           ensuring the model's logical consistency.

        4. Detects cycles within the directed graph that represents the model. Cycles indicate circular dependencies,
           which are not allowed as they can cause infinite loops and disrupt the model's logic.

        5. Checks for unconnected components within the graph. A fully connected model is necessary to ensure
           that changes in levers can influence all outcomes, making the model coherent and integrated.

        6. Verifies that each outcome is reachable from at least one lever, ensuring the model's
           practicality by confirming that levers can indeed influence outcomes.

        7. Confirms that every lever in the model can potentially affect at least one outcome, ensuring all
           elements of the model contribute to its decision-making capabilities.

        If any of these checks fail, a ValueError is raised with a message explaining the specific issue.
        This method ensures the model's structural and logical integrity before it's used for analysis or optimization.
        """

        # Check for at least one lever
        if len(self.levers) == 0:
            raise ValueError("The model must contain at least one Lever.")

        # Check for at least one outcome
        outcomes = [attrs["element"] for _, attrs in self.graph.nodes(data=True) if isinstance(attrs["element"], Outcome)]
        if len(outcomes) == 0:
            raise ValueError("The model must contain at least one Outcome.")

        # Ensure all inputs come from an existing decision element
        for node, attrs in self.graph.nodes(data=True):
            if isinstance(attrs["element"], DecisionTail):
                for input_name in attrs["element"].inputs:
                    if input_name not in self.graph:
                        raise ValueError(f"Input {input_name} for {node} does not exist in the model.")

        # Detect cycles within the model's graph to ensure no circular dependencies.
        if not nx.is_directed_acyclic_graph(self.graph):
            raise ValueError("The model contains cycles, indicating circular dependencies.")

        if nx.number_weakly_connected_components(self.graph) > 1:
            raise ValueError("The model has unconnected components.")

        # Check for unconnected components within the graph.
        if nx.number_weakly_connected_components(self.graph) > 1:
            raise ValueError("The model has unconnected components.")

        # Check connectivity between levers and outcomes
        for outcome in outcomes:
            paths = nx.single_source_shortest_path(self.graph.reverse(), outcome.name)
            if not any(lever.name in paths for lever in self.levers):
                raise ValueError(f"Outcome {outcome.name} is not reachable from any lever.")

        # Ensure every decision element can potentially influence at least one outcome.
        for lever in self.levers:
            if not any(nx.has_path(self.graph, lever.name, outcome.name) for outcome in self.outcomes):
                raise ValueError(f"Lever {lever.name} does not influence any outcomes.")

        return True
