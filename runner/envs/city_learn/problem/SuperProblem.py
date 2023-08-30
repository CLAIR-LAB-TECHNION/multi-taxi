from ai_dm.base.problem import Problem


class SuperProblem(Problem):
    """
    Initialize a new SuperProblem instance.

    Args:
        initial_state: The initial state of the problem.
        constraints: A list of constraints for the problem.
        stochastic: A boolean indicating whether the problem is stochastic.
    """
    def __init__(self, initial_state, constraints, stochastic=False):
        super().__init__(initial_state, constraints, stochastic)

    def summary(self):
        """
        Retrieve the result of the problem.

        Raises:
            NotImplemented: This method must be implemented by subclasses.
        """
        raise NotImplemented()

    def evaluation_criteria(self):
        raise NotImplemented()
