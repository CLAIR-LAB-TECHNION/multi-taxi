from aidm.environments.gymnasium_envs.gymnasium_problem import GymnasiumProblemS
from aidm.search.best_first_search import breadth_first_search
import aidm.search.utils as utils
from itertools import product

from multi_taxi import multi_taxi_v0, Action
from .common import test_env_cfgs

test_env_cfgs = {k: v for k, v in test_env_cfgs.items() if v.get('observation_type', 'symbolic') == 'symbolic'}


class MultiTaxiProblem(GymnasiumProblemS):
    def sample_applicable_actions_at_state(self, state, sample_size=None):
        action_lists = []
        for agent in self.env.possible_agents:
            action_lists.append(self.env.unwrapped.get_action_meanings(agent).keys())

        # get list of possible joint actions
        possible_joint_actions_tuples = list(product(*action_lists))
        return [{self.env.possible_agents[i]: action
                 for i, action in enumerate(joint_action)}
                for joint_action in possible_joint_actions_tuples]

    def get_action_cost(self, action, state):
        return 1

    def get_successors(self, action, node):
        successor_nodes = []

        # HERE WE USE OUR TRANSITION FUNCTION
        transitions = self.env.unwrapped.state_action_transitions(node.state.key, action)

        action_cost = self.get_action_cost(action, node.state)
        for next_state, rewards, terms, truncs, infos, prob in transitions:
            info = {}
            info['prob'] = prob
            info['reward'] = rewards
            info.update(infos)

            # state is a hashable key
            successor_state = utils.State(key=next_state, is_terminal=all(terms.values()))

            successor_node = utils.Node(state=successor_state,
                                        parent=node,
                                        action=action,
                                        path_cost=node.path_cost + action_cost,
                                        info=info)

            successor_nodes.append(successor_node)

        return successor_nodes


def __get_solution(env):
    mt_problem = MultiTaxiProblem(env, env.state())
    sol_len, final_node, solution, explore_count, terminated = breadth_first_search(mt_problem)
    solution = [eval(action) for action in solution]

    return solution


def __execute_solution(env, solution):
    while solution:
        env.step(solution.pop(0))


def test_sa_search():
    env = multi_taxi_v0.parallel_env(
        num_taxis=1,
        num_passengers=3,
        pickup_only=True
    )
    env.reset()
    solution = __get_solution(env)
    __execute_solution(env, solution)
    assert env.objective_achieved()


def test_ma_search():
    env = multi_taxi_v0.parallel_env(
        num_taxis=2,
        num_passengers=2,
        pickup_only=True,
        render_mode="human"
    )
    env.reset()

    # put taxis relatively close to passengers for reasonable planning time
    s = env.state()
    s.taxis[0].location = (0, 2)
    s.taxis[1].location = (4, 4)
    s.passengers[0].location = (6, 6)
    s.passengers[1].location = (0, 0)
    env.set_state(s)

    solution = __get_solution(env)
    __execute_solution(env, solution)
    assert env.objective_achieved()


def test_failed_search():
    env = multi_taxi_v0.parallel_env(
        num_taxis=1,
        num_passengers=3,
        pickup_only=True,
        max_fuel=2
    )
    env.reset()
    solution = __get_solution(env)
    __execute_solution(env, solution)
    assert not env.objective_achieved()
