from garage.logger import logger, tabular
from garage.np.algos import RLAlgorithm

from garage.misc import special
from garage.tf.misc import tensor_utils


class BatchPolopt(RLAlgorithm):
    """
    Base class for batch sampling-based policy optimization methods.
    This includes various policy gradient methods like vpg, npg, ppo, trpo,
    etc.
    """

    def __init__(self,
                 env_spec,
                 policy,
                 baseline,
                 scope=None,
                 max_path_length=500,
                 discount=0.99,
                 gae_lambda=1,
                 center_adv=True,
                 positive_adv=False,
                 fixed_horizon=False,
                 **kwargs):
        """
        :param env_spec: Environment specification.
        :type env_spec: EnvSpec
        :param policy: Policy
        :type policy: Policy
        :param baseline: Baseline
        :param scope: Scope for identifying the algorithm. Must be specified if
         running multiple algorithms
        simultaneously, each using different environments and policies
        :param max_path_length: Maximum length of a single rollout.
        :param discount: Discount.
        :param gae_lambda: Lambda used for generalized advantage estimation.
        :param center_adv: Whether to rescale the advantages so that they have
         mean 0 and standard deviation 1.
        :param positive_adv: Whether to shift the advantages so that they are
         always positive. When used in conjunction with center_adv the
         advantages will be standardized before shifting.
        :return:
        """
        self.env_spec = env_spec
        self.policy = policy
        self.baseline = baseline
        self.scope = scope
        self.max_path_length = max_path_length
        self.discount = discount
        self.gae_lambda = gae_lambda
        self.center_adv = center_adv
        self.positive_adv = positive_adv
        self.fixed_horizon = fixed_horizon

        self.eprewmean = deque(maxlen=100)

        self.init_opt()

    def train_once(self, itr, paths):
        self.log_diagnostics(paths)
        logger.log('Optimizing policy...')
        self.optimize_policy(itr, paths)
        return paths['average_return']

    def log_diagnostics(self, paths):
        logger.log('Logging diagnostics...')
        self.policy.log_diagnostics(paths)
        self.baseline.log_diagnostics(paths)

    def process_samples(self, itr, paths):
            baselines = []
            returns = []

            max_path_length = self.max_path_length

            if hasattr(self.baseline, 'predict_n'):
                all_path_baselines = self.baseline.predict_n(paths)
            else:
                all_path_baselines = [
                    self.baseline.predict(path) for path in paths
                ]

            for idx, path in enumerate(paths):
                path_baselines = np.append(all_path_baselines[idx], 0)
                deltas = path['rewards'] \
                    + self.discount * path_baselines[1:] \
                    - path_baselines[:-1]
                path['advantages'] = special.discount_cumsum(
                    deltas, self.discount * self.gae_lambda)
                path['deltas'] = deltas

            for idx, path in enumerate(paths):
                # baselines
                path['baselines'] = all_path_baselines[idx]
                baselines.append(path['baselines'])

                # returns
                path['returns'] = special.discount_cumsum(path['rewards'],
                                                          self.discount)
                returns.append(path['returns'])

            # make all paths the same length
            obs = [path['observations'] for path in paths]
            obs = tensor_utils.pad_tensor_n(obs, max_path_length)

            actions = [path['actions'] for path in paths]
            actions = tensor_utils.pad_tensor_n(actions, max_path_length)

            rewards = [path['rewards'] for path in paths]
            rewards = tensor_utils.pad_tensor_n(rewards, max_path_length)

            returns = [path['returns'] for path in paths]
            returns = tensor_utils.pad_tensor_n(returns, max_path_length)

            advantages = [path['advantages'] for path in paths]
            advantages = tensor_utils.pad_tensor_n(advantages, max_path_length)

            baselines = tensor_utils.pad_tensor_n(baselines, max_path_length)

            agent_infos = [path['agent_infos'] for path in paths]
            agent_infos = tensor_utils.stack_tensor_dict_list([
                tensor_utils.pad_tensor_dict(p, max_path_length)
                for p in agent_infos
            ])

            env_infos = [path['env_infos'] for path in paths]
            env_infos = tensor_utils.stack_tensor_dict_list([
                tensor_utils.pad_tensor_dict(p, max_path_length) for p in env_infos
            ])

            valids = [np.ones_like(path['returns']) for path in paths]
            valids = tensor_utils.pad_tensor_n(valids, max_path_length)

            average_discounted_return = (np.mean(
                [path['returns'][0] for path in paths]))

            undiscounted_returns = [sum(path['rewards']) for path in paths]
            self.eprewmean.extend(undiscounted_returns)

            ent = np.sum(
                self.policy.distribution.entropy(agent_infos) *
                valids) / np.sum(valids)

            samples_data = dict(
                observations=obs,
                actions=actions,
                rewards=rewards,
                advantages=advantages,
                baselines=baselines,
                returns=returns,
                valids=valids,
                agent_infos=agent_infos,
                env_infos=env_infos,
                paths=paths,
                average_return=np.mean(undiscounted_returns),
            )

            tabular.record('Iteration', itr)
            tabular.record('AverageDiscountedReturn', average_discounted_return)
            tabular.record('AverageReturn', np.mean(undiscounted_returns))
            tabular.record('Extras/EpisodeRewardMean', np.mean(self.eprewmean))
            tabular.record('NumTrajs', len(paths))
            tabular.record('Entropy', ent)
            tabular.record('Perplexity', np.exp(ent))
            tabular.record('StdReturn', np.std(undiscounted_returns))
            tabular.record('MaxReturn', np.max(undiscounted_returns))
            tabular.record('MinReturn', np.min(undiscounted_returns))

            return samples_data

    def init_opt(self):
        """
        Initialize the optimization procedure. If using tensorflow, this may
        include declaring all the variables and compiling functions
        """
        raise NotImplementedError

    def get_itr_snapshot(self, itr):
        """
        Returns all the data that should be saved in the snapshot for this
        iteration.
        """
        raise NotImplementedError

    def optimize_policy(self, itr, samples_data):
        raise NotImplementedError
