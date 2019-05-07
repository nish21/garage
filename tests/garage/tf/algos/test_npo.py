"""
This script creates a test that fails when garage.tf.algos.NPO performance is
too low.
"""
import gym
import tensorflow as tf

from garage.envs import normalize
from garage.experiment import LocalRunner
from garage.tf.algos import NPO
from garage.tf.baselines import GaussianMLPBaseline
from garage.tf.envs import TfEnv
from garage.tf.policies import GaussianMLPPolicy
from tests.fixtures import TfGraphTestCase


class TestNPO(TfGraphTestCase):
    def test_npo_pendulum(self):
        """Test NPO with Pendulum environment."""
        with LocalRunner(self.sess) as runner:
            env = TfEnv(normalize(gym.make('InvertedDoublePendulum-v2')))
            policy = GaussianMLPPolicy(
                env_spec=env.spec,
                hidden_sizes=(64, 64),
                hidden_nonlinearity=tf.nn.tanh,
                output_nonlinearity=None,
            )
            baseline = GaussianMLPBaseline(
                env_spec=env.spec,
                regressor_args=dict(hidden_sizes=(32, 32)),
            )
            algo = NPO(
                env_spec=env.spec,
                policy=policy,
                baseline=baseline,
                max_path_length=100,
                discount=0.99,
                gae_lambda=0.98,
                policy_ent_coeff=0.0)
            runner.setup(algo, env)
            last_avg_ret = runner.train(n_epochs=10, batch_size=2048)
            assert last_avg_ret > 20

            env.close()

    def test_npo_with_unknown_pg_loss(self):
        """Test NPO with unkown policy gradient loss."""
        env = TfEnv(normalize(gym.make('InvertedDoublePendulum-v2')))
        policy = GaussianMLPPolicy(
            env_spec=env.spec,
            hidden_sizes=(64, 64),
            hidden_nonlinearity=tf.nn.tanh,
            output_nonlinearity=None,
        )
        baseline = GaussianMLPBaseline(
            env_spec=env.spec,
            regressor_args=dict(hidden_sizes=(32, 32)),
        )
        with self.assertRaises(NotImplementedError, msg='Unknown PGLoss'):
            NPO(
                env_spec=env.spec,
                policy=policy,
                baseline=baseline,
                pg_loss='random pg_loss',
            )

        env.close()

    def test_npo_with_invalid_entropy_method(self):
        """Test NPO with invalid entropy method."""
        env = TfEnv(normalize(gym.make('InvertedDoublePendulum-v2')))
        policy = GaussianMLPPolicy(
            env_spec=env.spec,
            hidden_sizes=(64, 64),
            hidden_nonlinearity=tf.nn.tanh,
            output_nonlinearity=None,
        )
        baseline = GaussianMLPBaseline(
            env_spec=env.spec,
            regressor_args=dict(hidden_sizes=(32, 32)),
        )
        with self.assertRaises(ValueError, msg='Invalid entropy_method'):
            NPO(
                env_spec=env.spec,
                policy=policy,
                baseline=baseline,
                entropy_method=None,
            )

        env.close()

    def test_npo_with_invalid_max_entropy_configuration(self):
        """Test NPO with invalid max entropy arguments combination."""
        env = TfEnv(normalize(gym.make('InvertedDoublePendulum-v2')))
        policy = GaussianMLPPolicy(
            env_spec=env.spec,
            hidden_sizes=(64, 64),
            hidden_nonlinearity=tf.nn.tanh,
            output_nonlinearity=None,
        )
        baseline = GaussianMLPBaseline(
            env_spec=env.spec,
            regressor_args=dict(hidden_sizes=(32, 32)),
        )
        with self.assertRaises(AssertionError):
            NPO(
                env_spec=env.spec,
                policy=policy,
                baseline=baseline,
                entropy_method='max',
                center_adv=True,
                stop_entropy_gradient=False,
            )

        env.close()

    def test_npo_with_invalid_no_entropy_configuration(self):
        """Test NPO with invalid no entropy arguments combination."""
        env = TfEnv(normalize(gym.make('InvertedDoublePendulum-v2')))
        policy = GaussianMLPPolicy(
            env_spec=env.spec,
            hidden_sizes=(64, 64),
            hidden_nonlinearity=tf.nn.tanh,
            output_nonlinearity=None,
        )
        baseline = GaussianMLPBaseline(
            env_spec=env.spec,
            regressor_args=dict(hidden_sizes=(32, 32)),
        )
        with self.assertRaises(AssertionError):
            NPO(
                env_spec=env.spec,
                policy=policy,
                baseline=baseline,
                entropy_method='no_entropy',
                policy_ent_coeff=0.02,
            )

        env.close()

    def test_npo_with_invalid_regularized_entropy_configuration(self):
        """Test NPO with invalid regularized entropy arguments combination."""
        env = TfEnv(normalize(gym.make('InvertedDoublePendulum-v2')))
        policy = GaussianMLPPolicy(
            env_spec=env.spec,
            hidden_sizes=(64, 64),
            hidden_nonlinearity=tf.nn.tanh,
            output_nonlinearity=None,
        )
        baseline = GaussianMLPBaseline(
            env_spec=env.spec,
            regressor_args=dict(hidden_sizes=(32, 32)),
        )
        with self.assertRaises(AssertionError):
            NPO(
                env_spec=env.spec,
                policy=policy,
                baseline=baseline,
                entropy_method='regularized',
                policy_ent_coeff=0.02,
                stop_entropy_gradient=True,
            )

        env.close()
