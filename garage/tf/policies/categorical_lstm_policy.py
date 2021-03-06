from akro.tf import Discrete
import numpy as np
import tensorflow as tf

from garage.core import Serializable
from garage.misc.overrides import overrides
from garage.tf.core import LayersPowered
import garage.tf.core.layers as L
from garage.tf.core.network import LSTMNetwork
from garage.tf.distributions import RecurrentCategorical
from garage.tf.misc import tensor_utils
from garage.tf.policies.base import StochasticPolicy


class CategoricalLSTMPolicy(StochasticPolicy, LayersPowered, Serializable):
    def __init__(self,
                 env_spec,
                 name='CategoricalLSTMPolicy',
                 hidden_dim=32,
                 hidden_nonlinearity=tf.nn.tanh,
                 hidden_w_init=L.XavierUniformInitializer(),
                 recurrent_nonlinearity=tf.nn.sigmoid,
                 recurrent_w_x_init=L.XavierUniformInitializer(),
                 recurrent_w_h_init=L.OrthogonalInitializer(),
                 output_nonlinearity=tf.nn.softmax,
                 output_w_init=L.XavierUniformInitializer(),
                 feature_network=None,
                 prob_network=None,
                 state_include_action=True,
                 forget_bias=1.0,
                 use_peepholes=False,
                 lstm_layer_cls=L.LSTMLayer):
        """
        :param env_spec: A spec for the env.
        :param hidden_dim: dimension of hidden layer
        :param hidden_nonlinearity: nonlinearity used for each hidden layer
        :return:
        """
        assert isinstance(env_spec.action_space, Discrete)

        self._prob_network_name = 'prob_network'
        with tf.variable_scope(name, 'CategoricalLSTMPolicy'):
            Serializable.quick_init(self, locals())
            super(CategoricalLSTMPolicy, self).__init__(env_spec)

            obs_dim = env_spec.observation_space.flat_dim
            action_dim = env_spec.action_space.flat_dim

            if state_include_action:
                input_dim = obs_dim + action_dim
            else:
                input_dim = obs_dim

            l_input = L.InputLayer(shape=(None, None, input_dim), name='input')

            if feature_network is None:
                feature_dim = input_dim
                l_flat_feature = None
                l_feature = l_input
            else:
                feature_dim = feature_network.output_layer.output_shape[-1]
                l_flat_feature = feature_network.output_layer
                l_feature = L.OpLayer(
                    l_flat_feature,
                    extras=[l_input],
                    name='reshape_feature',
                    op=lambda flat_feature, input: tf.reshape(
                        flat_feature,
                        tf.stack([
                            tf.shape(input)[0],
                            tf.shape(input)[1], feature_dim
                        ])),
                    shape_op=lambda _, input_shape: (input_shape[
                        0], input_shape[1], feature_dim))

            if prob_network is None:
                prob_network = LSTMNetwork(
                    input_shape=(feature_dim, ),
                    input_layer=l_feature,
                    output_dim=env_spec.action_space.n,
                    hidden_dim=hidden_dim,
                    hidden_nonlinearity=hidden_nonlinearity,
                    hidden_w_init=hidden_w_init,
                    recurrent_nonlinearity=recurrent_nonlinearity,
                    recurrent_w_x_init=recurrent_w_x_init,
                    recurrent_w_h_init=recurrent_w_h_init,
                    output_nonlinearity=output_nonlinearity,
                    output_w_init=output_w_init,
                    forget_bias=forget_bias,
                    use_peepholes=use_peepholes,
                    lstm_layer_cls=lstm_layer_cls,
                    name=self._prob_network_name)

            self.prob_network = prob_network
            self.feature_network = feature_network
            self.l_input = l_input
            self.state_include_action = state_include_action

            flat_input_var = tf.placeholder(
                dtype=tf.float32, shape=(None, input_dim), name='flat_input')
            if feature_network is None:
                feature_var = flat_input_var
            else:
                with tf.name_scope('feature_network', values=[flat_input_var]):
                    feature_var = L.get_output(
                        l_flat_feature,
                        {feature_network.input_layer: flat_input_var})

            with tf.name_scope(self._prob_network_name, values=[feature_var]):
                out_prob_step, out_prob_hidden, out_step_cell = L.get_output(
                    [
                        prob_network.step_output_layer,
                        prob_network.step_hidden_layer,
                        prob_network.step_cell_layer
                    ], {prob_network.step_input_layer: feature_var})

            self.f_step_prob = tensor_utils.compile_function([
                flat_input_var,
                prob_network.step_prev_state_layer.input_var,
            ], [out_prob_step, out_prob_hidden, out_step_cell])

            self.input_dim = input_dim
            self.action_dim = action_dim
            self.hidden_dim = hidden_dim
            self.name = name

            self.prev_actions = None
            self.prev_hiddens = None
            self.prev_cells = None
            self.dist = RecurrentCategorical(env_spec.action_space.n)

            out_layers = [prob_network.output_layer]
            if feature_network is not None:
                out_layers.append(feature_network.output_layer)

            LayersPowered.__init__(self, out_layers)

    @overrides
    def dist_info_sym(self, obs_var, state_info_vars, name=None):
        with tf.name_scope(name, 'dist_info_sym', [obs_var, state_info_vars]):
            n_batches = tf.shape(obs_var)[0]
            n_steps = tf.shape(obs_var)[1]
            obs_var = tf.reshape(obs_var, tf.stack([n_batches, n_steps, -1]))
            obs_var = tf.cast(obs_var, tf.float32)
            if self.state_include_action:
                prev_action_var = state_info_vars['prev_action']
                prev_action_var = tf.cast(prev_action_var, tf.float32)
                all_input_var = tf.concat(
                    axis=2, values=[obs_var, prev_action_var])
            else:
                all_input_var = obs_var
            if self.feature_network is None:
                with tf.name_scope(
                        self._prob_network_name, values=[all_input_var]):
                    prob = L.get_output(self.prob_network.output_layer,
                                        {self.l_input: all_input_var})
                return dict(prob=prob)
            else:
                flat_input_var = tf.reshape(all_input_var,
                                            (-1, self.input_dim))
                with tf.name_scope(
                        self._prob_network_name,
                        values=[all_input_var, flat_input_var]):
                    prob = L.get_output(
                        self.prob_network.output_layer, {
                            self.l_input: all_input_var,
                            self.feature_network.input_layer: flat_input_var
                        })
                return dict(prob=prob)

    @property
    def vectorized(self):
        return True

    def reset(self, dones=None):
        if dones is None:
            dones = [True]
        dones = np.asarray(dones)
        if self.prev_actions is None or len(dones) != len(self.prev_actions):
            self.prev_actions = np.zeros((len(dones),
                                          self.action_space.flat_dim))
            self.prev_hiddens = np.zeros((len(dones), self.hidden_dim))
            self.prev_cells = np.zeros((len(dones), self.hidden_dim))

        self.prev_actions[dones] = 0.
        self.prev_hiddens[dones] = self.prob_network.hid_init_param.eval()
        self.prev_cells[dones] = self.prob_network.cell_init_param.eval()

    @overrides
    def get_action(self, observation):
        actions, agent_infos = self.get_actions([observation])
        return actions[0], {k: v[0] for k, v in agent_infos.items()}

    @overrides
    def get_actions(self, observations):
        flat_obs = self.observation_space.flatten_n(observations)
        if self.state_include_action:
            assert self.prev_actions is not None
            all_input = np.concatenate([flat_obs, self.prev_actions], axis=-1)
        else:
            all_input = flat_obs
        probs, hidden_vec, cell_vec = self.f_step_prob(
            all_input,
            np.concatenate([self.prev_hiddens, self.prev_cells], axis=-1))
        actions = list(map(self.action_space.weighted_sample, probs))
        prev_actions = self.prev_actions
        self.prev_actions = self.action_space.flatten_n(actions)
        self.prev_hiddens = hidden_vec
        self.prev_cells = cell_vec
        agent_info = dict(prob=probs)
        if self.state_include_action:
            agent_info['prev_action'] = np.copy(prev_actions)
        return actions, agent_info

    @property
    @overrides
    def recurrent(self):
        return True

    @property
    def distribution(self):
        return self.dist

    @property
    def state_info_specs(self):
        if self.state_include_action:
            return [
                ('prev_action', (self.action_dim, )),
            ]
        else:
            return []
