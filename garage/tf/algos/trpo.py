from garage.tf.algos.npo import NPO
from garage.tf.optimizers import ConjugateGradientOptimizer
from garage.tf.optimizers import PenaltyLbfgsOptimizer


class TRPO(NPO):
    """
    Trust Region Policy Optimization.

    See https://arxiv.org/abs/1502.05477.
    """

    def __init__(self,
                 kl_constraint='hard',
                 optimizer=None,
                 optimizer_args=None,
                 **kwargs):
        if not optimizer:
            if kl_constraint == 'hard':
                optimizer = ConjugateGradientOptimizer
            elif kl_constraint == 'soft':
                optimizer = PenaltyLbfgsOptimizer
            else:
                raise NotImplementedError('Unknown KLConstraint')

        if optimizer_args is None:
            optimizer_args = dict()

        super(TRPO, self).__init__(
            pg_loss='surrogate',
            optimizer=optimizer,
            optimizer_args=optimizer_args,
            name='TRPO',
            **kwargs)
