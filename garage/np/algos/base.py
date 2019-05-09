class Algorithm:
    pass


class RLAlgorithm(Algorithm):
    def train(self):
        raise NotImplementedError

    def process_samples(self, itr, paths):
        """
        Return processed sample data (typically a dictionary of concatenated
        tensors) based on the collected paths.
        :param itr: Iteration number.
        :param paths: A list of collected paths.
        :return: Processed sample data.
        """
        raise NotImplementedError
