class ResultsHolder:
    """
    Holds and manages the results obtained during an episode.
    """

    def __init__(self):
        self.results = dict()

    def after_step(self, infos):
        if 'metrics' in infos[0]:
            self.results.update(**infos[0]['metrics'])

    def get_final(self):
        return self.results

    def __repr__(self):
        return str(self.get_final())
