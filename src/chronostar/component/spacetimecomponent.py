from src.chronostar.component.base import BaseComponent


class SpaceTimeComponent(BaseComponent):
    def __init__(self, config_params):
        self.config_params = config_params

    def maximize(self, X, log_resp):
        pass

    def estimate_log_prob(self, X):
        pass

    @property
    def n_params(self):
        pass
