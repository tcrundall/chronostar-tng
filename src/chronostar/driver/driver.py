import yaml


class Driver:
    def __init__(
        self,
        config_file,
        mixture_class,
        icpool_class,
        introducer_class,
        component_class,
    ) -> None:
        """Constructor method"""

        self.config_params = self.read_config_file(config_file)

        self.component_class = component_class
        self.mixture_class = mixture_class
        self.icpool_class = icpool_class
        self.introducer_class = introducer_class

    def run(self, data):

        icpool = self.icpool_class(
            config_params=self.config_params['icpool'],
            introducer_class=self.introducer_class,
            component_class=self.component_class,
        )

        for unique_id, init_conds in icpool.pool():
            m = self.mixture_class(self.config_params)
            m.set_params(init_conds)
            m.fit(data)
            icpool.register_result(unique_id, m, m.bic(data))

        # loop will end when icg stops generating reasonable initial conditions
        best_mixture = icpool.best_mixture()

        return best_mixture, best_mixture.memberships

    def read_config_file(self, config_file: str) -> dict:
        with open(config_file, "r") as stream:
            try:
                config_params = yaml.safe_load(stream)
            except yaml.YAMLError as exc:
                raise exc

        return config_params
