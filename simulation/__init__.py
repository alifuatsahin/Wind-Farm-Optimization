from .utils import plot_data, plot_farm_deficit_map
from .core_types import WindFarm

import os

class Simulation:
    def __init__(self, config):
        self.config = config
        self.wind_farm = WindFarm(config)

    def save_results(self):
        # Ensure output directory exists
        os.makedirs(self.config.out_path, exist_ok=True)
        # save results (pickle for python)
        self.wind_farm.save_results(self.config.out_path)

    def plot_single_turbine(self, turbine_index=0, show_streamwise=True, save_graphic=False):
        t = self.wind_farm.turbines[turbine_index]
        save_path = self.config.out_path if save_graphic else None
        plot_data(t.wake_field, t, show_streamwise=show_streamwise, save_path=save_path)

    def plot_wind_farm_wake(self, save_graphic=False):
        save_path = self.config.out_path if save_graphic else None
        plot_farm_deficit_map(self.wind_farm, save_path=save_path)

    def run(self):
        self.wind_farm.simulate_turbine_vortex_fields()
        self.wind_farm.solve_streamwise()