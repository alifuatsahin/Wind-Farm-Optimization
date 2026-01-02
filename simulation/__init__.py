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

    def plot_single_turbine(self, turbine_index=0, show_streamwise=True, save_graphic=False, save_at_x=None):
        t = self.wind_farm.turbines[turbine_index]
        save_path = self.config.out_path if save_graphic else None
        plot_wake_field = t.wake_field[::max(1, len(t.wake_field)//100)]  # limit to 100 frames for plotting
        plot_data(plot_wake_field, t, show_streamwise=show_streamwise, save_path=save_path, save_at_x=save_at_x)

    def plot_wind_farm_wake(self, save_graphic=False):
        save_path = self.config.out_path if save_graphic else None
        plot_farm_deficit_map(self.wind_farm, save_path=save_path)

    def run(self):
        self.wind_farm.solve()