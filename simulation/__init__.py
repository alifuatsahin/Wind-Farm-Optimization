import os
import jax

# SINGLE precision by default. Set WFO_FLOAT64=1 to get the old behaviour back, which is
# worth doing whenever a result looks like it might be a rounding artefact -- but it is
# not needed for accuracy, and it is actively harmful on the GPU.
#
# MEASURED against double, on the three cases the model is validated on:
#
#   full-wake normalised power   0.64815825 -> 0.64815789,  0.65805729 -> 0.65805607
#   3x3 farm efficiency          0.57348380 -> 0.57348371
#   8-turbine row, T8/T1         0.69832005 -> 0.69830831
#   8-turbine power_bias/rmse    -0.07717983/0.09844260 -> -0.07719210/0.09845628
#
# i.e. ~1e-5 at worst, on the 8-turbine row, which marches furthest and accumulates the
# most explicit steps. That is four orders below the model's own error and invisible at
# the precision anything is reported to. The march being explicit over thousands of
# sequential steps is exactly where single precision WOULD accumulate, so this was
# measured on the longest case rather than assumed from the short ones.
#
# It also buys 1.16-1.35x on CPU for nothing. The real reason, though, is the GPU: a
# consumer card (this machine has an RTX 5070) runs float64 at 1/64 of float32, so in
# double precision a GPU run takes the slowest path the hardware has and can lose to the
# CPU outright. Data-centre parts are 1/2 and would not care.
jax.config.update("jax_enable_x64", os.environ.get("WFO_FLOAT64", "0") == "1")

from .utils import plot_data, plot_farm_deficit_map, plot_turbine_layout
from .core_types import WindFarm

import os

class Simulation:
    def __init__(self, config):
        self.config = config
        self.wind_farm = WindFarm(config)

    def save_results(self):
        self.config.ensure_run_dir()
        out_path = os.path.join(self.config.out_path, "results")
        os.makedirs(out_path, exist_ok=True)
        self.wind_farm.save_results(out_path, limit_frames=100)

    def plot_single_turbine(self, turbine_index=0, show=True, show_streamwise=True, save_graphic=False, save_at_x=None):
        t = self.wind_farm.turbines[turbine_index]
        if save_graphic:
            self.config.ensure_run_dir()
        save_path = os.path.join(self.config.out_path, "figures") if save_graphic else None
        plot_wake_field = t.wake_field[::max(1, len(t.wake_field)//100)]  # limit to 100 frames for plotting
        plot_data(plot_wake_field, t, show_streamwise=show_streamwise, save_path=save_path, save_at_x=save_at_x, show=show)

    def plot_wind_farm_wake(self, save_graphic=False):
        if save_graphic:
            self.config.ensure_run_dir()
        save_path = os.path.join(self.config.out_path, "figures") if save_graphic else None
        plot_farm_deficit_map(self.wind_farm, save_path=save_path)

    def plot_turbine_layout(self, save_graphic=False):
        if save_graphic:
            self.config.ensure_run_dir()
        save_path = os.path.join(self.config.out_path, "figures") if save_graphic else None
        plot_turbine_layout(self.wind_farm, save_path=save_path)

    def calculate_objective(self, verbose=False):
        return self.wind_farm.calculate_efficiency(verbose=verbose)

    def run(self):
        self.wind_farm.solve()