from config import Config
from simulation import Simulation

import time
import cProfile
import pstats

def run_simulation(profiling=False):
    config = Config()
    config.print()

    sim = Simulation(config)

    if profiling:
        profiler = cProfile.Profile()
        profiler.enable()
        sim.run()
        profiler.disable()
        stats = pstats.Stats(profiler).sort_stats('cumtime')
        stats.print_stats(20)
    else:
        start = time.perf_counter()
        sim.run()
        end = time.perf_counter()
        print(f"Simulation completed in {end - start:.2f} seconds.")

    # sim.plot_single_turbine(turbine_index=0, show=False, save_graphic=False, save_at_x=[0.1, 1, 2, 4, 8, 10])
    sim.plot_wind_farm_wake(save_graphic=True)
    # sim.save_results()
    sim.calculate_objective(verbose=True)

if __name__ == "__main__":
    run_simulation(profiling=False)