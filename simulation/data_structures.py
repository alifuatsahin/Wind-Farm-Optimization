import numpy as np
from dataclasses import dataclass, field

@dataclass
class VortexField:
    Y: np.ndarray = field(default_factory=lambda: np.array([]))      # Vortex Y positions
    Z: np.ndarray = field(default_factory=lambda: np.array([]))      # Vortex Z positions
    Rv: np.ndarray = field(default_factory=lambda: np.array([]))     # Vortex core radii
    Circ: np.ndarray = field(default_factory=lambda: np.array([]))   # Vortex circulations
    yloc: np.ndarray = field(default_factory=lambda: np.array([]))   # For velocity field grid (optional)
    zloc: np.ndarray = field(default_factory=lambda: np.array([]))   # For velocity field grid (optional)
    V: np.ndarray = field(default_factory=lambda: np.array([]))      # Velocity field (optional)
    W: np.ndarray = field(default_factory=lambda: np.array([]))      # Velocity field (optional)
    U: np.ndarray = field(default_factory=lambda: np.array([]))      # Streamwise velocity field (optional)
    OmegaX: np.ndarray = field(default_factory=lambda: np.array([])) # Vorticity field (optional)
    t: float = 0.0
    X: float = 0.0  # streamwise position