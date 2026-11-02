import numpy as np
import jax
from dataclasses import dataclass, field

@dataclass
class VortexField:
    Y: np.ndarray = field(default_factory=lambda: np.array([]))
    Z: np.ndarray = field(default_factory=lambda: np.array([]))
    Rv: np.ndarray = field(default_factory=lambda: np.array([]))
    Circ: np.ndarray = field(default_factory=lambda: np.array([]))
    age: np.ndarray = field(default_factory=lambda: np.array([]))
    active: np.ndarray = field(default_factory=lambda: np.array([], dtype=bool))
    yloc: np.ndarray = field(default_factory=lambda: np.array([]))
    zloc: np.ndarray = field(default_factory=lambda: np.array([]))
    V: np.ndarray = field(default_factory=lambda: np.array([]))
    W: np.ndarray = field(default_factory=lambda: np.array([]))
    U: np.ndarray = field(default_factory=lambda: np.array([]))
    OmegaX: np.ndarray = field(default_factory=lambda: np.array([]))
    kw: np.ndarray = field(default_factory=lambda: np.array([]))
    wage: np.ndarray = field(default_factory=lambda: np.array([]))
    ti: np.ndarray = field(default_factory=lambda: np.array([]))
    t: float = 0.0
    X: float = 0.0

jax.tree_util.register_dataclass(
    VortexField,
    data_fields=["Y", "Z", "Rv", "Circ", "age", "active", "yloc", "zloc", "V", "W", "U", "OmegaX", "kw", "wage", "ti", "t", "X"],
    meta_fields=[],
)
