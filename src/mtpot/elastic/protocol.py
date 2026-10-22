"""thermo_pw scf_elastic_constants strain protocol (24 works, see CLAUDE.md)."""
import numpy as np

STRAIN_VALUES = (-0.0075, -0.0025, 0.0025, 0.0075)
# tensor component strained in each block of 4 works: xx, yy, zz, xy, xz, yz
COMPONENTS = ((0, 0), (1, 1), (2, 2), (0, 1), (0, 2), (1, 2))
N_WORKS = len(COMPONENTS) * len(STRAIN_VALUES)


def thermo_pw_strain(work):
    """Symmetric 3x3 strain tensor of thermo_pw work 1..24 (shear: eps_ij = eps_ji = value)."""
    if not 1 <= work <= N_WORKS:
        raise ValueError(f"work must be in 1..{N_WORKS}, got {work}")
    i, j = COMPONENTS[(work - 1) // len(STRAIN_VALUES)]
    e = np.zeros((3, 3))
    e[i, j] = e[j, i] = STRAIN_VALUES[(work - 1) % len(STRAIN_VALUES)]
    return e
