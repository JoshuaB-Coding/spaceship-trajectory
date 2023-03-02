import numpy as np

class Orbit:
    def __init__(self, r0, v0) -> None:
        self.r0 = r0 # au
        self.v0 = v0 # year
        self.h = np.linalg.norm(np.cross(r0, v0)) # au^2/year
        self.mu = 39.4769 # au^3/year^2
        # self.e =

# Fixed constants
a_T0 = 1/3 * 10**(-4) # m/s^2

def a_T(r, v):
    return a_T0 * (1 / r)**2 * v / np.linalg.norm(v)

class Kepler:
    def __init__(self, orbit) -> None:
        self.r = np.array([orbit.r0])
        self.v = np.array([orbit.v0])
        self.theta = np.linspace(0, 2*np.pi, 100)

    def calculate_trajectory(self) -> None:
        # r = h / mu * 1 / (1 + e * cos(theta))
        pass


class Encke:
    def __init__(self, orbit) -> None:
        self.r = np.array([orbit.r0])
        self.v = np.array([orbit.v0])
        self.dt = 0.1 # year
        self.r0_norm = np.linalg.norm(orbit.r0)