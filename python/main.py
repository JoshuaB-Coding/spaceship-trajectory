import numpy as np
import matplotlib.pyplot as plt

class Orbit:
    def __init__(self) -> None:
        self.r0 = np.array([-1.05, 0, 0]) # au
        self.v0 = np.array([0, -6.1316, 0]) # au/year
        self.h_vector = np.cross(self.r0, self.v0) # au^2/year
        self.h = np.linalg.norm(self.h_vector) # au^2/year
        self.mu = 39.4769 # au^3/year^2
        self.p = self.h*self.h / self.mu
        # Lagrange multipliers
        self.e_vector = np.cross(self.v0, self.h_vector) / self.mu - self.r0 / np.linalg.norm(self.r0)
        self.e = np.linalg.norm(self.e_vector)
        self.T = 20 # years

    def trajectory(self, n=10) -> np.array:
        t = np.linspace(0, self.T, n)
        E = self.compute_E(t)
        theta = self.compute_theta(E)
        r = self.compute_r(theta)
        print(theta)
        print(r)

    def compute_E(self, t) -> np.array:
        # Assumes time at periapsis is 0
        constant = np.sqrt(2)
        E = []
        for i in range(len(t)):
            M = constant * (t[i] - 0)
            conv = 1
            E_current = M
            while conv > 1e-5:
                E_next = M + self.e * np.sin(E_current)
                conv = abs(E_current - E_next)
                E_current = E_next
            E += [E_current]
        return np.asarray(E)

    def compute_theta(self, E) -> np.array:
        constant = np.sqrt((1 + self.e) / (1 - self.e))
        return 2 * np.arctan(constant * np.tan(E / 2))

    def compute_r(self, theta) -> np.array:
        return self.p / (1 + self.e * np.cos(theta))

    def lagrange_f(self) -> float:
        pass

    def lagrange_g(self) -> float:
        pass

    

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

orbit = Orbit()
orbit.trajectory(20)