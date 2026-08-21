from dataclasses import dataclass
import numpy as np

@dataclass
class Config:
    def __init__(self):
            # Physical parameters from Task 1/2
            self.rho = 1100; self.cp = 1900; self.k = 0.2
            self.Hr = 2.5e5; self.h = 200; self.Apre = 5e5
            self.E = 8e4; self.R = 8.314; self.m = 1.2; self.n = 1.8
            self.Ts = 300; self.T0 = 300; self.Tm = 443; self.L = 0.02
            self.alpha_initial = 0.001; self.decrease_factor = 0.5
            self.max_temp = 0
            
            # Dimensionless Parameters
            self.t_diff = (self.rho * self.cp * self.L**2) / self.k
            self.Ex = self.Hr / (self.cp * (self.Tm - self.T0))
            self.Bi = (self.h * self.L) / self.k
            self.Da = self.t_diff * self.Apre
            self.gamma = self.E / (self.R * self.Tm)
            self.epsilon = (self.Tm - self.T0) / self.Tm
            self.delta = (self.rho * self.Hr * self.Apre * self.L**2 * self.E) / (self.k * self.R * self.Tm**2) * np.exp(-self.gamma)

    def update_constants(self):
        self.t_diff = (self.rho * self.cp * self.L**2) / self.k
        self.Ex = self.Hr / (self.cp * (self.Tm - self.T0))
        self.Bi = (self.h * self.L) / self.k
        self.Da = self.t_diff * self.Apre
        self.gamma = self.E / (self.R * self.Tm)
        self.epsilon = (self.Tm - self.T0) / self.Tm
        self.delta = (self.rho * self.Hr * self.Apre * self.L**2 * self.E) / (self.k * self.R * self.Tm**2) * np.exp(-self.gamma)
         
    def update_alpha(self, u, alpha):
        us = (self.Ts - self.Tm)/(self.Tm - self.T0)
        mask = (u > us)
        g = np.zeros_like(u)

        if not np.any(mask):
            return g
        
        g[mask] = self.Da * np.exp(-self.gamma/(1 + self.epsilon * u[mask])) * \
        (alpha[mask] ** self.m) * ((1.0 - alpha[mask]) ** self.n)
        
        return g