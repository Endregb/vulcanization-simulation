from dataclasses import dataclass

@dataclass
class Config:
    def __init__(self, dt, t_final, M):
            # Physical parameters from Task 1/2
            self.rho = 1100; self.cp = 1900; self.k = 0.2
            self.Hr = 2.5e5; self.h = 200; self.Apre = 5e5
            self.E = 8e4; self.R = 8.314; self.m = 1.2; self.n = 1.8
            self.Ts = 393; self.T0 = 300; self.Tm = 443; self.L = 0.02
            
            # Dimensionless Parameters
            self.t_diff = (self.rho * self.cp * self.L**2) / self.k
            # self.beta = self.Hr / (self.cp * (self.Tm - self.T0))
            self.Bi = (self.h * self.L) / self.k
            self.params = {"alpha": 1e-2}

            # Endre sine paramaters
            self.t_final = t_final
            self.dt = dt    # timestep
            self.M = M      # number of intervals
            self.N = M+1    # number of points along one axis
            self.U_threshold = (self.Ts - self.Tm) / (self.Tm - self.T0)

            self.gamma = self.E / (self.R*self.Tm)
            self.epsilon = (self.Tm - self.T0) / self.Tm
            self.D = self.Apre * self.t_diff       # NB! suspect t_diff is wrong
            self.S = self.Hr / (self.cp * (self.Tm - self.T0))
