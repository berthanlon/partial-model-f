# system_model.py
# -*- coding: utf-8 -*-
"""
System model
"""
import torch

class SystemModel:

    def __init__(self, F, Q, h, r, T, T_test, outlier_p=0, rayleigh_sigma=10000):

        self.outlier_p = outlier_p
        self.rayleigh_sigma = rayleigh_sigma

        ####################
        ### Motion Model ###
        ####################
        self.F = F
        self.m = self.F.size()[0]
        self.Q = Q

        #########################
        ### Observation Model ###
        #########################
        self.h = h
        self.n = 2

        # r can be a scalar std-dev OR a full covariance matrix R
        if isinstance(r, torch.Tensor) and r.ndim == 2:
            self.R = r.clone()
            self.r = None
        else:
            self.r = float(r)
            self.R = self.r * self.r * torch.eye(self.n)

        self.T = T
        self.T_test = T_test

        self.trajGen = None

    def InitSequence(self, m1x_0, m2x_0):
        self.m1x_0 = m1x_0
        self.m2x_0 = m2x_0

    def UpdateCovariance_Gain(self, q, r):
        self.q = q
        self.Q = q * q * torch.eye(self.m)
        self.r = r
        self.R = r * r * torch.eye(self.n)

    def UpdateCovariance_Matrix(self, Q, R):
        self.Q = Q
        self.R = R

    def GenerateSequence(self, Q_gen, R_gen, T):
        # Preallocate
        self.x = torch.empty(size=[self.m, T])
        self.y = torch.empty(size=[self.n, T])
        self.x_prev = self.m1x_0

        if self.trajGen is not None:
            self.trajGen.generateSequenceTorch()
            traj = self.trajGen.getTrajectoryArraysTorch()
            self.x = traj["X_true"]
            self.y = traj["measurements"]
            self.x_prev = self.m1x_0
            return

    def SetTrajectoryGenerator(self, trajGen) -> None:
        self.trajGen = trajGen

    def GenerateBatch(self, size, T, randomInit=False, seqInit=False, T_test=0):
        print(' T in generate batch', T)
        self.meas = torch.empty(size, self.n, T)
        self.gt = torch.empty(size, self.m, T)

        initConditions = self.m1x_0

        for i in range(0, size):
            if randomInit:
                variance = 100
                initConditions = torch.rand_like(self.m1x_0) * variance
            if seqInit:
                initConditions = self.x_prev
                if (i*T % T_test) == 0:
                    initConditions = torch.zeros_like(self.m1x_0)

            self.InitSequence(initConditions, self.m2x_0)
            self.GenerateSequence(self.Q, self.R, T)

            self.gt[i, :, :] = self.x
            self.meas[i, :, :] = self.y
