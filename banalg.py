#!/usr/bin/python3
from __future__ import print_function

import numpy as np

class AugConBan(object):
    """
    Augmented Contextual Bandit
    Optimizing 1/lambdaF * (f - l)^2 + 1/lambdaG * (g - l)^2 + (f-g)^2
    """
    def __init__(self, T, n, dF, dG, lamb=1E-2, lambF=0, lambG=0, usePosthoc = False):
        self.lambF = lambF # Regularization *away* from F
        self.lambG = lambG # Regularization *away* from G
        self.lamb = lamb # L2 Regularization
        self.n = n
        self.T = T
        self.dF = dF
        self.dG = dG
        self.usePosthoc = usePosthoc

        self.label = "lamb=%.2f, lambF=%.2f, lambG=%.2f" % (lamb, lambF, lambG)
        if usePosthoc:
            self.label = "PH, " + self.label

        # F Linear Regression Params
        self.phiF = np.zeros((n, dF))
        self.AF = np.array([np.eye(dF) for i in range(n)]) * lamb
        self.AFfull = np.eye(dF)
        self.bF = np.zeros((n, dF))

        # G Linear Regression Params, use minimal L2 Regularization
        L2G = 1E-7
        self.phiG = np.zeros((n, dG))
        self.AG = np.array([np.eye(dG) for i in range(n)]) * L2G
        self.AGfull = np.eye(dG)
        self.bG = np.zeros((n, dG))
        self.AFG = np.zeros((dF, dG))

    # Default to Random Choice
    def choice(self, t, context):
        return (t % self.n)

    # Greedy Option, used for test time
    def greedy(self, context, posthoc):
        if self.usePosthoc:
            dist = (self.phiG @ posthoc.reshape((self.dG, 1))).flatten()
        else:
            dist = (self.phiF @ context.reshape((self.dF, 1))).flatten()
        assert len(dist) == self.n

        return np.argmax(dist)

    def update(self, context, arm, reward, posthoc):
        # Record Context
        self.AF[arm, :] += np.outer(context, context)
        self.AFfull += np.outer(context, context)
        self.bF[arm, :] += reward * context

        # Record Posthoc
        self.AG[arm, :] += np.outer(posthoc, posthoc)
        self.AGfull += np.outer(posthoc, posthoc)
        self.bG[arm, :] += reward * posthoc

        # Cross Parameters
        XTP = np.outer(posthoc, context)
        assert XTP.T.shape == (self.dF, self.dG)
        self.AFG += XTP.T

        # phiF regression
        if self.lambF != 0:
            pinv = np.linalg.inv(self.AG[arm, :] + self.lambG * self.AGfull)
            A = self.AF[arm, :] + self.lambF * self.AFfull - self.lambF * self.lambG * self.AFG @ pinv @ self.AFG.T
            b = self.bF[arm, :] + self.lambF * self.AFG @ pinv @ self.bG[arm, :]

            self.phiF[arm, :] = np.linalg.solve(A, b)
        else:
            # Normal Ridge Regression
            self.phiF[arm, :] = np.linalg.solve(self.AF[arm, :], self.bF[arm, :])

        # phiG regression
        self.phiG[arm, :] = np.linalg.solve(self.AG[arm, :] + self.lambG * self.AGfull, self.bG[arm, :] + self.lambG * self.AFG.T @ self.phiF[arm, :])

            