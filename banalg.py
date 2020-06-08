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

class HardConstraint(AugConBan):
    """
    Bandit with Hard Constraint
    Optimizing (f - l)^2 + (g - l)^2 s.t. f=g
    """
    def __init__(self, T, n, dF, dG, lamb=1E-2, lambF=0, lambG=0, usePosthoc = False):
        super().__init__(T, n, dF, dG, lamb, lambF, lambG)
        self.label = "lamb=%.2f, Hard Constraint" % (self.lamb)


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
        if not self.usePosthoc:
            AGinv = np.linalg.inv(self.AGfull)
            A = self.AF[arm, :] + self.AFG @ AGinv @ self.AG[arm, :] @ AGinv @ self.AFG.T
            B = self.bF[arm, :] + self.AFG @ AGinv @ self.bG[arm, :]
            self.phiF[arm, :] = np.linalg.solve(A, B)
        # phiG regression
        else:
            AFinv = np.linalg.inv(self.AFfull)
            A = self.AG[arm, :] + self.AFG.T @ AFinv @ self.AF[arm, :] @ AFinv @ self.AFG
            B = self.bG[arm, :] + self.AFG.T @ AFinv @ self.bF[arm, :]
            self.phiF[arm, :] = np.linalg.solve(A, B)

class LinUCB(HardConstraint):
    """
        LinUCB with Augmented Contextual Bandit
    """

    def __init__(self, T, n, dF, dG, useHC=True, useHCBound=True):
        super().__init__(T, n, dF, dG, 1E7)
        self.label = "LinUCB"
        self.hcBound = useHC or useHCBound
        self.hc = useHC
        if self.hc:
            self.label += ", HC"
        if self.hcBound:
            self.lable = ", HCBound"


        self.delta = 1.0 / self.T
        self.Ainv = np.array([np.eye(dF) for i in range(n)]) / self.lamb
        self.counts = np.zeros(n)

    def choice(self, t, context):
        n = self.n
        d = self.dF
        lamb = 1E-2
        delta = self.delta
        context = context.reshape((d, 1))

        # Initial Estimate
        ret = (self.phiF @ context).flatten()

        # UCB Calculations
        for arm in range(n):
            # m2 is upper bound on ||theta^*|| = 1
            # L is upper bound on ||a|| = 1
            radical = 2.0 * np.log(1.0 / delta) + d * np.log((d*lamb + self.counts[arm])/(d*lamb))
            assert radical > 0
            beta = np.sqrt(lamb) + np.sqrt(radical)

            # Calculation of ||context||_ainv
            norm = context.T @ self.Ainv[arm, :, :] @ context

            ret[arm] += beta * norm
        

        return np.argmax(ret)

    def update(self, context, arm, reward, posthoc):
        # Up count
        self.counts[arm] += 1

        if self.hc:
            # Hard Constraint Learning
            super().update(context, arm, reward, posthoc)
        else:
            # Record Context
            self.AF[arm, :] += np.outer(context, context)
            self.AFfull += np.outer(context, context)
            self.bF[arm, :] += reward * context

            # Normal Ridge Regression
            self.phiF[arm, :] = np.linalg.solve(self.AF[arm, :], self.bF[arm, :])

        if self.hcBound:
            # Hard Constraint Bound
            AGinv = np.linalg.inv(self.AGfull)
            for arm in range(self.n):
                self.Ainv[arm, :, :] = np.linalg.inv(self.AF[arm, :] + self.AFG @ AGinv @ self.AG[arm, :] @ AGinv @ self.AFG.T)
        else:
            # Normal Bound
            self.Ainv[arm, :, :] = np.linalg.inv(self.AF[arm, :])


class EGreedy(LinUCB):
    """
        Greedy with Augmented Contextual Bandit
    """
    def __init__(self, T, n, dF, dG, eps=0.0, useHC=True):
        super().__init__(T, n, dF, dG, useHC, False)
        self.label = "%f-greedy" % eps
        self.eps = eps
        if self.hc:
            self.label += ", HC"

    def choice(self, t, context):
        if np.random.random() > self.eps:
            return self.greedy(context, None)

        return np.random.randint(self.n)