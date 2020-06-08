#!/usr/bin/python3
from __future__ import print_function

import sys
import numpy as np
import banalg
import matplotlib.pyplot as plt
from mnist import MNIST
from sklearn.decomposition import PCA

from os import path


#=================
# Plot Cumulative Regret
#=================

def plot(title, regret, labels, Trec=10):
    """
    @param title: graph title
    @param regret: T+1 x len(bandits) cumulative regret
    @param labels: label[i] for bandits[i]

    Plots regret curve.
    """
    plt.title(title)
    t = np.arange(regret.shape[0]) * Trec
    for i, l in enumerate(labels):
        plt.plot(t, regret[:, i], label=l)

    T = Trec * (regret.shape[0] - 1)
    plt.xlim(0, T)
    plt.xlabel("Time")
    plt.ylabel("Test Error Rate")
    plt.legend()
    plt.show()


#=================
# General Bandit Alg
#=================

def run(bandits, contexts, rewards, contexts_test, rewards_test, posthocs, postMat, Trec=10):
    """
    @param bandits: list of initialized bandit algorithms
    @param contexts: TxdF of PCA'd MNIST images
    @param rewards: Txn 1-hot labels
    @return floor(T/Trec)+1 x len(bandits) test class error vectors
    """

    # Define constants
    T = 2000
    errors = np.zeros((int(T/Trec)+1, len(bandits)))
    indices = np.arange(contexts.shape[0])
    np.random.shuffle(indices)

    for t in range(T):
        ind = indices[t]
        if t % Trec == 0:
            print("Round: %d" % t)
            for i, bandit in enumerate(bandits):
                err_count = 0.0
                for j in range(contexts_test.shape[0]):
                    ans = bandit.greedy(contexts_test[j, :], postMat @ rewards_test[j, :])
                    if ans != np.argmax(rewards_test[j, :]):
                        err_count += 1.0
                rate = err_count / float(contexts_test.shape[0])
                print("Bandit %d Error Rate %f" % (i, rate))
                errors[int(t/Trec), i] = rate

        # Choose arm for each bandit
        I_t = []
        for bandit in bandits:
            I_t.append(bandit.choice(t, contexts[ind, :]))

        # Update bandits
        for i, bandit in enumerate(bandits):
            bandit.update(contexts[ind, :], I_t[i], rewards[ind, I_t[i]], posthocs[ind, :])

    # Final Error Rate
    print("Final Error Rate Calculation")
    for i, bandit in enumerate(bandits):
        err_count = 0.0
        for j in range(contexts_test.shape[0]):
            ans = bandit.greedy(contexts_test[j, :], postMat @ rewards_test[j, :])
            if ans != np.argmax(rewards_test[j, :]):
                err_count += 1.0
        rate = err_count / float(contexts_test.shape[0])
        print("Bandit %d Error Rate %f" % (i, rate))
        errors[int(T/Trec), i] = rate

    # Finished, return error array
    print("Finished T=%d rounds!" % T)
    return errors

VAL_SIZE = 10000
def gen_context(mndata, dF, trueTest=True):
    print("Loading Training Set...")
    images, labels = mndata.load_training()

    if trueTest:
        print("Loading Test Set...")
        images_test, labels_test = mndata.load_testing()
    else:
        print("Loading Validation Set...")
        images_test = images[len(images) - VAL_SIZE:len(images)]
        images = images[0:len(images) - VAL_SIZE]
        labels_test = labels[len(labels) - VAL_SIZE:len(labels)]
        labels = labels[0:len(labels) - VAL_SIZE]

    # Format labels
    labels = np.array(labels)
    labels_test = np.array(labels_test)
    Ttrain = len(labels)
    Ttest = len(labels_test)
    print("T_train=%d" % Ttrain)
    print("T_val=%d" % Ttest)
    n = labels.max() + 1

    # Create 1-hot rewards
    rewards = np.zeros((Ttrain, n))
    rewards[np.arange(labels.size),labels] = 1
    rewards_test = np.zeros((Ttest, n))
    rewards_test[np.arange(labels_test.size),labels_test] = 1

    # PCA Contexts
    images = np.array(images)
    images_test = np.array(images_test)

    print("Performing PCA...")
    pca = PCA(n_components=dF)
    contexts = pca.fit_transform(images)
    contexts_test = pca.transform(images_test)
    assert contexts.shape == (Ttrain, dF)
    assert contexts_test.shape == (Ttest, dF)

    return contexts, rewards, contexts_test, rewards_test

def gen_posthoc(dG, rewards):
    # Generate a random invertible matrix reward[i, :] = A * posthocs[i, :]
    # (n x T) = (n x dg) * (dg x T)
    T, n = rewards.shape

    # Generate random matrix
    A = np.random.rand(n, dG)

    # Make sure it inverts
    Ainv = np.linalg.inv(A.T @ A) @ A.T
    assert Ainv.shape == (dG, n)

    posthocs = (Ainv @ rewards.T).T
    assert posthocs.shape == (T, dG)
    return posthocs, Ainv



def main(dF, dG):
    cacheFile = "cache_"+str(dF)+".npz"

    if path.exists(cacheFile):
        print("Loading from Cache...")
        with np.load(cacheFile) as data:
            contexts = data["contexts"]
            rewards = data["rewards"]
            contexts_test = data["contexts_test"]
            rewards_test = data["rewards_test"]
    else:
        # Import MNIST
        print("Loading MNIST...")
        mndata = MNIST('./mnist')
        mndata.gz = True

        # Load Contexts / Rewards
        contexts, rewards, contexts_test, rewards_test = gen_context(mndata, dF)
        print("Saving Cache...")
        np.savez_compressed(cacheFile, contexts=contexts, rewards=rewards, contexts_test=contexts_test, rewards_test=rewards_test)

    T, n = rewards.shape

    # Best Linear Fit Possible
    print("Best Linear Fit...")
    print("Generating Matrices")
    A = np.eye(dF)
    b = np.zeros((dF, n))
    for t in range(contexts.shape[0]):
        A += np.outer(contexts[t, :], contexts[t, :])
        b += np.outer(contexts[t, :], rewards[t, :])
    assert A.shape == (dF, dF)
    assert b.shape == (dF, n)
    print("Doing Fit")
    phi = np.linalg.solve(A, b)
    assert phi.shape == (dF, n)

    print("Testing...")
    err_count = 1.0
    for i in range(contexts_test.shape[0]):
        ans = contexts_test[i, :] @ phi
        assert ans.size == n
        if np.argmax(ans) != np.argmax(rewards_test[i, :]):
            err_count += 1.0

    print("Error Rate: " + str(err_count / float(contexts_test.shape[0])))
    

    # Generate Post-Hoc Contexts
    posthocs, postMat = gen_posthoc(dG, rewards)

    # Define bandits
    bandits = []
    # Test 1: L2 Regularization, no G
    #for lamb in [1E5, 1E6, 1E7, 1E8, 1E9, 1E10]:
    #    bandits.append(banalg.AugConBan(T, n, dF, dG, lamb, 0, 0, False))
    # Results: Best is 1E7

    # Test 2: L2 Regularization, only G
    #for lamb in [1E-4, 1E-3, 1E-2, 1E-1]:
    #    bandits.append(banalg.AugConBan(T, n, dF, dG, lamb, 0, 0, True))
    # Result: Best is 0 (leave minimial in class directly)

    # Test 3: Test cross-regularization when lambG=0 (i.e. trust completely)
    #for lambF in [0, 1E-2, 1E-1, 1E0, 1E1, 1E2]:
    #    bandits.append(banalg.AugConBan(T, n, dF, dG, 1E7, lambF, 0, False))
    # Result: Best is lambF=1E0

    # Test 4: Test Set Performance
    #bandits.append(banalg.AugConBan(T, n, dF, dG, 1E7, 0, 0, False))
    #bandits.append(banalg.AugConBan(T, n, dF, dG, 1E7, 1E0, 0, False))

    # Test 5: SEt lambF = lambG
    # for lamb in [0, 1E-2, 1E-1, 1E0, 1E1, 1E2]:
    #    bandits.append(banalg.AugConBan(T, n, dF, dG, 1E7, lamb, lamb, False))

    # Test 6: Test Set Performance + Hard Constraint
    bandits.append(banalg.AugConBan(T, n, dF, dG, 1E7, 0, 0, False))
    bandits.append(banalg.AugConBan(T, n, dF, dG, 1E7, 1E0, 0, False))
    bandits.append(banalg.HardConstraint(T, n, dF, dG, 1E7, 0, 0, False))

    # Run experiment
    print("Running Experiment...")
    errors = run(bandits, contexts, rewards, contexts_test, rewards_test, posthocs, postMat)

    # Plot Cumulative Regret
    labels = []
    for bandit in bandits:
        labels.append(bandit.label)

    title = "Augmented Contextual Bandit"
    plot(title, errors, labels)

if __name__ == '__main__':
    if len(sys.argv) != 3:
        print("Usage: main.py <dF> <dG>")
        sys.exit(-1)
    main(int(sys.argv[1]), int(sys.argv[2]))
