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

def plot(title, regret, labels):
    """
    @param title: graph title
    @param regret: T+1 x len(bandits) cumulative regret
    @param labels: label[i] for bandits[i]

    Plots regret curve.
    """
    plt.title(title)
    T = regret.shape[0]
    t = np.arange(T)
    for i, l in enumerate(labels):
        plt.plot(t, regret[:, i], label=l)

    plt.xlim(0, T)
    plt.xlabel("Time")
    plt.ylabel("Cumulative Regret")
    plt.legend()
    plt.show()


#=================
# General Bandit Alg
#=================

def run(T, bandits, contexts, rewards, contexts_test, rewards_test, posthocs, postMat):
    """
    @param bandits: list of initialized bandit algorithms
    @param contexts: TxdF of PCA'd MNIST images
    @param rewards: Txn 1-hot labels
    @return floor(T/Trec)+1 x len(bandits) test class error vectors
    """

    # Define constants
    cum_regret = np.zeros((T+1, len(bandits)))
    indices = np.arange(contexts.shape[0])
    np.random.shuffle(indices)

    for t in range(T):
        ind = indices[t]
        if t % 10 == 0:
            print("Round: %d" % t)
            for i in range(len(bandits)):
                print("Bandit %d Regret %f" % (i, cum_regret[t, i]))

        # Choose arm for each bandit
        I_t = []
        for bandit in bandits:
            I_t.append(bandit.choice(t, contexts[ind, :]))

        # Update bandits
        for i, bandit in enumerate(bandits):
            # Assume pi_star is perfect, reward=1
            regret = 1 - rewards[ind, I_t[i]]
            cum_regret[t+1, i] = cum_regret[t, i] + regret

            bandit.update(contexts[ind, :], I_t[i], rewards[ind, I_t[i]], posthocs[ind, :])

    # Finished, return error array
    print("Finished T=%d rounds!" % T)
    return cum_regret

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
    
    # Generate Post-Hoc Contexts
    posthocs, postMat = gen_posthoc(dG, rewards)

    # Limit horizon
    Tlimit = 2000

    # Define bandits
    bandits = []

    # Test 7: Greedy normal vs Hard constraint
    bandits.append(banalg.EGreedy(Tlimit, n, dF, dG, 0.0, False))
    bandits.append(banalg.EGreedy(Tlimit, n, dF, dG, 0.0, True))

    # Test 8: epsilon-greedy
    bandits.append(banalg.EGreedy(Tlimit, n, dF, dG, 0.1, False))
    bandits.append(banalg.EGreedy(Tlimit, n, dF, dG, 0.1, True))

    # Run experiment
    print("Running Experiment...")
    regrets = run(Tlimit, bandits, contexts, rewards, contexts_test, rewards_test, posthocs, postMat)

    # Plot Cumulative Regret
    labels = []
    for bandit in bandits:
        labels.append(bandit.label)

    title = "Augmented Contextual Bandit Regret"
    plot(title, regrets, labels)

if __name__ == '__main__':
    if len(sys.argv) != 3:
        print("Usage: main.py <dF> <dG>")
        sys.exit(-1)
    main(int(sys.argv[1]), int(sys.argv[2]))
