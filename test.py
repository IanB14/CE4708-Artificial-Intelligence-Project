##CE4708 Programming Project, Autumn 2016
##Greg Lynch 12147451
##28/11/16
from math import exp
import numpy as np
import matplotlib.pyplot as plt
from cvxopt import matrix, solvers
from numpy import arange, meshgrid, zeros


def rbfKernel(v1, v2, sigma2=0.25):
    assert len(v1) == len(v2)
    assert sigma2 >= 0.0
    mag2 = sum(map(lambda x, y: (x - y) * (x - y), v1, v2))  ## Squared mag of diff.
    return exp(-mag2 / (2.0 * sigma2))


def rbf2(x, y):
    return rbfKernel(x, y, 0.2)


def makeLambdas(Xs, Ts, C, K=rbfKernel):
    """Solve constrained maximaization problem and return list of l's."""
    P = makeP(Xs, Ts, K)  ## Build the P matrix.
    n = len(Ts)
    q = matrix(-1.0, (n, 1))  # This builds an n-element column
    # vector of -1.0's (note the double-
    # precision constant).
    h = matrix(0.0, (2 * n, 1))  # 2n-element column vector of zeros.
    h[n::] = C

    G = matrix(0.0, (2 * n, n))  # These lines generate G, an
    G[n::((2 * n) + 1)] = 1.0
    G[::((2 * n) + 1)] = -1.0  # 2n x n matrix with -1.0's on its
    # main diagonal
    A = matrix(Ts, (1, n), tc='d')  # A is an n-element row vector of
    # training outputs.

    ##
    ## Now call "qp". Details of the parameters to the call can be
    ## found in the online cvxopt documentation.
    ##
    r = solvers.qp(P, q, G, h, A, matrix(0.0))  ## "qp" returns a dict, r.
    ##
    ## print r ## Dump entire result dictionary
    ## ## to terminal.
    ##
    ## Return results. Return a tuple, (Status,Ls). First element is
    ## a string, which will be "optimal" if a solution has been found.
    ## The second element is a list of Lagrange multipliers for the problem,
    ## rounded to six decimal digits to remove algorithm noise.
    ##
    Ls = [round(l, 6) for l in list(r['x'])]  ## "L's" are under the 'x' key.
    return r['status'], Ls


def makeB(Xs, Ts, Ls=None, K=rbfKernel):
    "Generate the bias given Xs, Ts and (optionally) Ls and K"
    ## No Lagrange multipliers supplied, generate them.
    if Ls is None:
        status, Ls = makeLambdas(Xs, Ts, C)
        ## If Ls generation failed (non-seperable problem) throw exception
        if status != "optimal": raise Exception("Can't find Lambdas")
    ## Calculate bias as average over all support vectors (non-zero
    ## Lagrange multipliers.
    sv_count = 0
    b_sum = 0.0
    for n in range(len(Ts)):
        if Ls[n] >= 1e-10:  ## 1e-10 for numerical stability.
            if Ls[n] <= C:
                sv_count += 1
                b_sum += Ts[n]
                for i in range(len(Ts)):
                    if Ls[i] >= 1e-10:
                        if Ls[i] <= C:
                            b_sum -= Ls[i] * Ts[i] * K(Xs[i], Xs[n])

    return b_sum / sv_count


def classify(x, Xs, Ts, Ls=None, b=None, K=rbfKernel, verbose=True):
    "Classify an input x into {-1,+1} given support vectors, outputs and L."
    ## No Lagrange multipliers supplied, generate them.
    if Ls is None:
        status, Ls = makeLambdas(Xs, Ts)
        ## If Ls generation failed (non-seperable problem) throw exception
        if status != "optimal": raise Exception("Can't find Lambdas")
    ## Calculate bias as average over all support vectors (non-zero
    ## Lagrange multipliers.
    if b is None: b = makeB(Xs, Ts, Ls, K)
    ## Do classification. y is the "activation level".
    y = b
    for n in range(len(Ts)):
        if Ls[n] >= 1e-10:
            y += Ls[n] * Ts[n] * K(Xs[n], x)

    if verbose:
        print "%s %8.5f --> " % (x, y),
        if y > 0.0:
            print "+1"
        elif y < 0.0:
            print "-1"
        else:
            print "0 (ERROR)"
    if y > 0.0:
        return +1
    elif y < 0.0:
        return -1
    else:
        return 0


def makeP(xs, ts, K):
    """Make the P matrix given the list of training vectors,
    desired outputs and kernel."""
    N = len(xs)
    assert N == len(ts)
    P = matrix(0.0, (N, N), tc='d')
    for i in range(N):
        for j in range(N):
            P[i, j] = ts[i] * ts[j] * K(xs[i], xs[j])
    return P


def plotContours(Xs, Ts, Ls=None, b=None, K=rbfKernel, labelContours=False, labelPoints=False):
    """Plot contours of activation function for a 2-d classifier, e.g. 2-input XOR."""
    assert len(Xs) == len(Ts)
    assert len(Xs[0]) == 2  ## Only works with a 2-d classifier.
    ## No Ls supplied, generate them.
    if Ls is None:
        status, Ls = makeLambdas(Xs, Ts, K)
        ## If Ls generation failed (non-seperable problem) throw exception
        if status != "optimal": raise Exception("Can't find Lambdas")
        print "Lagrange multipliers:", Ls
    ## Calculate bias as average over all support vectors (non-zero
    ## Lagrange multipliers.
    if b is None:
        b = makeB(Xs, Ts, Ls, K)
        print "Bias:", b
    ## Build activation level array.
    xs = arange(-0.6, 1.61, 0.05)
    ys = arange(-0.6, 1.61, 0.05)
    als = zeros((len(xs), len(ys)))
    for i, x in enumerate(xs):
        for j, y in enumerate(ys):
            testvector = [x, y]
            al = b
            for n in range(len(Ts)):
                if Ls[n] >= 1e-10:
                    al += Ls[n] * Ts[n] * K(Xs[n], testvector)
            als[j, i] = al  ## N.B. Matplotlib array/matrix indexing is row/col
    ## with rows corresponding to x's and cols to y's.
    ## Plot contour lines at 0.0, +1.0 and -1.0.
    X, Y = meshgrid(xs, ys)
    CS = plt.contour(X, Y, als, levels=[-1.0, 0.0, 1.0], linewidths=(1, 2, 1))
    ## Plot the training points using red 7 blue circles.
    for i, t in enumerate(Ts):
        if t < 0:
            col = 'blue'
        else:
            col = 'red'
        ##if labelPoints:
        ## print "Plotting %s (%d) as %s"%(Xs[i],t,col)
        # plt.text(Xs[i][0]+0.1,Xs[i][1],"%s: %d"%(Xs[i],t),color=col)
        plt.plot([Xs[i][0]], [Xs[i][1]], marker='o', color=col)
    ## Generate labels for contours if flag 'labelContours' is set to
    ## strings 'manual' or 'auto'. Manual is manual labelling, auto is
    ## automatic labelling (which can mess up if hidden behind data
    ## points).
    if labelContours == 'manual':
        plt.clabel(CS, fontsize=9, manual=True)
    elif labelContours == 'auto':
        plt.clabel(CS, fontsize=9)
    plt.show()


C = 1000000
data = np.loadtxt('training-dataset-aut-2017.txt')
Ts = data[:, 2]
Xs = data[:, (0, 1)]
# Xs=25*[None]
# Ts=25*[None]
for i in range(5):
    for j in range(5):
        Xs[i * 5 + j] = [0.25 * j, 0.25 * i]
    if 4 > i > 0 < j < 4:
        Ts[i * 5 + j] = 1
    else:
        Ts[i * 5 + j] = -1
status, L3 = makeLambdas(Xs, Ts, C, K=rbf2)
plotContours(Xs, Ts, L3, b=None, K=rbfKernel, labelContours=False, labelPoints=False)
