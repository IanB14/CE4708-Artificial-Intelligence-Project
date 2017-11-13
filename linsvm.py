##--------------------------------------------------------------------------
##
##  linsvm.py
##
##  Routines to generate a hard-margin linear SVM.  No kernels are used
##  (just linear inner-products of the x vectors).  The solution to the
##  dual form Lagrangian optimization problem:
##
##      Maximize:
##
##        W(L) = \sum_i L_i - 1/2 sum_i sum_j L_i L_j t_i t_j x_i^t x_j
##
##      subject to:  \sum_i t_i L_i  =  0   and L_i >= 0.
##
##  (where L is a vector of n Lagrange multipliers) is found using the
##  quadratic program solver "qp" from the convex optimization package
##  "cvxopt" (which has to be installed).
##
##
##  Example run, generate weights and bias for a 2-input AND binary
##  classifier.
##
##  >>> Xs = makeBinarySequence(2)   ## Generate list [[0,0],[0,1]...]
##  >>> Ts = [-1,-1,-1,+1]           ## Desired response for a binary AND.
##  >>> stat,Ls = makeLambdas(xs,ts) ## Solve W(Ls) for Lagrange mults.
##                                   ## N.B., status == 'optimal' if a
##                                   ## solution has been found.
##  >>> w,b = makeWs(Xs, Ts, Ls)     ## Find weight vector and bias.
##  >>> classify(w,b,[0,1])          ## Test classification.
##
##--------------------------------------------------------------------------
##
##  Routines
##
##      makeLambdas  -- Generate the n Lagrange multipliers that
##                      represent the maximum-point of the dual
##                      optimization problem W(L).
##                      N.B., maximizing W(L) is a quadratic convex
##                      optimization problem, so the "qp" solver from
##                      "cvxopt" actaully does the work.  Most of what
##                      this routine does is simply setting up the
##                      arguments for the call to "qp".
##
##              Arguments:
##
##                 Xs -- A list of vectors (lists) representing
##                       the input training patterns for this
##                       problem.
##
##                       e.g., [[0,0],[0,1],[1,0],[1,1]], the set
##                       of binary training patterns.
##
##                 Ts -- A list of desired outputs.  These must
##                       be values in the set {-1,+1}.  If there
##                       are n input vectors in x, there must be
##                       exactly n values in t.
##
##                       e.g., [-1,-1,-1,1] -- the outputs
##                       for a 2-input AND function.
##
##              Returns:
##
##                 A 2-tuple (status,Ls).
##                 1) status:  The first element of the tuple is the 
##                    return status of the "qp" solver.  This will be the 
##                    string "optimal" if a solution has been found.  If 
##                    no solution can be found (say if an XOR problem is
##                    presented to the solver), status typically comes back
##                    as "unknown".
##                 2) Ls:  The second element of the tuple is a list of  
##                    the n Lagrange multipliers.  Element 0 of this list
##                    is the first multiplier and corresponds to Xs[0] and
##                    Ts[0], element 1 is the multiplier corresponding to 
##                    Xs[1] and Ts[1], etc.  These values are only 
##                    meaningful if the first element of the tuple, status
##                    has returned as "optimal".
##
##      --------------------------------------------------------------------
##
##      makeWs -- Given the set of training vectors, "Xs", the set of
##                training responses "Ts", and the set of Lagrange
##                multipliers for the problem "Ls", return the normal vector
##                for the decision surface, w, and the bias, b, of the
##                surface.
##
##              Arguments:
##
##                 Xs -- Inputs, as "Xs" in makeLambdas.
##
##                 Ts -- A list of desired outputs.  As "Ts" in
##                       makeLambdas.
##
##                 Ls -- A list of Lagrange multipliers, the
##                       solution to the contrained optimaztion
##                       of W(L) as returned by a call to
##                       makeLambdas.  N.B., if this argument is
##                       None (the default), this routine will call
##                       makeLambdas automatically.
##
##              Returns:
##
##                 A 2-tuple, (w,b).  Element[0] (w) is an n x 1
##                 cvxopt matrix representing the vector normal to the
##                 decision surface (i.e., for a linear machine, the
##                 hyperplane separating the two training classes.
##                 Element[1] (b) is a scalar, the bias of the separating
##                 hyperplane.
##
##      --------------------------------------------------------------------
##
##      classify -- Classify an input vector using the weights and bias
##                  generated by the above procedures.
##
##              Arguments:
##
##                 w -- A weight vector, as generated by makeWs.
##
##                 b -- The bias, as generated by makeWs.
##
##                 x -- An input vector (a list of values).
##
##
##              Returns:
##
##                 A classification, +1, -1 or 0 (which indicates an
##                 error in the classification, and shouldn't happen).
##
##     --------------------------------------------------------------------
##
##      testClassifier(w,b,Xs,Ts,verbose) --
##                  Test a classifier by checking to see if its response
##                  to every training input Xs[i] is the desired output
##                  Ts[i].
##
##              Arguments:
##
##                 w  -- A weight vector, as generated by genWs.
##
##                 b  -- The bias, as generated by genWs.
##
##                 Xs -- A list of vectors (lists) representing
##                       the input training patterns for this
##                       problem.
##
##                       e.g., [[0,0],[0,1],[1,0],[1,1]], the set
##                       of binary training patterns.
##
##                 Ts -- A list of desired outputs.  These must
##                       be values in the set {-1,+1}.  If there
##                       are n input vectors in x, there must be
##                       exactly n values in t.
##
##                       e.g., [-1,-1,-1,1] -- the outputs
##                       for a 2-input AND function.
##
##                 verbose -- Controls whether or not the routine
##                       prints details of misclassifications to the
##                       terminal as well as returning a status
##                       value.  Defaults to True.
##
##              Returns:
##
##                 True/False
##
##  ------------------------------------------------------------------------
##
##  Support routines
##
##
##      makeBinarySequence -- Generate a list of the 2^d d-element "vectors"
##                            comprising the complete binary sequence in
##                            d bits.
##
##              Argument:
##
##                 d -- Length of each vector in the output list.  Defaults
##                      to 2 (which will generate the 4 "vector" list
##                      [[0,0], [0,1], [1,0], [1,1]].
##
##              Returns:
##
##                 A list of 2^d elements, each of which is one vector in
##                 the binary sequence.  Example, a call with d=3 will
##                 return the 8-element list [[0,0,0], [0,0,1], [0,1,0],
##                 [0,1,1], [1,0,0], [1,0,1], [1,1,0], [1,1,1]].
##
##      
##      --------------------------------------------------------------------
##
##      makeP -- Generates the P matrix for a linear SVM problem.  See the
##               comments associated with generateLambdas below for a
##               discussion of the form and role of the P matrix.
##
##
##--------------------------------------------------------------------------
##


from cvxopt import matrix,solvers

##--------------------------------------------------------------------------
##
##  makeLambdas
##
##  Use the qp solver from the cvx package to find a list of Lagrange
##  multipliers (lambdas or L's) for an Xs/Ts problem, where Xs is a list
##  of input vectors (themselves represented as simple lists) and Ts a list
##  of desired outputs.
##
##
##  Note that we are trying to solve the problem:
##
##      Maximize:
##
##        W(L) =  \sum_i L_i
##                - 1/2 sum_i sum_j L_i * L_j * t_i * t_j * inner(x_i,x_j)
##
##      subject to:  \sum_i t_i L_i  =  0   and   L_i >= 0.
##
##
##  but the "standard" quadratic programming problem is subtly different,
##  it attempts to *minimize* the following quadratic form:
##
##        f(y) = 1/2 y^t P y  +  q^t y
##
##  subject to:  G L <= h   and   A y = b, where P is an n x n 
##  symmetric matrix, G is an n x n matrix, A is a 1 x n
##  (row) vector, q is an n x 1 (column) vector, as are h and y.
##  N.B., Vector y is the solution being searched for.
##
##  To turn the W(l) constrained maximazation into a constrained
##  minimization of f, it suffices to set:
##
##          [-1.0]
##             .
##             .
##      q = [-1.0]      (n element column vector).
##          [-1.0]
##             .
##             .
##          [-1.0]
##
##
##          [-1.0,  0.0  ....  0.0]
##          [ 0.0, -1.0           ]
##          [             .       ]   (n x n matrix with -1.0 on
##      G = [             .       ]    main diagonal and 0.0's
##          [            0.0,     ]    everywhere else. i.e -I) 
##          [           -1.0,  0.0]
##          [            0.0, -1.0]
##
##
##
##      A = [ t_1, t_2, t_3, ... t_n],  a row vector with n elements
##                                      made using the t list input.
##
##
##      h = n element column vector of 0.0's.
##
##      b = [0.0], i.e., a 1 x 1 matrix containing 0.0.
##
##
##          [                 ]   (n x n matrix with elements 
##      P = [ t_i t_j x_i x_j ]    t_i t_j x_i x_j).
##          [                 ]
##
##
##      Alternatively, P can be represented as the outer product
##      of the componentwise product of t and x.
##
##      P = (t*x)^t.(t*x),
##
##      where "*" represents "componentwise" multiplication.
##
##      e.g., say x = [[0,0],[0,1],[1,0],[1,1]] and t = [-1,-1,-1,1],
##
##
##             [ -1*[0,0] ] [ -1*[0]  -1*[0]  -1*[1]  +1*[1] ]
##      P =    [ -1*[0,1] ] [    [0]     [1]     [0]     [1] ]
##             [ -1*[1,0] ]
##             [ +1*[1,1] ]
##
##             [ 0.0   0.0   0.0   0.0 ]
##        =    [ 0.0   1.0   0.0  -1.0 ]
##             [ 0.0   0.0   1.0  -1.0 ]
##             [ 0.0  -1.0  -1.0   2.0 ]
##
##
##  The solution (if one exists) is returned by the "qp" solver as
##  a vector of elements.  The solver actually returns a dictionary
##  of values, this contains lots of information about the solution,
##  quality, etc.  But, from the point of view of this routine the
##  important part is the vector of "l" values, which is accessed
##  under the 'x' key in the returned dictionary.
##
##  N.B.  All the routines in the Cxvopt library are very sensitive to
##  the data types of their arguments.  In particular, all vectors,
##  matrices, etc., passed to "qp" must have elements that are
##  doubles.

def makeLambdas(Xs,Ts):
    "Solve constrained maximaization problem and return list of l's."
    P = makeP(Xs,Ts)              ## Build the P matrix.
    n = len(Ts)
    q = matrix(-1.0,(n,1))        ## This builds an n-element column 
                                  ## vector of -1.0's (note the double-
                                  ## precision constant).
    h = matrix(0.0,(n,1))         ## n-element column vector of zeros.

    G = matrix(0.0,(n,n))         ## These lines generate G, an 
    G[::(n+1)] = -1.0             ## n x n matrix with -1.0's on its
                                  ## main diagonal.
    A = matrix(Ts,(1,n),tc='d')   ## A is an n-element row vector of 
                                  ## training outputs.
                                
    ##
    ## Now call "qp". Details of the parameters to the call can be
    ## found in the online cvxopt documentation.
    ##
    r = solvers.qp(P,q,G,h,A,matrix(0.0))  ## "qp" returns a dict, r.
    ##
    ## print r                     ## Dump entire result dictionary
    ##                             ## to terminal.
    ##
    ## Return results. Return a tuple, (Status,Ls).  First element is
    ## a string, which will be "optimal" if a solution has been found.
    ## The second element is a list of Lagrange multipliers for the problem,
    ## rounded to six decimal digits to remove algorithm noise.
    ##
    Ls = [round(l,6) for l in list(r['x'])] ## "L's" are under the 'x' key.
    return (r['status'],Ls)


##--------------------------------------------------------------------------
##
##  Make a weight vector for a separating hypersurface given a list of
##  lambdas for the problem.  Also generate the bias.  Return a tuple
##  (w, b), where w is a vector (1-d matrix) of weights and b is the bias
##  (a scalar).
##
##  The weight vector = \sum_i^n L_i t_i x_i.  Only values with non-zero
##  a's need be summed, strictly speaking.  This is ignored here.
##  The x's corrsponding to non-zero (or not very small, < 1e-5) values of
##  L are the support vectors of the solution.
##
##  The bias = t_i - w.T - x_i for any support vector (i.e., vector
##  whose corresponding a_i is non zero).  For numerical stability an
##  average over all the non-zero support vectors is calculated.
##

def makeWs(Xs,Ts,Ls=None):
    "Generate the weight vector and bias given Xs, Ts and (optionally) Ls."
    ## No Ls supplied, generate them.
    if Ls == None:
        status,Ls = makeLambdas(Xs,Ts)
        ## If Ls generation failed (non-seperable problem)
        ## return a normal (i.e., weight) vector of n zeros and a bias
        ## of zero as well.
        if status != "optimal": return (len(Ts)*[0.0],b)

    Xs = matrix(Xs)
    w=sum([Ls[i]*Ts[i]*Xs[:,i] for i in range(len(Ts))])

    sv_count = 0
    b_sum = 0.0
    for i in range(len(Ts)):
        if Ls[i] >= 1e-5:   ## 1e-5 for numerical stability.
            sv_count += 1
            b_sum += Ts[i] - (w.T * Xs[:,i])[0,0]
            
    return (w, round(b_sum/sv_count,6))


def classify(w,b,x):
    "Classify an input x into {-1,+1} given weight vector and bias term." 
    v = sum(w[i] * x[i] for i in range(len(w))) + b
    if v > 0.0: return +1
    elif v < 0.0: return -1
    else: return 0 


def testClassifier(w,b,Xs,Ts,verbose=True):
    "Test a classifier specifed by weights and bias on all Xs/Ts pairs."
    good = True
    assert len(Xs) == len(Ts)
    for i in range(len(Xs)):
        c = classify(w,b,Xs[i])
        if c != Ts[i]:
            if verbose:
                print "Misclassification: input %s, output %d, expected %d" %\
                      (Xs[i],c,Ts[i])
            good = False
    return good



##--------------------------------------------------------------------------
##
##  Auxiliary routines.
##
##--------------------------------------------------------------------------
##

## Generate a list of binary vectors, [[0,0],[0,1],[1,0],[1,1]], etc.
## The argument is the length of the vectors (above = 2).
##
def makeBinarySequence(d=2):
    "Return a binary sequence of 2^d vectors, each of length d."
    s = []
    for i in range(2**d):
        v=[]
        for j in range(d-1,-1,-1):
            v.append((i>>j)&1)
        s.append(v)
    return s

## Make the P matrix for a LINEAR SVM problem.
##
def makeP(xs,ts):
    "Make the P matrix given the list of training vectors and desired outputs."
    N = len(xs)
    assert N == len(ts)
    X = matrix(xs,tc='d')
    for i in range(N): X[:,i] *= ts[i]
    P = X.T * X
    return P


##--------------------------------------------------------------------------
##
##
##
## Test data
##
##

def makeXor(Xs):
    "Make an n-XOR output vector for Xs, a list of n-element vectors."
    return [-1+2*(sum(x)%2==1) for x in Xs]

def makeAnd(Xs):
    "Make an n-AND output vector for Xs, a list of n-element vectors."
    return [-1+2*(sum(x)==len(x)) for x in Xs]

##
## Some input sequences, X2s is a 4 element list of all possible length
## 2 binary vectors, X3s an 8 element list of all possible length 3
## binary vectors.
##
X2s=makeBinarySequence(2)
X3s=makeBinarySequence(3)

## An output sequence for 3-input XOR and 3-input AND.
T3xor=makeXor(X3s)
T3and=makeAnd(X3s)

## Try to generate Lagrange multipliers for the 3-input XOR.  Should
## fail as XOR is not linearly seperable.
##
status,L3xor=makeLambdas(X3s,T3xor)
print "\nAttempted to generate Lagrange multipliers for 3-input XOR"
print "  Result status:", status
print "  L vector:", L3xor

w,b=makeWs(X3s,T3and,L3xor)
print "  Weight vector:", list(w), " and bias:", b
if testClassifier(w,b,X3s,T3and):
    print "  Check PASSED"
else:
    print "  Check FAILED"

print "\n\n"

## Now try to generate Lagrange multipliers for the 3-input AND.
## This should succeed.
##
status,L3and=makeLambdas(X3s,T3and)
print "\nAttempted to generate Lagrange multipliers for 3-input AND"
print "  Result status:", status
print "  L vector:", L3and

w,b=makeWs(X3s,T3and,L3and)
print "  Weight vector:", list(w), " and bias:", b
if testClassifier(w,b,X3s,T3and):
    print "  Check PASSED"
else:
    print "  Check FAILED"
