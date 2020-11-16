import numpy as np
from numpy import linalg


def compute_offsets( ell, varcl, clref, fsky=1., iter=10):
    Nl = sqrt( abs(varcl - ( 2./(2.*ell+1) * clref**2)/fsky))
    for i in range(iter):
        Nl = sqrt( abs(varcl - 2./(2.*ell+1)/fsky * (clref**2 + 2.*Nl*clref)))
    return( Nl * sqrt((2.*ell+1)/2.) )


def get_binning():
    dl = 10
    llmin = 2; llmax = 35
    hlmin = 36; hlmax = 150
    lmins = range(llmin,llmax+1)+range(hlmin,hlmax-dl+2,dl)
    lmaxs = range(llmin,llmax+1)+range(hlmin+dl-1,hlmax+1,dl)
    binc = Bins(lmins,lmaxs)
    return(binc)

def vec2mat( vect, isEB=True):
    mat = zeros( (2,2))
    mat[0,0] = vect[1]
    mat[1,1] = vect[2]
    if isEB:
        if len(vect) > 5:
            mat[1,0] = mat[0,1] = vect[5]
    return(mat)

def mat2vec( mat, isEB=True):
    if isEB: vec = [mat[0,0],mat[1,1],mat[0,1]]
    else: vec = [mat[0,0],mat[1,1]]
    return( vec)


def ghl(x):
    return( sign(x-1)*sqrt(2. * (x-log(x)-1)))

