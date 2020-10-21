#Offset H&L likelihoods
#---------------------------------------------------------------------------

def compute_offsets( ell, varcl, clref, fsky=1., iter=10):
    Nl = sqrt( abs(varcl - ( 2./(2.*ell+1) * clref**2)/fsky))
    for i in range(iter):
        Nl = sqrt( abs(varcl - 2./(2.*ell+1)/fsky * (clref**2 + 2.*Nl*clref)))
    return( Nl * sqrt((2.*ell+1)/2.) )


class oHL:
    import numpy as np
    
    def __init__( self, data, off, covariance, fiducial, nsimu=0.):
        self.data   = data
        self.off    = off
        self.fid    = fiducial
        self.invcov = linalg.inv(covariance)
        if nsimu:
            nd = len(covariance)
            self.invcov = self.invcov*(nsimu-nd-2.)/(nsimu-1.)

    def _ghl(self, x):
        return( sign(x-1)*sqrt(2. * (x-log(x)-1)))

    def compute_likelihood( self, model):
        x = (self.data+self.off)/(model+self.off)
        g = sign(x)*self._ghl( abs(x))

        X = (sqrt(self.fid+self.off)) * g * (sqrt(self.fid+self.off))

        chi2 = self.np.dot( X.transpose(),self.np.dot(self.invcov, X))

        return( chi2)
        


def HL_vec2mat( vect, ndim=3):
    #Set Cl matrix from Cl vector: TT,EE,BB,TE,TB,EB
    
    mat = zeros( (ndim,ndim))
    if ndim==1:
        mat[0,0] = vect[0]
    if ndim==2:
        mat[0,0] = vect[1]
        mat[1,1] = vect[2]
        if( len(vect) > 5):
            mat[1,0] = mat[0,1] = vect[5]
    if ndim==3:
        for i in range(3):
            mat[i,i] = vect[i]

        if( len(vect) >= 4):
            mat[0,1] = mat[1,0] = vect[3]
    
        if( len(vect) > 4):
            mat[0,2] = mat[2,0] = vect[4]
            mat[1,2] = mat[2,1] = vect[5]
    
    return(mat)


def HL_mat2vec( mat, ndim=3):

    if ndim==1:
        vec = [mat[0]]
    if ndim==2:
        vec = [mat[0,0],mat[1,1],mat[0,1]]
    if ndim==3:
        vec = [mat[0,0],mat[1,1],mat[2,2],mat[0,1],mat[0,2],mat[1,2]]
    
    return( vec)




class oHL2:
    import numpy as np
    """
    Compute offset-H&M approximation modified for cross-spectra
    for EE and BB (optional EB) spectra
    """
    
    def __init__( self, data, offset, covariance, fiducial, nsimu=0., rcond=0., EB=False):
        """
        init and compute invert covariance matrix
        
        * data: array(6,nbin)
                data array of spectra
        * offset: array(6,nbin)
                  data array of offsets
        * covariance: array(nspec*nbin,nspec*nbin)
                      Cl covariance matrix
        * fiducial: array(6,nbin)
                    data array of fiducial spectrum
        * rcond: scalar [optional]
                 scalar for psuedo-inversion of covariance matrix
        * EB: bool [optional]
              set True to include EB spectrum (default: False)
        """
        self.dat = data
        self.off = offset
        self.fid = fiducial
        self.isEB = EB

        self.off[3:] = 0. #force NO offsets for TE,TB, EB
        if rcond != 0.:
            self.invcov = linalg.pinv(covariance,rcond)
        else:
            self.invcov = linalg.inv(covariance)
        if nsimu != 0.:
            nd = len(covariance)
            self.invcov = self.invcov*(nsimu-nd-2.)/(nsimu-1.)

    def _ghl(self, x):
        return( sign(x-1)*sqrt(2. * (x-log(x)-1)))

    def _vec2mat( self, vect):
        mat = zeros( (2,2))
        mat[0,0] = vect[1]
        mat[1,1] = vect[2]
        if self.isEB:
            if len(vect) > 5:
                mat[1,0] = mat[0,1] = vect[5]
        return(mat)

    def _mat2vec( self, mat):
        if self.isEB: vec = [mat[0,0],mat[1,1],mat[0,1]]
        else: vec = [mat[0,0],mat[1,1]]
        return( vec)

    def compute_likelihood( self, model):
        nel = len(self.dat[0])
        ndim = 3 if self.isEB else 2
        
        x = zeros( (ndim, nel))
        for l in range(nel):
            O = self._vec2mat( self.off[:,l])
            D = self._vec2mat( self.dat[:,l]) + O
            M = self._vec2mat(    model[:,l]) + O
            F = self._vec2mat( self.fid[:,l]) + O
            
            #compute P = C_model^{-1/2}.C_data.C_model^{-1/2}
            w,V = eigh(M)
#            if prod( sign(w)) <= 0:
#                print( "WARNING: negative eigenvalue for l=%d" %l)
            L = dot( V, dot( diag(1./sqrt(w)), V.transpose()))
            P = dot( L.transpose(), dot( D, L))
            
            #apply H&L transformation
            w,V = eigh(P)
            g = sign(w)*self._ghl( abs(w))
            G = dot( V, dot(diag(g), V.transpose()))
            
            #cholesky fiducial
            w,V = eigh(F)
            L = dot(V,dot(diag(sqrt(w)),V.transpose()))
            
            #compute C_fid^1/2 * G * C_fid^1/2
            X = dot( L.transpose(), dot(G, L))
            x[:,l] = self._mat2vec(X)

        #compute chi2
        x = x.flatten()
        chi2 = dot( x, dot( self.invcov, x))
        
        return( chi2)


def ghl(x):
    return( sign(x-1)*sqrt(2. * (x-log(x)-1)))

def oHL3( data, model, offset, covariance, fiducial):
    nel = len(data[0])
    
    x = zeros( (6, nel))
    for l in range(nel):
        D = HL_vec2mat(     data[:,l]) + identity(3)*offset[:,l]
        M = HL_vec2mat(    model[:,l]) + identity(3)*offset[:,l]
        F = HL_vec2mat( fiducial[:,l]) + identity(3)*offset[:,l]
        
        #compute P = C_model^{-1/2}.C_data.C_model^{-1/2}
        w,V = eigh(M)
        L = dot( V, dot( identity(3)*1./sqrt(w), V.transpose()))
        P = dot( L.transpose(), dot( D, L))
        
        #apply H&L transformation
        w,V = eigh(P)        
        g = sign(w)*ghl( abs(w))
        G = dot( V, dot(identity(3)*g, V.transpose()))
        
        #cholesky fiducial
        w,V = eigh(F)
        L = dot(V,dot(identity(3)*sqrt(w),V.transpose()))
        
        #compute C_fid^1/2 * G * C_fid^1/2
        X = dot( L.transpose(), dot(G, L))
        x[:,l] = HL_mat2vec(X)
    
    #invert cov
#    bad = where( x[0,:] == 0)[0]
#    if len(bad) !=0:
#        print "nb of negativ : %d" % len(bad)
#    good = array(where( x[0,:] != 0))[0]
#    good = array([good,good+nel,good+2*nel,good+3*nel,good+4*nel,good+5*nel]).reshape(6*len(good))
#    x = x.reshape(6*nel)[good]
#    mycov = covariance[:,good]
#    invcov = linalg.inv(mycov[good,:])
    x = x.reshape(6*nel)
    invcov = linalg.inv(covariance)
    
    #compute chi2
    chi2 = dot( x, dot( invcov, x))
    
    return( chi2)

#---------------------------------------------------------------------------




