#
# LOLLIPOP
#
# Oct 2020   - M. Tristram -
import os
from typing import Optional

import numpy as np
import astropy.io.fits as fits

from cobaya.conventions import _packages_path
from cobaya.likelihoods._base_classes import _InstallableLikelihood
from cobaya.log import LoggedError
from cobaya.tools import are_different_params_lists

from lollipop.bins import Bins
from lollipop import tools


class lowlB(_InstallableLikelihood):
    """
    Low-L Likelihood for Polarized Planck for BB
    Spectra-based likelihood based on Hamimeche-Lewis for cross-spectra
    applied on CMB component separated map
    """
    
    def initialize( self):
        self.log.info("Initialising.")
        
        # Set path to data
        if (not getattr(self, "path", None)) and (not getattr(self, _packages_path, None)):
            raise LoggedError(
                self.log,
                "No path given to Lollipop data. Set the likelihood property 'path' or the common property '%s'.",
                _packages_path,
            )
        
        # If no path specified, use the modules path
        data_file_path = os.path.normpath(
            getattr(self, "path", None) or os.path.join(self.packages_path, "data")
        )
        
        self.data_folder = os.path.join(data_file_path)
        if not os.path.exists(self.data_folder):
            raise LoggedError(
                self.log,
                "The 'data_folder' directory does not exist. Check the given path [%s].",
                self.data_folder,
            )
        
#        self._fsky = fsky
        fsky = 0.52
        
        #Binning (fixed binning)
        self.binc = tools.get_binning()
        
        #Data
        self.log.debug("Reading cross-spectrum")
        filepath = os.path.join(self.data_folder,self.clfile)
        data = tools.read_dl(filepath)
        cldata = self.binc.bin_spectra(data)
        
        #Fiducial spectrum
        self.log.debug("Reading model")
        filepath = os.path.join(self.data_folder,self.fiducialfile)
        data = tools.read_dl(filepath)
        clfid = self.binc.bin_spectra(data)
        
        #covmat
        self.log.debug("Reading covariance")
        filepath = os.path.join(self.data_folder,self.clcovfile)
        clcov = fits.getdata(filepath)
        cbcov = tools.bin_covB( clcov, self.binc)
        
        #compute offsets
        self.log.debug("Compute offsets")
        cloff = tools.compute_offsets( self.binc.lbin, np.diag(cbcov), clfid[1], fsky=fsky)
        
        #construct BB likelihood
        self.invcov = np.linalg.inv(cbcov)
        self.data = cldata[1]
        self.off  = cloff
        self.fid  = clfid[1]
        
    def _compute_likelihood( self, cl):
        from numpy import dot, sign, sqrt
        
        x = (self.data+self.off)/(cl+self.off)
        g = sign(x)*tools.ghl( abs(x))
        
        X = (sqrt(self.fid+self.off)) * g * (sqrt(self.fid+self.off))
        
        chi2 = dot( X.transpose(),dot(self.invcov, X))
        
        return( chi2)
    
    def get_requirements(self):
        return dict(Cl={mode: self.binc.lmax for mode in ["bb"]})
    
    def logp(self, **params_values):
        cl = self.theory.get_Cl(ell_factor=False)
        return self.loglike(cl, **params_values)
    
    def loglike(self, cl, **params_values):
        '''
        Cl in muK^2 ?
        '''
        model = self.binc.bin_spectra( cl["bb"])
        return self._compute_likelihood(model)





class lowlEB(_InstallableLikelihood):
    """
    Low-L Likelihood for Polarized Planck for EE+BB+EB
    Spectra-based likelihood based on Hamimeche-Lewis for cross-spectra
    applied on CMB component separated map
    """

    def initialize( self):
        self.log.info("Initialising.")

        # Set path to data
        if (not getattr(self, "path", None)) and (not getattr(self, _packages_path, None)):
            raise LoggedError(
                self.log,
                "No path given to Lollipop data. Set the likelihood property 'path' or the common property '%s'.",
                _packages_path,
            )

        # If no path specified, use the modules path
        data_file_path = os.path.normpath(
            getattr(self, "path", None) or os.path.join(self.packages_path, "data")
        )

        self.data_folder = os.path.join(data_file_path)
        if not os.path.exists(self.data_folder):
            raise LoggedError(
                self.log,
                "The 'data_folder' directory does not exist. Check the given path [%s].",
                self.data_folder,
            )

#        self._fsky = fsky
        fsky = 0.5
        rcond=1e-9
        
        #Binning (fixed binning)
        self.binc = tools.get_binning()
        
        #Data (ell,ee,bb,eb)
        self.log.debug("Reading cross-spectrum")
        filepath = os.path.join(self.data_folder,self.clfile)
        data = tools.read_dl(filepath)
        cldata = self.binc.bin_spectra( data)
        
        #Fiducial spectrum (ell,ee,bb,eb)
        self.log.debug("Reading model")
        filepath = os.path.join(self.data_folder,self.fiducialfile)
        data = tools.read_dl(filepath)
        clfid = self.binc.bin_spectra( data)
        
        #covmat (ee,bb,eb)
        self.log.debug("Reading covariance")
        filepath = os.path.join(self.data_folder,self.clcovfile)
        clcov = fits.getdata(filepath)
        cbcov = tools.bin_covEB( clcov, self.binc)
        if rcond != 0.:
            self.invcov = np.linalg.pinv(cbcov,rcond)
        else:
            self.invcov = np.linalg.inv(cbcov)
        
        #compute offsets
        self.log.debug("Compute offsets")
        cloff = tools.compute_offsets( self.binc.lbin, np.diag(cbcov).reshape(-1,self.binc.nbins), clfid, fsky=fsky)
        cloff[2:] = 0. #force NO offsets EB
        
        #construct likelihood
        self.data = cldata
        self.off = cloff
        self.fid = clfid
        
    def _compute_likelihood( self, cl):
        from numpy import dot, diag, sqrt, sign
        from numpy.linalg import eigh
        
        nel = len(self.data[0])
        ndim = len(cl)
        
        x = np.zeros( (ndim, nel))
        for l in range(nel):
            O = tools.vec2mat(  self.off[:,l])
            D = tools.vec2mat( self.data[:,l]) + O
            M = tools.vec2mat(        cl[:,l]) + O
            F = tools.vec2mat(  self.fid[:,l]) + O
            
            #compute P = C_model^{-1/2}.C_data.C_model^{-1/2}
            w,V = eigh(M)
#            if prod( sign(w)) <= 0:
#                print( "WARNING: negative eigenvalue for l=%d" %l)
            L = dot( V, dot( diag(1./sqrt(w)), V.transpose()))
            P = dot( L.transpose(), dot( D, L))
            
            #apply HL transformation
            w,V = eigh(P)
            g = sign(w)*tools.ghl( abs(w))
            G = dot( V, dot(diag(g), V.transpose()))
            
            #cholesky fiducial
            w,V = eigh(F)
            L = dot(V,dot(diag(sqrt(w)),V.transpose()))
            
            #compute C_fid^1/2 * G * C_fid^1/2
            X = dot( L.transpose(), dot(G, L))
            x[:,l] = tools.mat2vec(X)

        #compute chi2
        x = x.flatten()
        chi2 = dot( x, dot( self.invcov, x))
        
        return( chi2)

    def get_requirements(self):
        return dict(Cl={mode: self.binc.lmax for mode in ["ee", "bb", "eb"]})

    def logp(self, **params_values):
        cl = self.theory.get_Cl(ell_factor=False)
        return self.loglike(cl, **params_values)

    def loglike(self, cl, **params_values):
        clth = []
        for mode in ["ee", "bb", "eb"]:
            clth += [cl[mode]]
        model = self.binc.bin_spectra( clth)
        return self._compute_likelihood(model)

