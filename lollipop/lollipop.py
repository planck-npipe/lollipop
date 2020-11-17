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
        
        fsky = 0.52
        
        #Binning (fixed binning)
        self.bins = tools.get_binning()
        
        #Data
        self.log.debug("Reading cross-spectrum")
        filepath = os.path.join(self.data_folder,self.clfile)
        data = tools.read_dl(filepath)
        self.cldata = self.bins.bin_spectra(data[1])
        
        #Fiducial spectrum
        self.log.debug("Reading model")
        filepath = os.path.join(self.data_folder,self.fiducialfile)
        data = tools.read_dl(filepath)
        self.clfid = self.bins.bin_spectra(data[1])
        
        #covmat
        self.log.debug("Reading covariance")
        filepath = os.path.join(self.data_folder,self.clcovfile)
        clcov = fits.getdata(filepath)
        cbcov = tools.bin_covB( clcov, self.bins)
        clvar = np.diag(cbcov)
        self.invclcov = np.linalg.inv(cbcov)
        
        #compute offsets
        self.log.debug("Compute offsets")
        self.cloff = tools.compute_offsets( self.bins.lbin, clvar, self.clfid, fsky=fsky)        
    
    def get_requirements(self):
        return dict(Cl={mode: self.bins.lmax for mode in ["bb"]})
    
    def logp(self, **params_values):
        cl = self.theory.get_Cl(ell_factor=False) #Cl in muK^2
        return self.loglike(cl, **params_values)
    
    def loglike(self, cl, **params_values):
        '''
        Compute offset-Hamimeche&Lewis likelihood
        Input: Cl in muK^2
        '''
        from numpy import dot, sign, sqrt
        
        #model in Cl, muK^2
        clth = self.bins.bin_spectra( cl["bb"])
        
        x = (self.cldata+self.cloff)/(clth+self.cloff)
        g = sign(x)*tools.ghl( abs(x))
        
        X = (sqrt(self.clfid+self.cloff)) * g * (sqrt(self.clfid+self.cloff))
        
        chi2 = dot( X.transpose(),dot(self.invclcov, X))
        
        return( chi2)





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

        fsky = 0.52
        rcond=1e-9
        
        #Binning (fixed binning)
        self.bins = tools.get_binning()
        
        #Data (ell,ee,bb,eb)
        self.log.debug("Reading cross-spectrum")
        filepath = os.path.join(self.data_folder,self.clfile)
        data = tools.read_dl(filepath)
        self.cldata = self.bins.bin_spectra( data)
        
        #Fiducial spectrum (ell,ee,bb,eb)
        self.log.debug("Reading model")
        filepath = os.path.join(self.data_folder,self.fiducialfile)
        data = tools.read_dl(filepath)
        self.clfid = self.bins.bin_spectra( data)
        
        #covmat (ee,bb,eb)
        self.log.debug("Reading covariance")
        filepath = os.path.join(self.data_folder,self.clcovfile)
        clcov = fits.getdata(filepath)
        cbcov = tools.bin_covEB( clcov, self.bins)
        clvar = np.diag(cbcov).reshape(-1,self.bins.nbins)
        if rcond != 0.:
            self.invcov = np.linalg.pinv(cbcov,rcond)
        else:
            self.invcov = np.linalg.inv(cbcov)
        
        #compute offsets
        self.log.debug("Compute offsets")
        self.cloff = tools.compute_offsets( self.bins.lbin, clvar, self.clfid, fsky=fsky)
        self.cloff[2:] = 0. #force NO offsets EB        
    
    def get_requirements(self):
        return dict(Cl={mode: self.bins.lmax for mode in ["ee", "bb"]})
#        return dict(Cl={mode: self.bins.lmax for mode in ["ee", "bb", "eb"]})
    
    def logp(self, **params_values):
        cl = self.theory.get_Cl(ell_factor=False)
        return self.loglike(cl, **params_values)
    
    def loglike(self, cl, **params_values):
        '''
        Compute offset-Hamimeche&Lewis likelihood
        Input: Cl in muK^2
        '''
        from numpy import dot, diag, sqrt, sign
        from numpy.linalg import eigh

        #get model in Cl, muK^2
        clth = []
#        for mode in ["ee", "bb", "eb"]:
        for mode in ["ee", "bb"]:
            clth.append(self.bins.bin_spectra( cl[mode]))
        clth = np.array(clth)
        
        ndim,nel = np.shape(self.cldata)
        
        x = np.zeros( (ndim, nel))
        for l in range(nel):
            O = tools.vec2mat(  self.cloff[:,l])
            D = tools.vec2mat( self.cldata[:,l]) + O
            M = tools.vec2mat(        clth[:,l]) + O
            F = tools.vec2mat(  self.clfid[:,l]) + O
            
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
        
        return( -0.5*chi2)

