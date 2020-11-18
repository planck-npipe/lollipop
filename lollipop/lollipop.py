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



class _lollipop_likelihood(_InstallableLikelihood):
    """
    Low-L Likelihood for Polarized Planck for EE+BB+EB
    Spectra-based likelihood based on Hamimeche-Lewis for cross-spectra
    applied on CMB component separated map
    """

    _mode = "EEBBEB"

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
        rcond = 0.
        if self._mode == "EEBBEB":
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
        if self._mode == "EEBBEB":
            cbcov = tools.bin_covEB( clcov, self.bins)
        elif self._mode == "EE":
            cbcov = tools.bin_covEE( clcov, self.bins)
        elif self._mode == "BB":
            cbcov = tools.bin_covBB( clcov, self.bins)
        clvar = np.diag(cbcov).reshape(-1,self.bins.nbins)
        if rcond != 0.:
            self.invclcov = np.linalg.pinv(cbcov,rcond)
        else:
            self.invclcov = np.linalg.inv(cbcov)
        
        #compute offsets
        self.log.debug("Compute offsets")
        self.cloff = tools.compute_offsets( self.bins.lbin, clvar, self.clfid, fsky=fsky)
        self.cloff[2:] = 0. #force NO offsets EB
    
    def _compute_chi2_2fields(self, cl, **params_values):
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
        chi2 = dot( x, dot( self.invclcov, x))
        
        return chi2

    def _compute_chi2_1field(self, cl, **params_values):
        '''
        Compute offset-Hamimeche&Lewis likelihood
        Input: Cl in muK^2
        '''
        from numpy import dot, sign, sqrt
        
        #model in Cl, muK^2
        if self._mode == "EE":
            clth = self.bins.bin_spectra( cl["ee"])
            m = 0
        elif self._mode == "BB":
            clth = self.bins.bin_spectra( cl["bb"])
            m = 1
        
        x = (self.cldata[m]+self.cloff[m])/(clth+self.cloff[m])
        g = sign(x)*tools.ghl( abs(x))
        
        X = (sqrt(self.clfid[m]+self.cloff[m])) * g * (sqrt(self.clfid[m]+self.cloff[m]))
        
        chi2 = dot( X.transpose(),dot(self.invclcov, X))
        
        return chi2

    def get_requirements(self):
        return dict(Cl={mode: self.bins.lmax for mode in ["ee", "bb"]})
    
    def logp(self, **params_values):
        cl = self.theory.get_Cl(ell_factor=False)
        return self.loglike(cl, **params_values)

    def loglike(self, cl, **params_values):

        if self._mode == "EEBBEB":
            chi2 = self._compute_chi2_2fields(cl, **params_values)
        elif self._mode == "EE":
            chi2 = self._compute_chi2_1field(cl, **params_values)
        elif self._mode == "BB":
            chi2 = self._compute_chi2_1field(cl, **params_values)
        
        return( -0.5*chi2)




class lowlEB(_lollipop_likelihood):
    """
    Low-L Likelihood for Polarized Planck for EE+BB+EB
    Spectra-based likelihood based on Hamimeche-Lewis for cross-spectra
    applied on CMB component separated map
    """
    _mode = "EEBBEB"


class lowlE(_lollipop_likelihood):
    """
    Low-L Likelihood for Polarized Planck for EE+BB+EB
    Spectra-based likelihood based on Hamimeche-Lewis for cross-spectra
    applied on CMB component separated map
    """
    _mode = "EE"


class lowlB(_lollipop_likelihood):
    """
    Low-L Likelihood for Polarized Planck for EE+BB+EB
    Spectra-based likelihood based on Hamimeche-Lewis for cross-spectra
    applied on CMB component separated map
    """
    _mode = "BB"
