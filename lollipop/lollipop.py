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
        
        self.data_folder = os.path.join(data_file_path, self.data_folder)
        if not os.path.exists(self.data_folder):
            raise LoggedError(
                self.log,
                "The 'data_folder' directory does not exist. Check the given path [%s].",
                self.data_folder,
            )

#        self._fsky = fsky
        fsky = 0.52
        
        #Binning (fixed binning)
        binc = tools.get_binning()
        
        #Data
        self.log.debug("Reading cross-spectrum")
        filepath = os.path.join(self.data_folder,self.clfile)
        data = fits.getdata(filepath)
        datab = binc.bin_spectra( data)
        
        #Fiducial spectrum
        self.log.debug("Reading model")
        filepath = os.path.join(self.data_folder,self.fiducialfile)
        clsim = fits.getdata(filepath)
        fid = binc.bin_spectra( clsims)
        
        #covmat
        self.log.debug("Reading covariance")
        filepath = os.path.join(self.data_folder,self.clcovfile)
        clcov = self._read_dl_data(filepath)
        self.invcov = np.linalg.inv(clcov)
        
        #compute offsets
        off = tools.compute_offsets( binc.lbin, diag(clcov).reshape(-1,binc.nbins), fid, fsky=fsky)
        
        #construct BB likelihood
        self.data = datab[2]
        self.off = off[2]
        self.fid = fid[2]
        
    def compute_likelihood( self, cl):
        x = (self.data+self.off)/(cl+self.off)
        g = sign(x)*tools.ghl( abs(x))
        
        X = (sqrt(self.fid+self.off)) * g * (sqrt(self.fid+self.off))
        
        chi2 = self.np.dot( X.transpose(),self.np.dot(self.invcov, X))
        
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

        self.data_folder = os.path.join(data_file_path, self.data_folder)
        if not os.path.exists(self.data_folder):
            raise LoggedError(
                self.log,
                "The 'data_folder' directory does not exist. Check the given path [%s].",
                self.data_folder,
            )

#        self._fsky = fsky
        fsky = 0.5
        
        #Binning (fixed binning)
        binc = tools.get_binning()
        
        #Data
        self.log.debug("Reading cross-spectrum")
        filepath = os.path.join(self.data_folder,self.clfile)
        data = fits.getdata(filepath)
        datab = binc.bin_spectra( data)
        
        #Fiducial spectrum
        self.log.debug("Reading model")
        filepath = os.path.join(self.data_folder,self.fiducialfile)
        clsim = fits.getdata(filepath)
        fid = binc.bin_spectra( clsims)
        
        #covmat
        self.log.debug("Reading covariance")
        filepath = os.path.join(self.data_folder,self.clcovfile)
        clcov = self._read_dl_data(filepath)
        if self.rcond != 0.:
            self.invcov = np.linalg.pinv(covariance,rcond)
        else:
            self.invcov = np.linalg.inv(covariance)
        
        #compute offsets
        off = tools.compute_offsets( binc.lbin, diag(clcov).reshape(-1,binc.nbins), fid, fsky=fsky)
        off[3:] = 0. #force NO offsets for TE,TB, EB
        
        #construct likelihood
        self.isEB = True
        self.rcond=1e-9
        self.data = datab
        self.off = off
        self.fid = fid
        
    def compute_likelihood( self, cl):
        nel = len(self.data[0])
        ndim = 3 if self.isEB else 2
        
        x = zeros( (ndim, nel))
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
