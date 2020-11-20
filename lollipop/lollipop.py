#
# LOLLIPOP
#
# Oct 2020   - M. Tristram -
import os
from typing import Optional

import astropy.io.fits as fits
import numpy as np
from cobaya.conventions import _packages_path
from cobaya.likelihoods._base_classes import _InstallableLikelihood
from cobaya.log import LoggedError
from cobaya.tools import are_different_params_lists

from lollipop import tools
from lollipop.bins import Bins

data_url = "https://portal.nersc.gov/project/cmb/planck2020/likelihoods"


class _LollipopLikelihood(_InstallableLikelihood):

    install_options = {"download_url": "{}/planck_2020_lollipop.tar.gz".format(data_url)}
    _mode = "EEBBEB"

    def initialize(self):

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

        # Binning (fixed binning)
        self.bins = tools.get_binning()

        # Data (ell,ee,bb,eb)
        self.log.debug("Reading cross-spectrum")
        filepath = os.path.join(self.data_folder, self.cl_file)
        data = tools.read_dl(filepath)
        self.cldata = self.bins.bin_spectra(data)

        # Fiducial spectrum (ell,ee,bb,eb)
        self.log.debug("Reading model")
        filepath = os.path.join(self.data_folder, self.fiducial_file)
        data = tools.read_dl(filepath)
        self.clfid = self.bins.bin_spectra(data)

        # covmat (ee,bb,eb)
        self.log.debug("Reading covariance")
        filepath = os.path.join(self.data_folder, self.cl_cov_file)
        clcov = fits.getdata(filepath)
        if self._mode == "EEBBEB":
            cbcov = tools.bin_covEB(clcov, self.bins)
        elif self._mode == "EE":
            cbcov = tools.bin_covEE(clcov, self.bins)
        elif self._mode == "BB":
            cbcov = tools.bin_covBB(clcov, self.bins)
        clvar = np.diag(cbcov).reshape(-1, self.bins.nbins)
        if self._mode == "EEBBEB":
            rcond = getattr(self, "rcond", 1e-9)
            self.invclcov = np.linalg.pinv(cbcov, rcond)
        else:
            self.invclcov = np.linalg.inv(cbcov)

        # compute offsets
        self.log.debug("Compute offsets")
        fsky = getattr(self, "fsky", 0.52)
        self.cloff = tools.compute_offsets(self.bins.lbin, clvar, self.clfid, fsky=fsky)
        self.cloff[2:] = 0.0  # force NO offsets EB

        self.log.info("Initialized!")

    def _compute_chi2_2fields(self, cl, **params_values):
        """
        Compute offset-Hamimeche&Lewis likelihood
        Input: Cl in muK^2
        """
        from numpy import diag, dot, sign, sqrt
        from numpy.linalg import eigh

        # get model in Cl, muK^2
        clth = np.array([self.bins.bin_spectra(cl[mode]) for mode in ["ee", "bb"]])

        nell = self.cldata.shape[1]
        x = np.zeros(self.cldata.shape)
        for ell in range(nell):
            O = tools.vec2mat(self.cloff[:, ell])
            D = tools.vec2mat(self.cldata[:, ell]) + O
            M = tools.vec2mat(clth[:, ell]) + O
            F = tools.vec2mat(self.clfid[:, ell]) + O

            # compute P = C_model^{-1/2}.C_data.C_model^{-1/2}
            w, V = eigh(M)
            #            if prod( sign(w)) <= 0:
            #                print( "WARNING: negative eigenvalue for l=%d" %l)
            L = dot(V, dot(diag(1.0 / sqrt(w)), V.transpose()))
            P = dot(L.transpose(), dot(D, L))

            # apply HL transformation
            w, V = eigh(P)
            g = sign(w) * tools.ghl(abs(w))
            G = dot(V, dot(diag(g), V.transpose()))

            # cholesky fiducial
            w, V = eigh(F)
            L = dot(V, dot(diag(sqrt(w)), V.transpose()))

            # compute C_fid^1/2 * G * C_fid^1/2
            X = dot(L.transpose(), dot(G, L))
            x[:, ell] = tools.mat2vec(X)

        # compute chi2
        x = x.flatten()
        chi2 = dot(x, dot(self.invclcov, x))

        return chi2

    def _compute_chi2_1field(self, cl, **params_values):
        """
        Compute offset-Hamimeche&Lewis likelihood
        Input: Cl in muK^2
        """
        from numpy import abs, dot, sign, sqrt

        # model in Cl, muK^2
        m = 0 if self._mode == "EE" else 1
        clth = self.bins.bin_spectra(cl[self._mode.lower()])

        x = (self.cldata[m] + self.cloff[m]) / (clth + self.cloff[m])
        g = sign(x) * tools.ghl(abs(x))

        X = (sqrt(self.clfid[m] + self.cloff[m])) * g * (sqrt(self.clfid[m] + self.cloff[m]))

        chi2 = dot(X.transpose(), dot(self.invclcov, X))

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

        return -0.5 * chi2

    @classmethod
    def get_path(cls, path):
        return os.path.realpath(os.path.join(path, "data"))


class lowlEB(_LollipopLikelihood):
    """
    Low-L Likelihood for Polarized Planck for EE+BB+EB
    Spectra-based likelihood based on Hamimeche-Lewis for cross-spectra
    applied on CMB component separated map
    """

    _mode = "EEBBEB"


class lowlE(_LollipopLikelihood):
    """
    Low-L Likelihood for Polarized Planck for EE
    Spectra-based likelihood based on Hamimeche-Lewis for cross-spectra
    applied on CMB component separated map
    """

    _mode = "EE"


class lowlB(_LollipopLikelihood):
    """
    Low-L Likelihood for Polarized Planck for BB
    Spectra-based likelihood based on Hamimeche-Lewis for cross-spectra
    applied on CMB component separated map
    """

    _mode = "BB"
