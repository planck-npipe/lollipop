#
# LOLLIPOP
#
# Oct 2020   - M. Tristram -
import os
from typing import Optional

import astropy.io.fits as fits
import numpy as np
from cobaya.conventions import data_path, packages_path_input
from cobaya.likelihoods.base_classes import InstallableLikelihood
from cobaya.log import LoggedError
from cobaya.tools import are_different_params_lists

from . import tools
from .bins import Bins

data_url = "https://portal.nersc.gov/cfs/cmb/planck2020/likelihoods"


class _LollipopLikelihood(InstallableLikelihood):
    install_options = {"download_url": f"{data_url}/planck_2020_lollipop.tar.gz"}

    def initialize(self):
        # Set path to data
        if (not getattr(self, "path", None)) and (not getattr(self, packages_path_input, None)):
            raise LoggedError(
                self.log,
                "No path given to Lollipop data. Set the likelihood property "
                f"'path' or the common property '{packages_path_input}'.",
            )

        # If no path specified, use the modules path
        data_file_path = os.path.normpath(
            getattr(self, "path", None) or os.path.join(self.packages_path, data_path)
        )

        self.data_folder = os.path.join(data_file_path, self.data_folder)
        if not os.path.exists(self.data_folder):
            raise LoggedError(
                self.log,
                "The 'data_folder' directory does not exist. Check the given path [%s].",
                self.data_folder,
            )

        # Setting mode given likelihood name
        likelihood_name = self.__class__.__name__
        self.mode = likelihood_name
        self.log.debug(f"mode = {self.mode}")
        if self.mode not in ["lowlE", "lowlB", "lowlEB"]:
            raise LoggedError(
                self.log,
                "The '{} likelihood is not currently supported. Check your likelihood name.",
                self.mode,
            )

        # Binning (fixed binning)
        self.bins = tools.get_binning(self.lmin,self.lmax)
        self.log.debug(f"lmax = {self.bins.lmax}")

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
        if self.mode == "lowlEB":
            cbcov = tools.bin_covEB(clcov, self.bins)
        elif self.mode == "lowlE":
            cbcov = tools.bin_covEE(clcov, self.bins)
        elif self.mode == "lowlB":
            cbcov = tools.bin_covBB(clcov, self.bins)
        clvar = np.diag(cbcov).reshape(-1, self.bins.nbins)

        if self.mode == "lowlEB":
            rcond = getattr(self, "rcond", 1e-9)
            self.invclcov = np.linalg.pinv(cbcov, rcond)
        else:
            self.invclcov = np.linalg.inv(cbcov)

        # Hartlap et al. 2008
        if self.hartlap_factor:
            if self.Nsim != 0:
                self.invclcov *= (self.Nsim - len(cbcov) - 2) / (self.Nsim - 1)

        if self.marginalised_over_covariance:
            if self.Nsim <= 1:
                raise LoggedError(
                    self.log,
                    "Need the number of MC simulations used to compute the covariance in order to marginalise over (Nsim>1).",
                )

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
        # get model in Cl, muK^2
        clth = np.array(
            [self.bins.bin_spectra(cl[mode]) for mode in ["ee", "bb", "eb"] if mode in cl]
        )

        cal = params_values["A_planck"] ** 2

        nell = self.cldata.shape[1]
        x = np.zeros(self.cldata.shape)
        for ell in range(nell):
            O = tools.vec2mat(self.cloff[:, ell])
            D = tools.vec2mat(self.cldata[:, ell]*cal) + O
            M = tools.vec2mat(clth[:, ell]) + O
            F = tools.vec2mat(self.clfid[:, ell]) + O

            # compute P = C_model^{-1/2}.C_data.C_model^{-1/2}
            w, V = np.linalg.eigh(M)
            #            if prod( sign(w)) <= 0:
            #                print( "WARNING: negative eigenvalue for l=%d" %l)
            L = V @ np.diag(1.0 / np.sqrt(w)) @ V.transpose()
            P = L.transpose() @ D @ L

            # apply HL transformation
            w, V = np.linalg.eigh(P)
            g = np.sign(w) * tools.ghl(np.abs(w))
            G = V @ np.diag(g) @ V.transpose()

            # cholesky fiducial
            w, V = np.linalg.eigh(F)
            L = V @ np.diag(np.sqrt(w)) @ V.transpose()

            # compute C_fid^1/2 * G * C_fid^1/2
            X = L.transpose() @ G @ L
            x[:, ell] = tools.mat2vec(X)

        # compute chi2
        x = x.flatten()
        if self.marginalised_over_covariance:
            chi2 = self.Nsim * np.log(1 + (x @ self.invclcov @ x) / (self.Nsim - 1))
        else:
            chi2 = x @ self.invclcov @ x

        self.log.debug(f"chi2/ndof = {chi2}/{len(x)}")
        return chi2

    def _compute_chi2_1field(self, cl, **params_values):
        """
        Compute offset-Hamimeche&Lewis likelihood
        Input: Cl in muK^2
        """
        # model in Cl, muK^2
        m = 0 if self.mode == "lowlE" else 1
        clth = self.bins.bin_spectra(cl["ee" if self.mode == "lowlE" else "bb"])

        cal = params_values["A_planck"] ** 2

        x = (self.cldata[m]*cal + self.cloff[m]) / (clth + self.cloff[m])
        g = np.sign(x) * tools.ghl(np.abs(x))

        X = (np.sqrt(self.clfid[m] + self.cloff[m])) * g * (np.sqrt(self.clfid[m] + self.cloff[m]))

        if self.marginalised_over_covariance:
            # marginalised over S = Ceff
            chi2 = self.Nsim * np.log(1 + (X @ self.invclcov @ X) / (self.Nsim - 1))
        else:
            chi2 = X @ self.invclcov @ X

        self.log.debug(f"chi2/ndof = {chi2}/{len(X)}")
        return chi2

    def get_requirements(self):
        return dict(Cl={mode: self.bins.lmax for mode in ["ee", "bb"]})

    def logp(self, **params_values):
        cl = self.provider.get_Cl(ell_factor=False)
        return self.loglike(cl, **params_values)

    def loglike(self, cl, **params_values):
        if self.mode == "lowlEB":
            chi2 = self._compute_chi2_2fields(cl, **params_values)
        elif self.mode in ["lowlE", "lowlB"]:
            chi2 = self._compute_chi2_1field(cl, **params_values)

        return -0.5 * chi2

    @classmethod
    def get_path(cls, path):
        if path.rstrip(os.sep).endswith(data_path):
            return path
        return os.path.realpath(os.path.join(path, data_path))

    @classmethod
    def is_installed(cls, **kwargs):
        if kwargs.get("data", True):
            path = cls.get_path(kwargs["path"])
            if not (
                cls.get_install_options() and os.path.exists(path) and len(os.listdir(path)) > 0
            ):
                return False
            if not os.path.exists(os.path.join(path, "planck_2020/lollipop")):
                return False
        return True


class lowlEB(_LollipopLikelihood):
    """
    Low-L Likelihood for Polarized Planck EE+BB+EB
    Spectra-based likelihood based on Hamimeche-Lewis for cross-spectra
    applied on CMB component separated map
    """


class lowlE(_LollipopLikelihood):
    """
    Low-L Likelihood for Polarized Planck EE
    Spectra-based likelihood based on Hamimeche-Lewis for cross-spectra
    applied on CMB component separated map
    """


class lowlB(_LollipopLikelihood):
    """
    Low-L Likelihood for Polarized Planck BB
    Spectra-based likelihood based on Hamimeche-Lewis for cross-spectra
    applied on CMB component separated map
    """
