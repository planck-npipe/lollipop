import os
import tempfile
import unittest
import camb
from astropy.utils import minversion

packages_path = os.environ.get("COBAYA_PACKAGES_PATH") or os.path.join(
    tempfile.gettempdir(), "Lollipop_packages"
)

cosmo_params = {
    "H0": 67.66,
    "As": 2.088434844099595e-09,
    "ombh2": 0.02226,
    "omch2": 0.1188,
    "ns": 0.9680,
    "tau": 0.0580,
}

params = {'A_planck': 0.9996}

if minversion(camb, "2.0.0"):
    chi2s = {"lowlB": 30.21, "lowlE": 32.85, "lowlEB": 101.33}
else:
    chi2s = {"lowlB": 30.74, "lowlE": 36.47, "lowlEB": 104.97}
    

class LollipopTest(unittest.TestCase):
    def setUp(self):
        from cobaya.install import install

        install(
            {"likelihood": {"planck_2020_lollipop.lowlEB": None}},
            path=packages_path,
            skip_global=True,
        )

    def test_lollipop(self):
        import camb
        import planck_2020_lollipop

        camb_cosmo = cosmo_params.copy()
        camb_cosmo.update({"lmax": 30, "lens_potential_accuracy": 1})
        pars = camb.set_params(**camb_cosmo)
        results = camb.get_results(pars)
        powers = results.get_cmb_power_spectra(pars, CMB_unit="muK", raw_cl=True)
        cl_dict = {k: powers["total"][:, v] for k, v in {"ee": 1, "bb": 2}.items()}

        for mode, chi2 in chi2s.items():
            _llp = getattr(planck_2020_lollipop, mode)
            my_lik = _llp({"packages_path": packages_path})
            loglike = my_lik.loglike(cl_dict, **params)
            self.assertLess( abs(-2 * loglike - chi2), 1)
            
##     def test_cobaya(self):
##         for mode, chi2 in chi2s.items():
##             info = {
##                 "debug": True,
##                 "likelihood": {"planck_2020_lollipop.{}".format(mode): None},
##                 "theory": {"camb": {"extra_args": {"lens_potential_accuracy": 1}}},
##                 "params": dict(**cosmo_params,**params),
##                 "packages_path": packages_path,
##             }
##             from cobaya.model import get_model

##             model = get_model(info)
##             self.assertLess( abs(-2 * model.loglikes({})[0][0] - chi2), 1)
