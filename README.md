LoLLiPoP: Low-L Likelihood Polarized for Planck
================================================

``Lollipop`` is a Planck low-l polarization likelihood based on cross-power-spectra for which the bias is zero when the noise is uncorrelated between maps. It uses the approximation presented in Hamimeche & Lewis (2008), modified as described in Mangilli et al. (2015) to apply to cross-power spectra.

It was previously applied to Planck EE data for investigating the reionization history in Planck Collaboration Int. XLVII (2016) and in Tristram et al. (2020) for estimating constraints on the tensor-to-scalar ratio.

The model consists of a linear combination of the CMB power spectrum and several foregrounds residuals. These are:
- Galactic dust (estimated directly from the 353 GHz channel);
- the cosmic infrared background (as measured in [Planck Collaboration XXX 2014](https://arxiv.org/abs/1309.0382));
- thermal Sunyaev-Zeldovich emission (based on the Planck measurement reported in [Planck Collaboration XXI 2014](https://arxiv.org/abs/1303.5081));
- kinetic Sunyaev-Zeldovich emission, including homogeneous and patchy reionization components from [Shaw et al. (2012)](https://arxiv.org/abs/1109.0553) and [Battaglia et al. (2013)](https://arxiv.org/abs/1211.2832);
- a tSZ-CIB correlation consistent with both models above; and
- unresolved point sources as a Poisson-like power spectrum with two components (extragalactic radio galaxies and infrared dusty galaxies).

HiLLiPoP has been used as an alternative to the public Planck likelihood in the 2013 and 2015 Planck releases [[Planck Collaboration XV 2014](https://arxiv.org/abs/1303.5075); [Planck Collaboration XI 2016](https://arxiv.org/abs/1507.02704)], and is described in detail in [Couchot et al. (2017)](https://arxiv.org/abs/1609.09730).

It is interfaced with the ``cobaya`` MCMC sampler.

Requirements
------------
* Python >= 3.5
* `healpy`
* `numpy`
* `scipy`
* `astropy`

Install
-------

Likelihood versions
-------------------

* Planck 2018 (PR3)
* Planck 2020 (PR4)