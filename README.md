LoLLiPoP: Low-L Likelihood Polarized for Planck
================================================

``Lollipop`` is a Planck low-l polarization likelihood based on cross-power-spectra for which the bias is zero when the noise is uncorrelated between maps. It uses the approximation presented in Hamimeche & Lewis (2008), modified as described in Mangilli et al. (2015) to apply to cross-power spectra.

It was previously applied to Planck EE data for investigating the reionization history in [Planck Collaboration Int. XLVII (2016)](https://arxiv.org/abs/1605.03507) and in [Tristram et al. (2020)](https://arxiv.org/abs/2010.01139) for estimating constraints on the tensor-to-scalar ratio.

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