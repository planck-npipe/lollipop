LoLLiPoP: Low-L Likelihood Polarized for Planck
================================================

![GitHub Workflow Status](https://img.shields.io/github/workflow/status/planck-npipe/lollipop/Unit%20test)
![https://pypi.python.org/pypi/planck-2020-lollipop](https://img.shields.io/pypi/v/planck-2020-lollipop)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)

``Lollipop`` is a Planck low-l polarization likelihood based on cross-power-spectra for which the
bias is zero when the noise is uncorrelated between maps. It uses the approximation presented in
[Hamimeche & Lewis (2008)](https://arxiv.org/abs/0801.0554), modified as described in [Mangilli et
al. (2015)](https://arxiv.org/abs/1503.01347) to apply to cross-power spectra.  This version is
based on the Planck PR4 data. Cross-spectra are computed on the CMB maps from Commander component
separation applied on each detset-split Planck frequency maps.

It was previously applied and described in
- [Planck Collaboration Int. XLVII (2016)](https://arxiv.org/abs/1605.03507) for investigating the
  reionization history,
- [Tristram et al. (2020)](https://arxiv.org/abs/2010.01139) for estimating constraints on the
  tensor-to-scalar ratio.

It is interfaced with the ``cobaya`` MCMC sampler.

Requirements
------------
* Python >= 3.5
* `numpy`
* `astropy`

Install
-------

The easiest way to install the `Lollipop` likelihood is *via* `pip`

```shell
$ pip install planck-2020-lollipop [--user]
```

If you plan to dig into the code, it is better to clone this repository to some location

```shell
$ git clone https://github.com/planck-npipe/lollipop.git /where/to/clone
```

Then you can install the `Lollipop` likelihoods and its dependencies *via*

```shell
$ pip install -e /where/to/clone
```

The ``-e`` option allow the developer to make changes within the `Lollipop` directory without having
to reinstall at every changes. If you plan to just use the likelihood and do not develop it, you can
remove the ``-e`` option.

Installing Lollipop likelihood data
-----------------------------------

You should use the `cobaya-install` binary to automatically download the data needed by the
`lollipop.lowlE` or `lollipop.lowlB` or `lollipop.lowlEB` likelihoods

```shell
$ cobaya-install /where/to/clone/examples/test_lollipop.yaml -p /where/to/put/packages
```

Data and code such as [CAMB](https://github.com/cmbant/CAMB) will be downloaded and installed within
the ``/where/to/put/packages`` directory. For more details, you can have a look to `cobaya`
[documentation](https://cobaya.readthedocs.io/en/latest/installation_cosmo.html).


Likelihood versions
-------------------

* ``lowlE``
* ``lowlB``
* ``lowlEB``
