from setuptools import find_packages, setup

import versioneer

setup(
    name="planck_2020_lollipop",
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
    description="A cobaya low-ell likelihood polarized for planck",
    zip_safe=True,
    packages=find_packages(),
    python_requires=">=3.5",
    install_requires=["astropy", "cobaya>=3.0"],
    package_data={"planck_2020_lollipop": ["lowl*.yaml", "lowl*.bibtex"]},
)
