from setuptools import find_packages, setup

setup(
    name="planck_2020_lollipop",
    version="0.1",
    description="A cobaya low-ell likelihood polarized for planck",
    zip_safe=True,
    packages=find_packages(),
    python_requires=">=3.5",
    install_requires=["astropy", "cobaya>=3.0"],
    package_data={"planck_2020_lollipop": ["lowl*.yaml", "lowl*.bibtex"]},
)
