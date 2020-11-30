from setuptools import find_packages, setup

import versioneer

with open("README.md") as f:
    long_description = f.read()

setup(
    name="planck_2020_lollipop",
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
    description="A cobaya low-ell likelihood polarized for planck",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Matthieu Tristram",
    url="https://github.com/planck-npipe/lollipop",
    license="GNU license",
    classifiers=[
        "Intended Audience :: Developers",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Natural Language :: English",
        "Programming Language :: Python :: 3.5",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
    ],
    packages=find_packages(),
    python_requires=">=3.5",
    install_requires=["astropy", "cobaya>=3.0"],
    package_data={"planck_2020_lollipop": ["lowl*.yaml", "lowl*.bibtex"]},
)
