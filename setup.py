from setuptools import setup, find_namespace_packages

# see for guide on setting up repository
# https://docs.python-guide.org/writing/structure/
VERSION = "0.0.1"
DESCRIPTION = "Statistical Distributions for Fast Fitting"
LONG_DESCRIPTION = "A couple of distributions with skewness\
                    and kurtosis parameters that allow fast fitting"

# Setting up
setup(
    # the name must match the folder name 'FastDistributions'
    name="FastDistributions",
    version=0.9,
    author="Tim Wilding",
    author_email="tim_wilding@yahoo.com",
    url="https://github.com/TimWilding/FinanceDistributions",
    description=DESCRIPTION,
    long_description=LONG_DESCRIPTION,
    packages=find_namespace_packages(where="FastDistributions"),
    # add any additional packages that need installing
    install_requires=[
        "numpy",
        "matplotlib",
        "yfinance",
        "scipy",
        "scikit-learn",
        "Py-BOBYQA",
        "seaborn",
        "statsmodels",
        "cvxpy"
    ],
    package_dir={"": "FastDistributions"},
    package_data = {"interp_data": ["*"],},
    keywords=["python", "finance", "distributions", "utilities"],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Researchers",
        "Programming Language :: Python :: 3",
        "Operating System :: MacOS :: MacOS X",
        "Operating System :: Microsoft :: Windows",
    ],
)
