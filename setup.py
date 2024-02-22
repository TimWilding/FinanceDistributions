from setuptools import setup, find_packages

VERSION = '0.0.1'
DESCRIPTION = 'Statistical Distributions for Fast Fitting'
LONG_DESCRIPTION = 'A couple of distributions with skewness and kurtosis parameters that allow fast fitting'

# Setting up
setup(
        # the name must match the folder name 'verysimplemodule'
        name="FastDistributions", 
        version=0.9,
        author="Tim Wilding",
        author_email="tim_wilding@yahoo.com",
        description=DESCRIPTION,
        long_description=LONG_DESCRIPTION,
        packages=find_packages(),
        # add any additional packages that need installing
        install_requires=['numpy', 'matplotlib', 'yfinance', 'scipy',' Py-BOBYQA', 'seaborn'],        
        keywords=['python', 'first package'],
        classifiers= [
            "Development Status :: 3 - Alpha",
            "Intended Audience :: Researchers",
            "Programming Language :: Python :: 3",
            "Operating System :: MacOS :: MacOS X",
            "Operating System :: Microsoft :: Windows",
        ]
)