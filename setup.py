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
        install_requires=['numpy', 'matplotlib', 'yfinance', 'scipy'], # add any additional packages that 
        # needs to be installed along with your package. Eg: 'caer' -0 what about PyBobyQA
        
        keywords=['python', 'first package'],
        classifiers= [
            "Development Status :: 3 - Alpha",
            "Intended Audience :: Researchers",
            "Programming Language :: Python :: 3",
            "Operating System :: MacOS :: MacOS X",
            "Operating System :: Microsoft :: Windows",
        ]
)