<a id="readme-top"></a>
<div align="center">
  <a href="https://github.com/TimWilding/FinanceDistributions">
    <img src="images/logo.png" alt="Logo" width="160" height="160">
  </a>
</div>

## Finance Distributions

Finance Distributions is a set of useful calculations and plots for statistical distribution used in financial models. There are a set of univariate 
distributions that include skewness and kurtosis parameters. There are some routines for calculating robust correlations, and some simple allocation 
recipes for using those routines.

The code can be installed by running python setup.py install. The following distributions are added:
* *LevyStableInterp* - an interpolated version of the Levy Stable distribution for fast fitting
* *GeneralisedSkewT* - an analytical distribution with skewness and fat tails that can be used for models
* *EntropyDistributions* - a generalisation of the Normal Distribution to include skewness and kurtosis based on the principles of maximum entropy
* *MeixnerDistribution* - a Generalised HyperSecant Distribution that includes skewness and kurtosis parameters. Includes a fast random variable generation using rejection sampling
* *Johnson SU* - the unbounded version of the Johnson distribution

* *fit_tail_model* - fits a Generalised Pareto Distribution to the tail of a particular sample (Extreme Value Theory)

* *PRIIPSCalcs* - calculations for PRIIPS statistics


There are also routines for calculation of useful multivariate statistics relating to correlation
* *Multivariate T-Distribution* - multivariate distribution including fat tails
* *correlations* - various adjustments for serial correlation in financial returns


And a few useful plotting routines
* *plot_qq* - multiple qq plots on a single figure
* *plot_hist* - multiple distribution comparisons on a single histogram
* *plot_log_cdf* - plots the Cumulative Density Function tails and compares to the sample

<!-- GETTING STARTED -->
<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Getting Started

Install Python locally.

### Prerequisites

Please make sure you have Python installed on your system.
* Python 3.10


### Installation

1. Clone the repo
   ```sh
   git clone https://github.com/TimWilding/FinanceDistributions.git
   ```
2. Install Python package
   ```sh
   pip install .
   ```
3. Start your Python coding
   ```sh
   # Fit Johnson SU distribution to a returns array
   import FastDistributions as fd
   jsu = fd.JohnsonSU.fitclass(ret_val) # sp_ret is an array of returns
   ```
<p align="right">(<a href="#readme-top">back to top</a>)</p>
