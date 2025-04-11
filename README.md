 # Dynamical Survival Analyais + Edge-Based Compartimental Model

This repository contains code and datasets used in the research conducted for the paper: **"Inferring Contact Network Characteristics from Epidemic Data via Compact Mean-Field Models"**
Available at: https://arxiv.org/abs/2502.12040

In this work, we use the Edge-Based Compartmental Model (EBCM)—a compact and analytically tractable framework—and integrate it with Dynamical Survival Analysis (DSA) to infer both key network properties and epidemic parameters.
Within the DSA framework, we leverage the solution of a mean-field SIR model to relate the evolution of the susceptible population to a survival probability for the time of infection. Meanwhile, the EBCM uses the probability generating function (PGF) as input, enabling the model to capture essential structural features of the contact network through the parameters of the PGF.

Despite correlations between structural and epidemic parameters, our framework demonstrates robustness in accurately inferring contact network properties from synthetic epidemic simulations. Additionally, we apply the
framework to real-world outbreaks—the 2001 UK foot-and-mouth disease outbreak and the COVID-19 epidemic in Seoul— to estimate both disease parameters and network characteristics. Our results show that our framework achieves good fits to real-world epidemic 
data and reliable short-term forecasts. These findings highlight the potential of network-based inference approaches to uncover hidden contact structures, providing insights that can inform the design of targeted interventions and public health strategies.

## Structure of the Repository

The repository is devided into three folders:

### DSA_EBCM_Synthetic

This folder contains the python fucntions used to perform the ifnerence in synthetic data. The synthetic data comes from Gillespie simulation in static netowrks. In this folder we have the following files:

- **DSA_EBCM_synthetic.py**: Contains the core functions used to perform the inference. This includes the main process orchestration, the RAM function for running the MCMC algorithm, various likelihood function implementations, data preprocessing utilities, EBCM-specific functions, and other auxiliary routines. 
- **PGF.py** : Implements a set of probability generating functions (PGFs) and their derivatives for commonly used degree distributions, including Regular, Poisson, and Negative Binomial.
- **RAM.py**: Includes auxiliary functions required for the RAM (Robust Adaptive Metropolis) MCMC algorithm.
- **Posteiror_analysis_final.py**: Provides functions for analyzing the resulting posterior distribution after inference.
- **Final_code_synthetic.ipynb** : A Python notebook demonstrating a complete example of the inference process on synthetic data.

### DSA_EBCM_Real

This folder contains the Python functions used to perform inference on real-world data. The main difference from the synthetic case is that, for real data, the input is provided to the main function in a preprocessed form. This is necessary because different datasets typically require dataset-specific preprocessing steps.
Another key difference lies in the likelihood functions: for real data, the likelihoods omit terms related to the total population size, as this information is often unavailable.. 

- **DSA_EBCM_real.py**: Contains the core functions used to perform the inference. This includes the main process orchestration, the RAM function for running the MCMC algorithm, various likelihood function implementations, data preprocessing utilities, EBCM-specific functions, and other auxiliary routines. 
- **PGF.py** : Implements a set of probability generating functions (PGFs) and their derivatives for commonly used degree distributions, including Regular, Poisson, and Negative Binomial.
- **RAM.py**: Includes auxiliary functions required for the RAM (Robust Adaptive Metropolis) MCMC algorithm.
- **Posteiror_analysis_final.py**: Provides functions for analyzing the resulting posterior distribution after inference.
- **Final_code_real.ipynb** : A Python notebook demonstrating a complete example of the inference process on synthetic data.


### Synthetic_Datasets

This folder contains samples of synthetic data sets produced from Gillespie simualtions. There are two folders one for simulations in a network with Poisson degree distribution and one with Negative Binomial degree distribution. 
The paraemters used for the sumualtions are: beta=0.2, gamma=1, rho=0.001.
For the network structures: Poisson with average degree mu=10, and Negative binomial with average degree 10, r=1.




