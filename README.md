







# Search and Parameter estimation in presence of glitches

## Installation

### using uv as package manager
In this work we use the package manager `uv` to manage the environment and dependencies.

To install `uv`, please follow the instructions here: https://docs.astral.sh/uv/getting-started/installation/

### Installation using uv
The primary dependencies of this work are 

- BBHx: a GPU accelerated package for generating binary black hole waveforms and LISA response functions. It can be found here: https://github.com/mikekatz04/BBHx
- Eryn: a package for MCMC ensemble sampling with advanced moves and reversible jump capabilities. It can be found here: https://github.com/mikekatz04/Eryn
- lisatools: a package with LISA-specific analysis utilities including glitch modeling and sensitivity curves. It can be found here: https://github.com/mikekatz04/lisatools

First, locate the root directory of this project. Make sure that `uv` is installed and working. 

Then, create a new environment for this project by running:
```code bash
   uv venv
   source .venv/bin/activate
   uv pip install .
```
Then you need to install BBHx from source, clone the repository and follow the build instructions in the README. It is important that `lapack` and `pkgconfig` are installed on your system. 

Using brew, I added the following arguments prior to installing BBHx.
```code bash
   brew install lapack gsl pkgconf
   export LDFLAGS="-L/opt/homebrew/opt/lapack/lib -L/opt/homebrew/opt/gsl/lib" 
   export CPPFLAGS="-I/opt/homebrew/opt/lapack/include -I/opt/homebrew/opt/gsl/include"
   export CMAKE_ARGS="-DBBHX_LAPACKE_DETECT_WITH=PKGCONFIG -DBBHX_LAPACKE_FETCH=OFF -DCMAKE_PREFIX_PATH=/opt/homebrew/opt/lapack:/opt/homebrew/opt/gsl" 
   export PKG_CONFIG_PATH="/opt/homebrew/opt/lapack/lib/pkgconfig:/opt/homebrew/opt/gsl/lib/pkgconfig:$PKG_CONFIG_PATH"
```

Then, using uv, I located the root directory of `BBHx` and ran the following commands:
```code bash
   uv pip install -e .
````

This will automatically install `lisaanalysistools`. The package `Eryn` can be installed using

```code bash
   uv add eryn
```


## This repository
This repository provides codes to re-produce the results of this paper:

https://arxiv.org/pdf/2505.19870

There are some dependences that are needed in order to run the codes provided. 

In particular to run the analysis of the Spritz pipeline Eryn, lisanalysis tool and BBhx which can be found in https://github.com/mikekatz04. Python is also a requirement. To learn how to you use the sampler Eryn please see https://mikekatz04.github.io/Eryn/html/tutorial/Eryn_tutorial.html

To proceed we suggest to create a virtual environment and install all the necessary dependences.

'' conda create -n "test_env" ''

Important: for now install BBHx, lisatools and Eryn accordingly to requirement.txt

The use of GPU is also encouraged. 

There are different notebook present in the folder:

1) In case only the glith or shapelet waveform in time or frequency domain is need:
   glitch_shapelet_analytical_waveform.py

2) To produce glitch distributions from LISAPathfinder with or without instrumental noise with following reversible jump analysis approach (note that this notebook require this file glitch_params_all_PRDLPF.h5 which is not publically available. Contact the authors in case you need this file to generate your glitch distributions or alternative create your distributions):

   a) glitch_fitting_freq_doman_RJ.py

   b) The move used and developed for glitch searches is : group_stretch_proposal.py.

3) To only perform parameter estimation without reversible jump in presence of glitches, mbhb and noise

   glitch_fitting_mbhb_with_noise.py

     
   
4) the codes used for analysing Spritz data are:

   searches : LDC2_spritz_analysis_search.py

   parameter estimation:  LDC2_spritz_analysis_new.py 

5) additional files present and that are needed are:

   a) to generate synthetic noise (PSD): synthetic_noise_generator.py
   b) the codes for computing the maximum overlap between the MBHB and the glitch:  max_matching_glitch_MBHB.py





