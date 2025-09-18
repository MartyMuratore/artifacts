# Search and Parameter estimation in presence of glitches


This repository provides codes to re-produce the results of this paper:

https://arxiv.org/pdf/2505.19870

There are some dependences that are needed in order to run the codes provided. 

In particular to run the analysis of the Spritz pipeline Eryn, lisanalysis tool and BBhx which can be found in https://github.com/mikekatz04.

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

   searches : LDC2_spritz_analysis_shorter_segments.py

   parameter estimation:  LDC2_spritz_analysis_new.py 

5) additional files present and that are needed are:

   a) to generate synthetic noise (PSD): synthetic_noise_generator.py
   b) the codes for computing the maximum overlap between the MBHB and the glitch:  max_matching_glitch_MBHB.py

## Analyzing (Light-)Spritz data 



