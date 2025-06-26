# artifacts


The following python files are use to produce the results of the paper:

https://arxiv.org/pdf/2505.19870

If something is missing or not working I appreciated that you notify myself about this so I can fix it! Eryn and BBHx are required for running the notebook. The use of GPU is also encouraged.

In particular:

glitches and shapelet models can be found in: glitch_shapelet_analytical_waveform.py

the reversible jump MCMC code to search and fit for multiple glitches: glitch_fitting_freq_doman_RJ.py

the moves used and developed for glitch searches are: group_stretch_proposal.py

the code for doing PE with glitch and MBHB and noise:  glitch_fitting_mbhb_with_noise.py

the codes for computing the maximum overlap between the MBHB and the glitch:  max_matching_glitch_MBHB.py

the codes used for analysing Spritz data are:

for glitch search: LDC2_spritz_analysis_shorter_segments.py

for search and PE:  LDC2_spritz_analysis_new.py
