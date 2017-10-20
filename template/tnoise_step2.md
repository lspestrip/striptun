## -- coding: utf-8 --

<h1>${title}</h1>

This document contains a preliminary analysis of the noise temperature for
the Strip polarimeter ${polarimeter_name}.

The report has been generated on ${date}.

<h2>Results</h2>

  Parameter | Gaussian estimat
:----------:| -----------------:
Noise temperature [K] | \
   ${ '{0:.2f}'.format(tnoise['mean']) } ± ${ '{0:.2f}'.format(tnoise['std']) }
Polarimeter gain [K/ADU] | \
   ${ '{0:.2f}'.format(average_gain['mean']) } ± ${ '{0:.2f}'.format(average_gain['std']) }
Gain cross product [K/ADU] | \
   ${ '{0:.2f}'.format(gain_prod['mean']) } ± ${ '{0:.2f}'.format(gain_prod['std']) }

The analysis has been done using data for ${len(steps)} temperature steps. Here are the details:

#   | T_A [K] | T_B [K] | PWR0 [ADU] | PWR1 [ADU] | PWR2 [ADU] | PWR3 [ADU]
:--:| -------:| -------:| ----------:| ----------:| ----------:| ----------:
    % for cur_step in steps:
${loop.index + 1} | \
${ '{0:.2f}'.format(cur_step['t_load_a_K']) } | \
${ '{0:.2f}'.format(cur_step['t_load_b_K']) } | \
${ '{0:.2f}'.format(cur_step['pwr0_adu']) } ± ${ '{0:.2f}'.format(cur_step['pwr0_rms_adu']) } | \
${ '{0:.2f}'.format(cur_step['pwr1_adu']) } ± ${ '{0:.2f}'.format(cur_step['pwr1_rms_adu']) } | \
${ '{0:.2f}'.format(cur_step['pwr2_adu']) } ± ${ '{0:.2f}'.format(cur_step['pwr2_rms_adu']) } | \
${ '{0:.2f}'.format(cur_step['pwr3_adu']) } ± ${ '{0:.2f}'.format(cur_step['pwr3_rms_adu']) }
    % endfor

<h3>Linear correlation between measures and the model</h3>

![](tnoise_linear_correlation.svg){: class="plot"}

% if mcmc:

<h3>Convergence of the MC chain</h3>

The following parameters have been estimated using a Monte Carlo Markov Chain (MCMC) method.

  Parameter | Gaussian estimate | Median | Upper error | Lower error 
:----------:| -----------------:| ------:| ----------: | -----------:
Noise temperature [K] | \
   ${ '{0:.2f}'.format(mcmc_tnoise['mean']) } ± ${ '{0:.2f}'.format(mcmc_tnoise['std']) } | \
   ${ '{0:.2f}'.format(mcmc_tnoise['median']) } | \
   ${ '{0:.2f}'.format(mcmc_tnoise['upper_err_95CL']) } | \
   ${ '{0:.2f}'.format(mcmc_tnoise['lower_err_95CL']) }
Polarimeter gain [K/ADU] | \
   ${ '{0:.2f}'.format(mcmc_average_gain['mean']) } ± ${ '{0:.2f}'.format(mcmc_average_gain['std']) } | \
   ${ '{0:.2f}'.format(mcmc_average_gain['median']) } | \
   ${ '{0:.2f}'.format(mcmc_average_gain['upper_err_95CL']) } | \
   ${ '{0:.2f}'.format(mcmc_average_gain['lower_err_95CL']) }
Gain cross product [K/ADU] | \
   ${ '{0:.2f}'.format(mcmc_gain_prod['mean']) } ± ${ '{0:.2f}'.format(mcmc_gain_prod['std']) } | \
   ${ '{0:.2f}'.format(mcmc_gain_prod['median']) } | \
   ${ '{0:.2f}'.format(mcmc_gain_prod['upper_err_95CL']) } | \
   ${ '{0:.2f}'.format(mcmc_gain_prod['lower_err_95CL']) }


These are the parameters used to build and run the MCMC simulation:

Parameter | Value
:--------:| ----:
Number of walkers | ${num_of_walkers}
Number of iterations | ${num_of_iterations}
Length of the burn-in phase | ${burn_in_length}
Autocorrelation length (upper limit) | ${autocorrelation_length}
Number of independent samples | ${num_of_samples}

![](tnoise_corner_plot.svg){: class="plot"}

% endif