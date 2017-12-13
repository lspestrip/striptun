## -- coding: utf-8 --

<h1>${title}</h1>

This document contains a preliminary analysis of the noise temperature for
the Strip polarimeter ${polarimeter_name}.

The report has been generated on ${analysis_date} using striptun
v${striptun_version} (commit `${latest_git_commit}`).

Data have been taken from
% if 'http' in test_file_name:
[${test_file_name}](${test_file_name})
% else:
${test_file_name}
% endif

<h2>Results</h2>

![](temperature_timestream.svg){: class="plot"}

# Parameter | Gaussian estimate
:----------:| -----------------:
Noise temperature [K] | \
   ${ '{0:.1f}'.format(tnoise['mean']) } &pm; ${ '{0:.1f}'.format(tnoise['std']) }
Q1 gain (PWR0) [K/ADU] | \
   ${ '{0:.0f}'.format(gain_q1['mean']) } &pm; ${ '{0:.0f}'.format(gain_q1['std']) }
U1 gain (PWR1) [K/ADU] | \
   ${ '{0:.0f}'.format(gain_u1['mean']) } &pm; ${ '{0:.0f}'.format(gain_u1['std']) }
U2 gain (PWR2) [K/ADU] | \
   ${ '{0:.0f}'.format(gain_u2['mean']) } &pm; ${ '{0:.0f}'.format(gain_u2['std']) }
Q2 gain (PWR3) [K/ADU] | \
   ${ '{0:.0f}'.format(gain_q2['mean']) } &pm; ${ '{0:.0f}'.format(gain_q2['std']) }
Unbalance [%] | \
   ${ '{0:.1f}'.format(unbalance['mean'] * 100.0) } &pm; ${ '{0:.1f}'.format(unbalance['std'] * 100.0) }
  
The analysis has been done using data for ${len(steps)} temperature steps. Here are the details:

#   | T_A [K] | T_B [K] | PWR0 [ADU] | PWR1 [ADU] | PWR2 [ADU] | PWR3 [ADU]
:--:| -------:| -------:| ----------:| ----------:| ----------:| ----------:
    % for cur_step in steps:
${loop.index + 1} | \
${ '{0:.2f}'.format(cur_step['t_load_a_K']) } | \
${ '{0:.2f}'.format(cur_step['t_load_b_K']) } | \
${ '{0:.0f}'.format(cur_step['pwr0_adu']) } ± ${ '{0:.0f}'.format(cur_step['pwr0_rms_adu']) } | \
${ '{0:.0f}'.format(cur_step['pwr1_adu']) } ± ${ '{0:.0f}'.format(cur_step['pwr1_rms_adu']) } | \
${ '{0:.0f}'.format(cur_step['pwr2_adu']) } ± ${ '{0:.0f}'.format(cur_step['pwr2_rms_adu']) } | \
${ '{0:.0f}'.format(cur_step['pwr3_adu']) } ± ${ '{0:.0f}'.format(cur_step['pwr3_rms_adu']) }
    % endfor

<h3>Linear correlation between measures and the model</h3>

![](tnoise_linear_correlation.svg){: class="plot"}
