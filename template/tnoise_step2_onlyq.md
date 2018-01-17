## -- coding: utf-8 --

<h1>${title}</h1>

This document contains a preliminary analysis of the noise temperature for
the Strip polarimeter ${polarimeter_name}.

The report has been generated on ${analysis_date} using striptun
v${striptun_version} (commit
[${latest_git_commit[0:6]}](https://github.com/lspestrip/striptun/commit/${latest_git_commit})).
Data have been taken from
% if 'http' in test_file_name:
[${test_file_name}](${test_file_name})
% else:
${test_file_name}
% endif

The model assumes that the gains of the two legs of the polarimeters have been
balanced. The mathematical model used is the following (assuming PHSW state to
be `${phsw_state}`):
% if phsw_state in ('0101', '1010'):
$$ \begin{align} Q_1 & = G_{Q1} \left(T_a + N_{Q1}\right), \\ U_1 & = G_{U1} \left(\frac{T_a + T_b}2 + N_{U1}\right), \\ U_2 & = G_{U2} \left(\frac{T_a  + T_b}2 + N_{U2}\right), \\ Q_2 & = G_{Q2} \left(T_b + N_{Q2}\right), \end{align} $$
% else:
$$ \begin{align} Q_1 & = G_{Q1} \left(T_b + N_{Q1}\right), \\ U_1 & = G_{U1} \left(\frac{T_a + T_b}2 + N_{U1}\right), \\ U_2 & = G_{U2} \left(\frac{T_a  + T_b}2 + N_{U2}\right), \\ Q_2 & = G_{Q2} \left(T_a + N_{Q2}\right), \end{align} $$
% endif
where $T_a$ and $T_b$ are the overall temperature signals entering the ports A
and B of the magic-tee, $\varepsilon$ is the unbalance between the two legs of
the polarimeter,$G_{Q1}$, $G_{Q2}$, $G_{U1}$ and $G_{U2}$ are the gains (in
ADU/K) of the four outputs, and $N_{Q1}$, $N_{U1}$, $N_{U2}$, and $N_{Q2}$ are
the noise temperature. One of the Q1/Q2 channels is typically
&laquo;blind&raquo;, which means that the code will not be able to estimate its
parameters (gains and noise temperature). In this case, they will be set to zero.

<h2>Test data</h2>

The following plots show the behaviour of the detectors' output when the load is
varied. The first plot compares the measured output (corrected for the HEMT-off
biases) with the estimates of the analytical model reported above; the dots and
strong lines are the measurements, while the faint lines are the expectations
from the model. The second plot shows the discrepancy between the measured
output and the analytical model, converted from ADU to K in order to make the
four lines comparable.

![](temperature_timestream.svg){: class="plot"}

The following offsets have been applied to the four outputs:

# Detector | Offset
:---------:| ----------:
PWR0 (Q1)  | ${ offsets[0] }
PWR1 (U1)  | ${ offsets[1] }
PWR2 (U2)  | ${ offsets[2] }
PWR3 (Q2)  | ${ offsets[3] }


<h2>Results of the analysis</h2>

<h3>Overall fit</h3>

In this section we consider the full set of temperature steps. The estimate for the
noise temperature comes from the non-blind Q detector.

# Parameter | Gaussian estimate
:----------:| -----------------:
Noise temperature [K] | \
   ${ '{0:.1f}'.format(tnoise['mean']) } &pm; ${ '{0:.1f}'.format(tnoise['std']) }
Q1 gain (PWR0) [ADU/K] | \
   ${ '{0:.0f}'.format(gain_q1['mean']) } &pm; ${ '{0:.0f}'.format(gain_q1['std']) }
U1 gain (PWR1) [ADU/K] | \
   ${ '{0:.0f}'.format(gain_u1['mean']) } &pm; ${ '{0:.0f}'.format(gain_u1['std']) }
U2 gain (PWR2) [ADU/K] | \
   ${ '{0:.0f}'.format(gain_u2['mean']) } &pm; ${ '{0:.0f}'.format(gain_u2['std']) }
Q2 gain (PWR3) [ADU/K] | \
   ${ '{0:.0f}'.format(gain_q2['mean']) } &pm; ${ '{0:.0f}'.format(gain_q2['std']) }
  
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


% if y_factor_estimates:

<h2>Y-factor analysis</h2>

In this section we investigate the result of the classical Y-factor analysis, done on all the
pairs of temperatures. The outcome of this analysis is no longer one value for $N$, but rather
as many estimates as the number of pairs. The following plot provides a visual representation
of all the estimates. They are calculated as the $x$ coordinate of the intersection point
between the $x$ axis and the line connecting two points on the $T \times \mathrm{PWR}$ plane
(represented by the horizontally-aligned grey points):

![](tnoise_estimates_from_y_factor.svg){: class="plot"}

The following matrix plot and table detail the pattern of values for the noise temperatures
$N$ shown above:

![](tnoise_matrix.svg){: class="plot"}

#   | Detector | T<sub>1</sub> [K] | T<sub>2</sub> [K] | PWR<sub>1</sub> | PWR<sub>2</sub> | N [K] | 
:--:| --------:|------------------:| -----------------:| ---------------:| ---------------:| -----:|
% for cur_y_estimate in y_factor_estimates:
${loop.index + 1} | \
  ${ cur_y_estimate['detector_name'] } | \
  ${ '{0:.1f}'.format(cur_y_estimate['temperature_1']) } | \
  ${ '{0:.1f}'.format(cur_y_estimate['temperature_2']) } | \
  ${ '{0:.0f}'.format(cur_y_estimate['output_1']) } | \
  ${ '{0:.0f}'.format(cur_y_estimate['output_2']) } | \
  ${ '{0:.1f}'.format(cur_y_estimate['tnoise']) }
% endfor

% endif
