## -- coding: utf-8 --

<h1>${title}</h1>

This document contains the analysis of the frequency response of the STRIP polarimeter ${polarimeter_name}.

The report has been generated on ${analysis_date} using striptun v${striptun_version} (commit `${latest_git_commit}`). 

We measured the bandpasses for each detector of the polarimeter.
In particular, we used a signal generator to inject a signal at frequencies in the range ${low_frequency} - ${high_frequency} GHz (steps of 0.1 GHz) into the receiver and measure the output signal as a function of frequency.

<h2>Results</h2>

In the following figures we report the frequency response of the four detectors:

% for cur_results in detailed_results:
![](${polarimeter_name}_RFtest_${cur_results['PSStatus']}_${loop.index}.svg){: class="plot", width=70% }
% endfor

<h2>Bandwidth and central frequency estimation for the four detectors in different phase-switch configurations:</h2>

% for cur_results in detailed_results:

**File ${loop.index}. Phase-switch configuration: ${cur_results['PSStatus']}**

DETECTOR  | Bandwidth [GHz]          | Central Frequency [GHz]          
--------- |:--------------------:|:-------------------:
PW0/Q1   | ${'{0:.2f}'.format(cur_results['PW0Q1']['bandwidth'])} | ${'{0:.2f}'.format(cur_results['PW0Q1']['central_nu'])} 
PW1/U1   | ${'{0:.2f}'.format(cur_results['PW1U1']['bandwidth'])} | ${'{0:.2f}'.format(cur_results['PW1U1']['central_nu'])}
PW2/U2   | ${'{0:.2f}'.format(cur_results['PW2U2']['bandwidth'])} | ${'{0:.2f}'.format(cur_results['PW2U2']['central_nu'])}
PW3/Q2   | ${'{0:.2f}'.format(cur_results['PW3Q2']['bandwidth'])} | ${'{0:.2f}'.format(cur_results['PW3Q2']['central_nu'])}

% endfor

<h2>Final bandwidth and central frequency estimation</h2>
In the following figures we plot:

- The frequency response of the 3 "non-blind" detectors for all the analyzed files together with the final bandwidth. 
  All the courves are normalized to range 0-1.
  The final band is also plotted. We estimate it as the median of the different measurementes at each frequency.

- The final band alone together with its error bar (95% C.L.)

![](${polarimeter_name}_RFtest_AllDetNorm.svg){: class="plot", width=70%}
![](${polarimeter_name}_RFtest_FinalBand.svg){: class="plot", width=70%}

<div style="text-align:center; color:red; font-size: 1.2em;">Final Bandwidth: ${'{0:.2f}'.format(final_bandwidth)} ± ${'{0:.2f}'.format(final_bandwidth_err)} GHz (95% C.L.)</div>
<div style="text-align:center; color:red; font-size: 1.2em;">Central Frequency: ${'{0:.2f}'.format(final_central_nu)} ± ${'{0:.2f}'.format(final_central_nu_err)} GHz (95% C.L.)</div>



