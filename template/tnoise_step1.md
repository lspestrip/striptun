## -- coding: utf-8 --

<h1>${title}</h1>

This document contains a preliminary analysis of the noise temperature for
the Strip polarimeter ${polarimeter}. The analysis assumes that the samples
have been acquired using a sampling frequency equal to ${sampling_frequency} Hz.

The report has been generated on ${analysis_date} using striptun
v${striptun_version} (commit `${latest_git_commit}`).

[TOC]

% for idx in (0, 1, 2, 3):

<h2>Detector PWR${idx}</h2>

% if idx == blind_channel:
This detector was blind because of the PH/SW settings used during the test.
% endif

![](plot_pwr${idx}.svg){: class="plot"}

## Include the table with the stability regions only if the detector was not blind
% if idx != blind_channel:
The following table lists the region where the signal is stable enough to
run an analysis of the noise temperature. The «index» is the zero-based index
of the sample in the file. The average output and the RMS of the output are
reported as well.

  # | Start index | End index | Start time [s] | End time [s] | Length [s] | Average [ADU] | RMS [ADU]
:--:| -----------:| ---------:| --------------:| -----------: | ---------: | -------------:| --------:
    % for region in regions[idx]:
${loop.index + 1} | \
${region.index0} | ${region.index1} | \
${region.time0_s} | ${region.time1_s} | \
${region.time1_s - region.time0_s} | \
${ '{0:.0f}'.format(region.mean_output_adu) } | \
${ '{0:.1f}'.format(region.rms_output_adu) }
    % endfor
% endif

% endfor
