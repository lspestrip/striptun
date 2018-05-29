## -- coding: utf-8 --

<h1>${title}</h1>

This document contains the configuration of the HEMT biases used during the
testing of the Strip polarimeter ${polarimeter}. The report has been generated
on ${analysis_date} using striptun v${striptun_version} (commit
`${latest_git_commit}`).

[TOC]

Polarimeter ${polarimeter} is a ${band}-band polarimeter.
% if detector_type > 0:
The wiring of the HEMTs is of type ${detector_type}.
% endif

The following table lists the biases used during the tests:

HEMT     | Vd [mV]   | Id [mA]   | Vg [mV]   |
:-------:| ---------:| ---------:| ---------:|
H0 (HA1) | ${ha1_vd} | ${ha1_id} | ${ha1_vg} |
H1 (HB1) | ${hb1_vd} | ${hb1_id} | ${hb1_vg} |
H2 (HA2) | ${ha2_vd} | ${ha2_id} | ${ha2_vg} |
H3 (HB2) | ${hb2_vd} | ${hb2_id} | ${hb2_vg} |
H4 (HA3) | ${ha3_vd} | ${ha3_id} | ${ha3_vg} |
H5 (HB3) | ${hb3_vd} | ${hb3_id} | ${hb3_vg} |
