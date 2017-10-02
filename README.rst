STRIP polarimeter tuning
========================

This repository contains a Python program which reads the data acquired with the
Keithley machine used at the Bicocca labs and produces a report containing the
tuning parameters of the six HEMTs used in Strip's polarimeters.

The program does not need to be installed. Just run it with the following
parameters::

     python striptun.py POLARIMETER_NAME INPUT_PATH OUTPUT_PATH

The meaning of the parameters is the following:

- ``POLARIMETER_NAME`` is a string identifying the polarimeter (e.g., ``STRIP07``)
- ``INPUT_PATH`` is the name of the directory containing the files produced by
  Keithley (usually the name is ``prove_DC``, or something similar)
- ``OUTPUT_PATH`` is the name of the directory that will contain the report (if the
  directory does not exist, it will be created silently)

At the end of the execution, the directory ``OUTPUT_PATH`` will contain a file named
``index.html``, which can be opened using any web browser (e.g., Firefox).


How the program works
---------------------

The program runs the following steps:

1. Plot Vd versus Id
2. Compute TransHEMT (see below) and plot Vd versus TransHEMT
3. Plot Vd versus Ig
4. Plot Vg versus Id
5. Set Vd to 0.9 V and find Id ~ 4 mA (Q1, Q6), Id ~ 7 mA (Q2, Q5)
6. Find TransHEMT for the Vd and Id found in the previous step 
7. Set Vd to 0.9 V and find Id ~ 6 mA for Q3
8. TransHEMT_Q4 = TransHEMT_Q1 * TransHEMT_Q2 * TransHEMT_Q3 / (TransHEMT_Q6 * TransHEMT_Q5)
9. From TransHEMT_Q4 and Vd = 0.9 V derive Id for Q4 
10. Generate a report

Transconductance
----------------

The formula used to compute transconductance is the following::

    TransHEMT = 1e3 * (Id2 - Id1) / ((Vg2 - 1e5 * Ig2) - (Vg1 - 1e5 * Ig1))

where pairs like ``(Id1, Id2)`` refer to two consecutive lines with different values of Vg.
The presence of the factor 1e5 seems to be due to the partitor.
