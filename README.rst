STRIP polarimeter tuning
========================

This repository contains a Python program which reads the data acquired with the
Keithley machine used at the Bicocca labs and produces a report containing the
tuning parameters of the six HEMTs used in Strip's polarimeters.

Installation
------------

The program needs the following packages:

- `Mako <https://pypi.python.org/pypi/mako>`_ (templating engine)
- `Markdown <https://pypi.python.org/pypi/Markdown>`_ (markdown to HTML conversion)
- `Matplotlib <https://pypi.python.org/pypi/matplotlib>`_ (plotting library)
- `NumPy <https://pypi.python.org/pypi/numpy>`_ (vector/matrix operations)
- `SciPy <https://pypi.python.org/pypi/scipy>`_ (interpolation)
- `Xlrd <https://pypi.python.org/pypi/xlrd>`_ (Excel file importing)

If you are using `Anaconda Python <https://www.anaconda.com/>`_, you can install them 
using the command ``conda``::

    conda install mako markdown matplotlib numpy scipy xlrd

In any other case, you can use ``pip``::

    pip install mako markdown matplotlib numpy scipy xlrd

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

1. Read all the Excel files saved by Keithley

2. From the Vd versus Id curve, compute TransHEMT (see below) and plot Vd versus TransHEMT

3. Plot Vd versus Ig and Vg versus Id

4. Set Vd to 0.9 V and find Id ≈ 4 mA (Q1, Q6), Id ≈ 7 mA (Q2, Q5)

5. Find TransHEMT for the Vd and Id found in the previous step 

6. Set Vd to 0.9 V and find Id ≈ 6 mA for Q3

7. Find the configuration which maximizes the match between the product of the
   transconductances across the two legs (Q1→Q2→Q3 and Q6→Q5→Q4)

8. From TransHEMT_Q4 and Vd = 0.9 V derive Id for Q4 

9. Save all the plots

10. Generate a Markdown report and an HTML report


Transconductance
----------------

The formula used to compute transconductance is the following::

    TransHEMT = 1e3 * (Id2 - Id1) / ((Vg2 - Vg1) - 1e5 * (Ig2 - Ig1))

where pairs like ``(Id1, Id2)`` refer to two consecutive lines with different values of Vg.
The factor 1e5 is a resistance, and its presence seems to be due to the partitor.
