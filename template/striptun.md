## -- coding: utf-8 --

<h1>${title}</h1>

This document contains the result of the tuning of the six amplifiers in
polarimeter ${polarimeter}.

The report has been generated on ${date}.

[TOC]


<h2>Tuning table</h2>

Best parameters found for the tuning:

HEMT  | Idrain [mA] | Vdrain [mV] | Igate [mA] | Vgate [mV] | Transconductance [mA/V]
----- |:-----------:|:-----------:|:----------:|:----------:|:----------------------:
Q1/H0 | ${'{0:.2f}'.format(q1_id)} | ${'{0:.0f}'.format(q1_vd)} | ${'{0:.2f}'.format(q1_ig)} | ${'{0:.0f}'.format(q1_vg)} | ${'{0:.1f}'.format(q1_transconductance)}
Q6/H1 | ${'{0:.2f}'.format(q6_id)} | ${'{0:.0f}'.format(q6_vd)} | ${'{0:.2f}'.format(q6_ig)} | ${'{0:.0f}'.format(q6_vg)} | ${'{0:.1f}'.format(q6_transconductance)}
Q2/H2 | ${'{0:.2f}'.format(q2_id)} | ${'{0:.0f}'.format(q2_vd)} | ${'{0:.2f}'.format(q2_ig)} | ${'{0:.0f}'.format(q2_vg)} | ${'{0:.1f}'.format(q2_transconductance)}
Q5/H3 | ${'{0:.2f}'.format(q5_id)} | ${'{0:.0f}'.format(q5_vd)} | ${'{0:.2f}'.format(q5_ig)} | ${'{0:.0f}'.format(q5_vg)} | ${'{0:.1f}'.format(q5_transconductance)}
Q3/H4 | ${'{0:.2f}'.format(q3_id)} | ${'{0:.0f}'.format(q3_vd)} | ${'{0:.2f}'.format(q3_ig)} | ${'{0:.0f}'.format(q3_vg)} | ${'{0:.1f}'.format(q3_transconductance)}
Q4/H5 | ${'{0:.2f}'.format(q4_id)} | ${'{0:.0f}'.format(q4_vd)} | ${'{0:.2f}'.format(q4_ig)} | ${'{0:.0f}'.format(q4_vg)} | ${'{0:.1f}'.format(q4_transconductance)}

The product of the transconductances across the two legs for this tuning is:

1. ${leg1_transconductance} (Q1 → Q2 → Q3)
1. ${leg2_transconductance} (Q6 → Q5 → Q4)

The following list shows the list of all the solutions that have been considered
in the optimization, ranked by their balance (the first is the best, and it is
the one selected above). The «balance» is defined as the absolute value of the
difference between the product of the two transconductances.

Rank   | Id1 (Q3) | Id2 (Q4) | Q1 → Q2 → Q3 | Q6 → Q5 → Q4 | Balance
:-----:| ---:| ---:| ------------:| ------------:| ---------:
% for cur_solution in solutions:
${ loop.index + 1 } | \
${ '{0:.2f}'.format(cur_solution.q3_point.id) } mA | \
${ '{0:.2f}'.format(cur_solution.q4_point.id) } mA | \
${ '{0:.0f}'.format(cur_solution.leg1) } | \
${ '{0:.0f}'.format(cur_solution.leg2) } | \
${ '{0:.0f}'.format(cur_solution.balance) }
% endfor


<h2>Amplifier pair Q1/Q6</h2>

HEMT  | Idrain [mA] | Vdrain [mV] | Igate [mA] | Vgate [mV] 
----- |:-----------:|:-----------:|:----------:|:----------:
Q1/H0 | ${'{0:.2f}'.format(q1_id)} | ${'{0:.0f}'.format(q1_vd)} | ${'{0:.2f}'.format(q1_ig)} | ${'{0:.0f}'.format(q1_vg)}
Q6/H1 | ${'{0:.2f}'.format(q6_id)} | ${'{0:.0f}'.format(q6_vd)} | ${'{0:.2f}'.format(q6_ig)} | ${'{0:.0f}'.format(q6_vg)}

<h3>Q1/H0</h3>

![](id_vs_vd_q1.svg){: class="plot"}

![](trans_hemt_vs_vd_q1.svg){: class="plot"}

![](id_vs_vg_q1.svg){: class="plot"}

<h3>Q6/H1</h3>

![](id_vs_vd_q6.svg){: class="plot"}

![](trans_hemt_vs_vd_q6.svg){: class="plot"}

![](id_vs_vg_q6.svg){: class="plot"}



<h2>Amplifier pair Q2/Q5</h2>

HEMT  | Idrain [mA] | Vdrain [mV] | Igate [mA] | Vgate [mV] 
----- |:-----------:|:-----------:|:----------:|:----------:
Q2/H2 | ${'{0:.2f}'.format(q2_id)} | ${'{0:.0f}'.format(q2_vd)} | ${'{0:.2f}'.format(q2_ig)} | ${'{0:.0f}'.format(q2_vg)}
Q5/H3 | ${'{0:.2f}'.format(q5_id)} | ${'{0:.0f}'.format(q5_vd)} | ${'{0:.2f}'.format(q5_ig)} | ${'{0:.0f}'.format(q5_vg)}

Here are a few plots:

<h3>Q2/H2</h3>

![](id_vs_vd_q2.svg){: class="plot"}

![](trans_hemt_vs_vd_q2.svg){: class="plot"}

![](id_vs_vg_q2.svg){: class="plot"}


<h3>Q5/H3</h3>

![](id_vs_vd_q5.svg){: class="plot"}

![](trans_hemt_vs_vd_q5.svg){: class="plot"}

![](id_vs_vg_q5.svg){: class="plot"}



<h2>Amplifier pair Q3/Q4</h2>

HEMT  | Idrain [mA] | Vdrain [mV] | Igate [mA] | Vgate [mV] 
----- |:-----------:|:-----------:|:----------:|:----------:
Q3/H4 | ${'{0:.2f}'.format(q3_id)} | ${'{0:.0f}'.format(q3_vd)} | ${'{0:.2f}'.format(q3_ig)} | ${'{0:.0f}'.format(q3_vg)}
Q4/H5 | ${'{0:.2f}'.format(q4_id)} | ${'{0:.0f}'.format(q4_vd)} | ${'{0:.2f}'.format(q4_ig)} | ${'{0:.0f}'.format(q4_vg)}


<h3>Q3/H4</h3>

![](id_vs_vd_q3.svg){: class="plot"}

![](trans_hemt_vs_vd_q3.svg){: class="plot"}

![](id_vs_vg_q3.svg){: class="plot"}


<h3>Q4/H5</h3>

![](id_vs_vd_q4.svg){: class="plot"}

![](trans_hemt_vs_vd_q4.svg){: class="plot"}

![](id_vs_vg_q4.svg){: class="plot"}
