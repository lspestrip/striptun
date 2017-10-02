# ${title}

This document contains the result of the tuning of the six amplifiers in
polarimeter ${polarimeter}.

The report has been generated on ${date}.

[TOC]


## Tuning table

Best parameters found for the tuning:

HEMT    | Idrain [mA] | Vdrain [mV] | Igate [mA] | Vgate [mV] | Transconductance [mA/V]
------- |:-----------:|:-----------:|:----------:|:----------:|:----------------------:
Q1/H0   | ${q1_id}    | ${q1_vd}    | ${q1_ig}   | ${q1_vg}   | ${q1_transconductance}
Q6/H1   | ${q6_id}    | ${q6_vd}    | ${q6_ig}   | ${q6_vg}   | ${q6_transconductance}
Q2/H2   | ${q2_id}    | ${q2_vd}    | ${q2_ig}   | ${q2_vg}   | ${q2_transconductance}
Q5/H3   | ${q5_id}    | ${q5_vd}    | ${q5_ig}   | ${q5_vg}   | ${q5_transconductance}
Q3/H4   | ${q3_id}    | ${q3_vd}    | ${q3_ig}   | ${q3_vg}   | ${q3_transconductance}
Q4/H5   | ${q4_id}    | ${q4_vd}    | ${q4_ig}   | ${q4_vg}   | ${q4_transconductance}

The product of the transconductances across the two legs is:

1. ${leg1_transconductance} (Q1 → Q2 → Q3)
1. ${leg1_transconductance} (Q6 → Q5 → Q4)


## Amplifier pair Q1/Q6

HEMT    | Idrain [mA] | Vdrain [mV] | Igate [mA] | Vgate [mV] | Transconductance [mA/V]
------- |:-----------:|:-----------:|:----------:|:----------:|:----------------------:
Q1/H0   | ${q1_id}    | ${q1_vd}    | ${q1_ig}   | ${q1_vg}   | ${q1_transconductance}
Q6/H1   | ${q6_id}    | ${q6_vd}    | ${q6_ig}   | ${q6_vg}   | ${q6_transconductance}

### Q1/H0

![](id_vs_vd_q1.svg){: class="plot"}

![](trans_hemt_vs_vd_q1.svg){: class="plot"}

![](id_vs_vg_q1.svg){: class="plot"}

### Q6/H1

![](id_vs_vd_q6.svg){: class="plot"}

![](trans_hemt_vs_vd_q6.svg){: class="plot"}

![](id_vs_vg_q6.svg){: class="plot"}



## Amplifier pair Q2/Q5

HEMT    | Idrain [mA] | Vdrain [mV] | Igate [mA] | Vgate [mV] | Transconductance [mA/V]
------- |:-----------:|:-----------:|:----------:|:----------:|:----------------------:
Q2/H2   | ${q2_id}    | ${q2_vd}    | ${q2_ig}   | ${q2_vg}   | ${q2_transconductance}
Q5/H3   | ${q5_id}    | ${q5_vd}    | ${q5_ig}   | ${q5_vg}   | ${q5_transconductance}

Here are a few plots:

### Q2/H2

![](id_vs_vd_q2.svg){: class="plot"}

![](trans_hemt_vs_vd_q2.svg){: class="plot"}

![](id_vs_vg_q2.svg){: class="plot"}


### Q5/H3

![](id_vs_vd_q5.svg){: class="plot"}

![](trans_hemt_vs_vd_q5.svg){: class="plot"}

![](id_vs_vg_q5.svg){: class="plot"}



## Amplifier pair Q3/Q4

HEMT    | Idrain [mA] | Vdrain [mV] | Igate [mA] | Vgate [mV] | Transconductance [mA/V]
------- |:-----------:|:-----------:|:----------:|:----------:|:----------------------:
Q3/H4   | ${q3_id}    | ${q3_vd}    | ${q3_ig}   | ${q3_vg}   | ${q3_transconductance}
Q4/H5   | ${q4_id}    | ${q4_vd}    | ${q4_ig}   | ${q4_vg}   | ${q4_transconductance}


### Q3/H4

![](id_vs_vd_q3.svg){: class="plot"}

![](trans_hemt_vs_vd_q3.svg){: class="plot"}

![](id_vs_vg_q3.svg){: class="plot"}


### Q4/H5

![](id_vs_vd_q4.svg){: class="plot"}

![](trans_hemt_vs_vd_q4.svg){: class="plot"}

![](id_vs_vg_q4.svg){: class="plot"}
