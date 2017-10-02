#!/usr/bin/env python3
# -*- encoding: utf-8 -*-

'''Read Keithley data and output the best tuning for a given Strip
polarimeter.'''

from argparse import ArgumentParser
import logging as log
import os
import os.path
from collections import namedtuple
from datetime import datetime
from shutil import copyfile
from string import Template
from typing import Any, Dict, List, Tuple

import xlrd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pylab as plt
import numpy as np
from scipy.interpolate import interp2d
from markdown import markdown

import excel_import as excel

TuningPoint = namedtuple('TuningPoint', [
    'vg',
    'vd',
    'ig',
    'id',
    'transconductance'
])


class HemtCurves:
    '''Base class for reading the data from a HEMT characterization Excel file

    The class must be initialized with the metadata and data table read from the
    Excel file produced by the Keithley machine. Use the functions in the
    "excel" package to create the dictionaries needed to call the __init__
    constructor.

    This function accepts a "parameter_name" parameter in its constructor. It is
    used to tell the object which is the parameter which has been varied during
    the test, and it must be one of the following strings:

    - ``Drain``
    - ``Source``
    - ``Gate``
    '''

    def __init__(self, metadata: Dict[str, Any], table: Dict[str, Any], parameter_name: str):
        self.metadata = metadata
        self.table = table

        # Perform a basic consistency check of the structure of the metadata
        assert tuple(metadata['Device Terminal']) == (
            'Drain', 'Source', 'Gate')

        self.parameter_name = parameter_name
        self.parameter_idx = metadata['Device Terminal'].index(
            self.parameter_name)
        self.num_of_curves = int(
            self.metadata['Number of Points'][self.parameter_idx])

    def get_array(self, curve_num: int, column_name: str, mul_factor: float):
        '''Return the array of values in one of the columns of the Excel file

        The parameter ``curve_num`` is a 0-based index of the parameter. It must
        be a value in the range [0, self.num_of_curves[.

        The parameter ``column_name`` must contain the base name of the parameter,
        like ``DrainV``: the column number will be appended automatically.

        The parameter ``mul_factor`` can be used to convert the measure unit of the
        parameter. For instance, passing ``1.0e3`` for a column containing voltages
        in Volt will make the function return values in mV.
        '''
        assert curve_num < self.num_of_curves
        return self.table['{0}({1})'.format(column_name, curve_num + 1)] * mul_factor

    def get_value(self, curve_num: int, column_name: str, mul_factor: float):
        '''Return the value associated with a curve.

        This works like self.get_array, but it must be used for columns where all
        elements have the same value (this is enforced by the code).'''
        assert curve_num < self.num_of_curves

        full_column_name = '{0}({1})'.format(column_name, curve_num + 1)
        array = self.table[full_column_name] * mul_factor
        # All the values in the array must be the same
        assert np.allclose(array, array[0]), \
            'Array "{0}" not constant: {1}'.format(full_column_name, array)

        return array[0]

    def create_plot(self, plot_file_name: str, hemt_name: str,
                    num_of_curves: int,
                    vg_range_mV: Tuple[float],
                    get_x: Any, get_y1: Any, get_y2: Any,
                    x_label: str, y1_label: str, y2_label: str,
                    point_x=None, point_y=None):
        '''Create a plot of the curves and save it in a PNG file.

        The parameters ``point_x`` and ``point_y``, if set, are the coordinates
        of a point to be marked in the plot. This is typically the setpoint
        found by a tuning calculation. These coordinate refers to the (x, y1)
        coordinate system.'''

        fig, ax1 = plt.subplots()
        for curve_idx in range(num_of_curves):
            ax1.plot(getattr(self, get_x)(curve_idx),
                     getattr(self, get_y1)(curve_idx),
                     color='blue')

        # Set the title
        title = ('{2} ($V_g$ in $[{0:.0f}\\,\\mathrm{{mV}}, {1:.0f}\\,\\mathrm{{mV}}]$)'
                 .format(vg_range_mV[0], vg_range_mV[1], hemt_name))
        if point_x and point_y:
            log.debug('Marking the point {0:.1f}, {1:.1f}'
                      .format(point_x, point_y))
            ax1.plot(point_x, point_y, 'o', color='black')

            title += ', tuning point: {0:.1f}, {1:.1f}'.format(
                point_x, point_y)

        ax1.set_title(title)

        ax1.set_xlabel(x_label)
        ax1.set_ylabel(y1_label)

        if get_y2:
            ax2 = ax1.twinx()
            for curve_idx in range(self.num_of_curves):
                ax2.plot(getattr(self, get_x)(curve_idx),
                         getattr(self, get_y2)(curve_idx),
                         color='red')
            ax2.set_ylabel(y2_label)

        # Save the plot
        plt.savefig(plot_file_name, bbox_inches='tight')
        plt.close()

        log.info('New plot saved in "%s"', plot_file_name)


class IdVdCurves(HemtCurves):
    '''Class containing the data read from a HEMT Id/Vd test file.

    This is one of the most important classes in the script. All the tuning
    calculations are done on the data exposed by this class.
    '''

    def __init__(self, metadata: Dict[str, Any], table: Dict[str, Any]):
        super(IdVdCurves, self).__init__(metadata, table, 'Gate')

        # Compute the transconductance, in mA/V
        trans = []
        vg = []
        for curve_idx in range(self.num_of_curves - 1):
            id0_mA, id1_mA = [self.get_id_mA(curve_idx + i) for i in (0, 1)]
            ig0_muA, ig1_muA = [self.get_ig_muA(curve_idx + i) for i in (0, 1)]
            vg0_mV, vg1_mV = [self.get_vg_mV(curve_idx + i) for i in (0, 1)]

            resistance = 10
            delta_vg_mV = (vg1_mV - vg0_mV) - resistance * (ig1_muA - ig0_muA)

            vg.append(0.5 * (vg0_mV + vg1_mV))
            trans.append(1e3 * (id1_mA - id0_mA) / delta_vg_mV)

        self._transconductance_fn = interp2d(x=self.get_vd_mV(0),
                                             y=vg,
                                             z=trans,
                                             bounds_error=False,
                                             fill_value=None)

    def get_id_mA(self, curve_num: int):
        'Return the array of Idrain values associated with the given curve'
        return self.get_array(curve_num, 'DrainI', mul_factor=1e3)

    def get_ig_muA(self, curve_num: int):
        'Return the array of Igate values associated with the given curve'
        return self.get_array(curve_num, 'GateI', mul_factor=1e6)

    def get_vd_mV(self, curve_num: int):
        'Return the arrays of Vdrain values associated with the given curve'
        return self.get_array(curve_num, 'DrainV', mul_factor=1e3)

    def get_vg_mV(self, curve_num: int):
        'Return the Vgate value associated with the given curve'
        return self.get_value(curve_num, 'GateV', mul_factor=1e3)

    def get_vg_range(self):
        'Return a tuple containing the minimum and maximum value for Vg'
        vg = np.array([self.get_vg_mV(i)
                       for i in range(self.num_of_curves)])
        return (np.min(vg), np.max(vg))

    def transconductance(self, vd: float, vg: float):
        '''Return the transconductance for the given voltages Vdrain and Vgate.

        The function uses a linear interpolation formula.'''

        result = self._transconductance_fn(vd, vg)
        if len(result) == 1:
            return result[0]
        else:
            return result

    def transconductance_curve(self, curve_idx: int):
        vg = self.get_vg_mV(curve_idx)
        return np.array([self._transconductance_fn(vd, vg)
                         for vd in self.get_vd_mV(curve_idx)])

    def create_plot(self, plot_file_name: str, hemt_name: str, tuning_point: TuningPoint):
        if tuning_point:
            point_x = tuning_point.vd
            point_y = tuning_point.id
        else:
            point_x, point_y = None, None

        super(IdVdCurves, self).create_plot(plot_file_name=plot_file_name,
                                            hemt_name=hemt_name,
                                            num_of_curves=self.num_of_curves,
                                            vg_range_mV=self.get_vg_range(),
                                            get_x='get_vd_mV',
                                            get_y1='get_id_mA',
                                            get_y2='get_ig_muA',
                                            x_label='$V_g$ [mV]',
                                            y1_label='$I_d$ [mA]',
                                            y2_label='$I_g$ [$\mu$A]',
                                            point_x=point_x,
                                            point_y=point_y)

    def create_transconductance_plot(self, plot_file_name: str,
                                     hemt_name: str,
                                     tuning_point: TuningPoint):
        if tuning_point:
            point_x = tuning_point.vd
            point_y = tuning_point.transconductance
        else:
            point_x, point_y = None, None

        super(IdVdCurves, self).create_plot(plot_file_name=plot_file_name,
                                            hemt_name=hemt_name,
                                            num_of_curves=self.num_of_curves - 1,
                                            vg_range_mV=self.get_vg_range(),
                                            get_x='get_vd_mV',
                                            get_y1='transconductance_curve',
                                            get_y2=None,
                                            x_label='$V_g$ [mV]',
                                            y1_label='Transconductance [mA/V]',
                                            y2_label=None,
                                            point_x=point_x,
                                            point_y=point_y)


class IdVgCurves(HemtCurves):
    '''Class containing the data read from a HEMT Id/Vd test file.

    This class is used only to produce a few plots to include in the report. It
    is not used for the tuning itself, as this is the duty of
    :class:`IdVdCurves`.
    '''

    def __init__(self, metadata: Dict[str, Any], table: Dict[str, Any]):
        super(IdVgCurves, self).__init__(metadata, table, 'Drain')

    def get_id_mA(self, curve_num: int):
        'Return the array of Idrain values associated with the given curve'
        return self.get_array(curve_num, 'DrainI', mul_factor=1e3)

    def get_ig_muA(self, curve_num: int):
        'Return the array of Igate values associated with the given curve'
        return self.get_array(curve_num, 'GateI', mul_factor=1e6)

    def get_vd_mV(self, curve_num: int):
        'Return the value of Vdrain associated with the given curve'
        return self.get_value(curve_num, 'DrainV', mul_factor=1e3)

    def get_vg_mV(self, curve_num: int):
        'Return the array of Vgate associated with the given curve'
        return self.get_array(curve_num, 'GateV', mul_factor=1e3)

    def get_vg_range(self):
        'Return a tuple containing the minimum and maximum value for Vg'
        vg = np.array([self.get_vg_mV(i)
                       for i in range(self.num_of_curves)])
        return (np.min(vg), np.max(vg))

    def create_plot(self, plot_file_name: str, hemt_name: str,
                    tuning_point: TuningPoint):
        if tuning_point:
            point_x = tuning_point.vg
            point_y = tuning_point.id
        else:
            point_x, point_y = None, None

        super(IdVgCurves, self).create_plot(plot_file_name=plot_file_name,
                                            hemt_name=hemt_name,
                                            num_of_curves=self.num_of_curves,
                                            vg_range_mV=self.get_vg_range(),
                                            get_x='get_vg_mV',
                                            get_y1='get_id_mA',
                                            get_y2='get_ig_muA',
                                            x_label='$V_g$ [mV]',
                                            y1_label='$I_d$ [mA]',
                                            y2_label='$I_g$ [$\mu$A]')


# The ordering of the amplifiers in the two legs is the following:
#
# +---------+---------+
# |         |         |
# |  H0/Q1  |  H1/Q6  |
# |         |         |
# +---------+---------+
# |         |         |
# |  H2/Q2  |  H3/Q5  |
# |         |         |
# +---------+---------+
# |         |         |
# |  H4/Q3  |  H5/Q4  |
# |         |         |
# +---------+---------+
H_INDEX_FROM_Q = {
    1: 0,
    2: 2,
    3: 4,
    4: 5,
    5: 3,
    6: 1
}


class HemtProperties:
    '''Class that collects all information about a HEMT (including Keithley data)

    When instantiating this class, the relevant Excel file will be read and
    interpreted.
    '''

    def __init__(self, q_index: int, input_path: str, output_path: str):
        self.q_index = q_index
        self.q_name = 'Q{0}'.format(q_index)  # E.g., "q0"
        self.h_name = 'H{0}'.format(H_INDEX_FROM_Q[q_index])  # E.g., "h1"
        self.full_name = '{0}/{1}'.format(self.q_name, self.h_name)

        self._read_id_vd(os.path.join(
            input_path,
            'Id_vs_Vd_{0}#1@1.xls'.format(self.h_name)
        ))
        self._read_id_vg(os.path.join(
            input_path,
            'Id_vs_Vg_{0}#1@1.xls'.format(self.h_name)
        ))

        self.id_vd_plot_file = os.path.join(
            output_path,
            'id_vs_vd_{0}.svg'.format(self.q_name.lower())
        )
        self.id_vg_plot_file = os.path.join(
            output_path,
            'id_vs_vg_{0}.svg'.format(self.q_name.lower())
        )
        self.trans_vd_plot_file = os.path.join(
            output_path,
            'trans_hemt_vs_vd_{0}.svg'.format(self.q_name.lower())
        )

        self.tuning_point = None  # Type: TuningPoint

    def _read_id_vd(self, file_name: str):
        log.info('Reading file "%s"', file_name)
        with xlrd.open_workbook(file_name) as workbook:
            self.id_vd = IdVdCurves(
                metadata=excel.read_worksheet_settings(workbook),
                table=excel.read_worksheet_table(workbook)
            )

    def _read_id_vg(self, file_name: str):
        log.info('Reading file "%s"', file_name)
        with xlrd.open_workbook(file_name) as workbook:
            self.id_vg = IdVgCurves(
                metadata=excel.read_worksheet_settings(workbook),
                table=excel.read_worksheet_table(workbook)
            )

    def create_plots(self):
        'Generate all the plots related to this HEMT'
        self.id_vd.create_plot(plot_file_name=self.id_vd_plot_file,
                               hemt_name=self.full_name,
                               tuning_point=self.tuning_point)
        self.id_vd.create_transconductance_plot(plot_file_name=self.trans_vd_plot_file,
                                                hemt_name=self.full_name,
                                                tuning_point=self.tuning_point)
        self.id_vg.create_plot(plot_file_name=self.id_vg_plot_file,
                               hemt_name=self.full_name,
                               tuning_point=self.tuning_point)


def find_matching_vd_id(id_vd: IdVdCurves, ref_vd_mV: float, ref_id_mA: float):
    '''Look for a datapoint in a Id/Vd curve which matches some reference point.

    Return a pair containing the index of the curve and the index of the point
    in the curve.
    '''

    vd = id_vd.get_vd_mV(0)
    # Find the point that matches the value for Vd
    # (index usable with any of the vd/id arrays)
    datapoint_idx = np.argmin(np.abs(vd - ref_vd_mV))

    # Retrieve all the Id values from each curve that
    # match the value in Vd (use datapoint_idx for this)
    id_values = np.array([id_vd.get_id_mA(i)[datapoint_idx]
                          for i in range(id_vd.num_of_curves)])

    # Look for the datapoint which matches Id as well
    curve_idx = np.argmin(np.abs(id_values - ref_id_mA))

    return curve_idx, datapoint_idx


def tune(hemt_dict: Dict[str, HemtProperties],
         ref_vd_mV=900.0,
         id_mA_q1_q6=4.5,
         id_mA_q2_q5=7.5,
         id_mA_q3_q4=6.0,
         id_mA_tolerance=0.5):
    '''Tune the amplifiers in ``hemt_dict``.

    The result of the tuning is saved in ``hemt_dict`` itself.'''

    assert len(hemt_dict) == 6

    curve_idx = {}  # Type: Dict[str, int]
    point_idx = {}  # Type: Dict[str, int]

    # Tune the first two amplifiers in each leg
    for hemt_name, ref_value in [('q1', id_mA_q1_q6),
                                 ('q6', id_mA_q1_q6),
                                 ('q2', id_mA_q2_q5),
                                 ('q5', id_mA_q2_q5)]:

        id_vd = hemt_dict[hemt_name].id_vd
        curve, point = \
            find_matching_vd_id(id_vd,
                                ref_vd_mV=ref_vd_mV,
                                ref_id_mA=ref_value)

        vg_mV = id_vd.get_vg_mV(curve)
        vd_mV = id_vd.get_vd_mV(curve)[point]
        hemt_dict[hemt_name].tuning_point = \
            TuningPoint(vg=vg_mV,
                        ig=id_vd.get_ig_muA(curve)[point],
                        vd=vd_mV,
                        id=id_vd.get_id_mA(curve)[point],
                        transconductance=id_vd.transconductance(vd=vd_mV, vg=vg_mV))

        log.debug('Tuning point for {0}: {1}'
                  .format(hemt_name, hemt_dict[hemt_name].tuning_point))
        curve_idx[hemt_name], point_idx[hemt_name] = curve, point

    # To calibrate the last amplifier, we need to balance the total
    # transconductance of the two legs

    leg1_partial_tr = np.prod([hemt_dict[x].tuning_point.transconductance
                               for x in ('q1', 'q2')])
    leg2_partial_tr = np.prod([hemt_dict[x].tuning_point.transconductance
                               for x in ('q6', 'q5')])

    # For each of the two amplifiers Q3 and Q4, build a list that
    # associates Idrain with the transconductance, provided that
    # iD is not too far from "id_mA_q3_q4"
    setpoints = {}  # Type: Dict[str, List[TuningPoint]]
    for hemt_name in ('q3', 'q4'):
        id_vd = hemt_dict[hemt_name].id_vd
        vd_mV = id_vd.get_vd_mV(0)
        datapoint_idx = np.argmin(np.abs(vd_mV - ref_vd_mV))

        setpoints[hemt_name] = []
        for curve_idx in range(id_vd.num_of_curves):
            ig_muA = id_vd.get_ig_muA(curve_idx)[datapoint_idx]
            vg_mV = id_vd.get_vg_mV(curve_idx)
            id_mA = id_vd.get_id_mA(curve_idx)[datapoint_idx]
            transconductance = id_vd.transconductance(vd=vd_mV[datapoint_idx],
                                                      vg=vg_mV)

            if np.abs(id_mA - id_mA_q3_q4) < id_mA_tolerance:
                cur_setpoint = TuningPoint(vg=vg_mV,
                                           ig=ig_muA,
                                           vd=vd_mV[datapoint_idx],
                                           id=id_mA,
                                           transconductance=transconductance)
                setpoints[hemt_name].append(cur_setpoint)

    if len(setpoints.keys()) == 0:
        log.error('No configurations found')
        return

    balances = [(q3_point, q4_point, np.abs(leg1_partial_tr * q3_point.transconductance -
                                            leg2_partial_tr * q4_point.transconductance))
                for q3_point in setpoints['q3']
                for q4_point in setpoints['q4']]

    log.info('Range of balances: {0:.6e} - {1:.6e} ({2} configurations)'
             .format(np.min([x[2] for x in balances]),
                     np.max([x[2] for x in balances]),
                     len(balances)))

    hemt_dict['q3'].tuning_point, \
        hemt_dict['q4'].tuning_point, \
        best_balance = balances[np.argmin([x[2] for x in balances])]

    log.info('Best configuration for Q3: {0}'.format(
        hemt_dict['q3'].tuning_point))
    log.info('Best configuration for Q4: {0}'.format(
        hemt_dict['q4'].tuning_point))
    log.info('Best balance: {0}'.format(best_balance))


def create_plots(hemt_list: List[HemtProperties]):
    for cur_hemt in hemt_list:
        cur_hemt.create_plots()


def create_report(pol_name: str,
                  hemt_dict: Dict[str, HemtProperties],
                  output_path: str):
    '''Saves a report of the tuning in the output path.

    This function assumes that ``output_path`` points to a directory that already exists.
    '''

    template_path = os.path.join(os.path.dirname(__file__), 'template')

    # Copy all the static files into the destination directory
    for static_file_name in ['report_style.css']:
        copyfile(os.path.join(template_path, static_file_name),
                 os.path.join(output_path, static_file_name))

    # Load the file containing the Markdown template in a string
    template_file_name = os.path.join(template_path, 'report.md')
    log.info('Reading report template from "%s"', template_file_name)
    with open(template_file_name) as f:
        report_template = Template(''.join(f.readlines()))

    # Assemble all the parameters to substitute in the Markdown template into a
    # dictionary
    params = {
        'polarimeter': pol_name,
        'title': 'Tuning report for polarimeter {0}'.format(pol_name),
        'date': datetime.now().strftime('%d %b %Y, %H:%M:%S'),
        'leg1_transconductance': '{0:.2f}'.format(
            np.prod([hemt_dict[x].tuning_point.transconductance
                     for x in ('q1', 'q2', 'q3')])
        ),
        'leg2_transconductance': '{0:.2f}'.format(
            np.prod([hemt_dict[x].tuning_point.transconductance
                     for x in ('q6', 'q5', 'q4')])
        ),
    }

    for hemt_name in ('q1', 'q2', 'q3', 'q4', 'q5', 'q6'):
        tuning_point = hemt_dict[hemt_name].tuning_point

        params['{0}_id'.format(hemt_name)] = '{0:.2f}'.format(tuning_point.id)
        params['{0}_vd'.format(hemt_name)] = '{0:.1f}'.format(tuning_point.vd)
        params['{0}_ig'.format(hemt_name)] = '{0:.2f}'.format(tuning_point.ig)
        params['{0}_vg'.format(hemt_name)] = '{0:.1f}'.format(tuning_point.vg)
        params['{0}_transconductance'.format(hemt_name)] = '{0:.2f}'.format(
            tuning_point.transconductance)

    # Fill the template and save the report in Markdown format
    md_report = report_template.safe_substitute(params)
    md_report_path = os.path.join(output_path, 'index.md')
    with open(md_report_path, 'wt') as f:
        f.write(md_report)
    log.info('Markdown report saved to "%s"', md_report_path)

    # Convert the report to HTML and save it too
    html_report = '''<!DOCTYPE html>
<html>
    <head>
        <title>{title}</title>
        <meta charset="UTF-8">
        <link rel="stylesheet" href="report_style.css" type="text/css" />
    </head>
    <body>
        <div id="main">
{contents}
        </div>
    </body>
</html>
'''.format(title=params['title'],
           contents=markdown(md_report, extensions=[
               'markdown.extensions.attr_list',
               'markdown.extensions.tables',
               'markdown.extensions.toc']
    ))

    html_report_path = os.path.join(output_path, 'index.html')
    with open(html_report_path, 'wt') as f:
        f.write(html_report)
    log.info('HTML report saved to "%s"', html_report_path)


def parse_arguments():
    '''Return a class containing the values of the command-line arguments.

    The field accessible from the object returned by this function are the following:

    - ``polarimeter_name``
    - ``input_path``
    - ``output_path``
    '''
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('polarimeter_name', type=str,
                        help='''Name of the polarimeter (any text string
                                is ok, it is used in the reports)''')
    parser.add_argument('input_path', type=str,
                        help='Path to the Keithley input data')
    parser.add_argument('output_path', type=str,
                        help='''Path to the directory that will contain the
                        report. If the path does not exist, it will be created''')
    return parser.parse_args()


def main():
    'Entry point of the program'

    log.basicConfig(format='[%(asctime)s %(levelname)s] %(message)s',
                    level=log.DEBUG)
    args = parse_arguments()

    log.info('Tuning radiometer "%s"', args.polarimeter_name)
    log.info('Reading data from "%s"', args.input_path)
    log.info('Writing the report into "%s"', args.output_path)

    # Create the directory that will contain the report
    os.makedirs(args.output_path, exist_ok=True)

    # Create a dictionary associating names like "q1" with
    # a HemtProperties object
    hemt_dict = dict([('q{0}'.format(q),
                       HemtProperties(q_index=q,
                                      input_path=args.input_path,
                                      output_path=args.output_path))
                      for q in (1, 2, 3, 4, 5, 6)])

    tune(hemt_dict)
    create_plots(hemt_dict.values())
    create_report(pol_name=args.polarimeter_name,
                  hemt_dict=hemt_dict,
                  output_path=args.output_path)


if __name__ == '__main__':
    main()
