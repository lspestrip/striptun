#!/usr/bin/env python3
# -*- encoding: utf-8 -*-

from argparse import ArgumentParser
from collections import namedtuple
import logging as log
import os.path
from typing import Any, Dict, List, Tuple

from json_save import save_parameters_to_json
from reports import create_report, get_code_version_params
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pylab as plt
from matplotlib.patches import Rectangle
import matplotlib.transforms as transforms

from file_access import load_timestream

SAMPLING_FREQUENCY_HZ = 25.0
MAX_REGION_LENGTH_S = 30.0

SlopeInformation = namedtuple(
    "SlopeInformation", ["time0_s", "time1_s", "abs_slope_adu_s"]
)


def slope(time, values, chunk_len: int, step: int):
    result = []
    first_idx = 0
    while (first_idx + chunk_len) < len(values):
        cur_chunk_time = time[first_idx : (first_idx + chunk_len)]
        cur_chunk_values = values[first_idx : (first_idx + chunk_len)]

        poly_coeffs = np.polyfit(cur_chunk_time, cur_chunk_values, 1)

        result.append(
            SlopeInformation(
                time0_s=cur_chunk_time[0],
                time1_s=cur_chunk_time[-1],
                abs_slope_adu_s=np.abs(poly_coeffs[0]),
            )
        )

        first_idx += step

    return result


def find_blind_channel(slopes: List[List[SlopeInformation]]) -> Tuple[int, float]:

    max_slope = np.empty(4)
    for cur_curve in range(4):
        cur_slope_arr = np.array([x.abs_slope_adu_s for x in slopes[cur_curve]])
        max_slope[cur_curve] = np.max(cur_slope_arr)

    # We do not want a NumPy type here (it makes JSON serialization difficult)
    blind_idx = int(np.argmin(max_slope))
    return blind_idx, max_slope[blind_idx]


def contiguous_regions(condition):
    """Finds contiguous True regions of the boolean array "condition". Returns
    a 2D array where the first column is the start index of the region and the
    second column is the end index.

    Taken from
    https://stackoverflow.com/questions/4494404/find-large-number-of-consecutive-values-fulfilling-condition-in-a-numpy-array#4495197
    """

    # Find the indicies of changes in "condition"
    d = np.diff(condition)
    (idx,) = d.nonzero()

    # We need to start things after the change in "condition". Therefore,
    # we'll shift the index by 1 to the right.
    idx += 1

    if condition[0]:
        # If the start of condition is True prepend a 0
        idx = np.r_[0, idx]

    if condition[-1]:
        # If the end of condition is True, append the length of the array
        idx = np.r_[idx, condition.size]  # Edit

    # Reshape the result into two columns
    idx.shape = (-1, 2)
    return idx


Region = namedtuple("Region", ["time0_s", "time1_s"])


def find_stable_regions(
    slopes: List[SlopeInformation],
    slope_threshold_adu_s: float,
    duration_threshold_s: float,
    clipping_s: float = 5.0,
    first_region_length: float = None,
    regions: List[Tuple[float, float]] = None,
) -> List[Region]:

    if regions:
        return [Region(time0_s=x[0], time1_s=x[1]) for x in regions]

    mask = np.array([x.abs_slope_adu_s for x in slopes]) < slope_threshold_adu_s

    regions = contiguous_regions(mask)

    if first_region_length:
        assert first_region_length > 0.0
        result = [Region(time0_s=0.0, time1_s=first_region_length)]
    else:
        result = []  # Type: List[Region]

    for start_idx, stop_idx in list(regions):
        assert np.alltrue(mask[start_idx:stop_idx])
        if stop_idx < len(mask):
            assert not mask[stop_idx]

        time0_s = slopes[start_idx].time0_s + clipping_s
        time1_s = slopes[stop_idx - 1].time1_s - clipping_s
        if time1_s - time0_s < duration_threshold_s:
            continue

        if result:
            # Check that this region does not overlap with the last one
            last_region = result[-1]
            if last_region.time1_s > time0_s:
                # Yes, it does: remove the last one and append a new
                # region that covers both the old and the current one
                result.pop()
                result.append(
                    Region(
                        time0_s=max(last_region.time0_s, time1_s - MAX_REGION_LENGTH_S),
                        time1_s=time1_s,
                    )
                )
                continue

        time0_s = max(time0_s, time1_s - MAX_REGION_LENGTH_S)
        result.append(Region(time0_s=time0_s, time1_s=time1_s))

    log.info("%d regions found:", len(result))
    for region in result:
        log.info(
            "  %.1f s − %.1f s, duration: %.1f s",
            region.time0_s,
            region.time1_s,
            region.time1_s - region.time0_s,
        )

    return result


def save_plot(
    time,
    smoothed,
    slopes,
    regions: List[Region],
    slope_threshold_adu_s: float,
    output_file_name: str,
):
    fig, axes = plt.subplots(ncols=1, nrows=2, sharex=False, figsize=(5, 8))

    axes[0].plot(time, smoothed)
    axes[1].plot(
        [(x.time0_s + x.time1_s) * 0.5 for x in slopes],
        [x.abs_slope_adu_s for x in slopes],
    )

    for cur_region in regions:
        for ax_idx in (0, 1):
            trans = transforms.blended_transform_factory(
                axes[ax_idx].transData, axes[ax_idx].transAxes
            )
            axes[ax_idx].add_patch(
                Rectangle(
                    (cur_region.time0_s, 0),
                    width=(cur_region.time1_s - cur_region.time0_s),
                    height=1,
                    transform=trans,
                    alpha=0.5,
                    facecolor="#aaaaaa",
                    edgecolor="#aaaaaa",
                )
            )

        trans = transforms.blended_transform_factory(
            axes[ax_idx].transAxes, axes[ax_idx].transData
        )
        axes[ax_idx].add_patch(
            Rectangle(
                (0.0, 0.0),
                width=1.0,
                height=slope_threshold_adu_s,
                transform=trans,
                alpha=0.5,
                facecolor="#cccccc",
                edgecolor="#aaaaaa",
            )
        )
    axes[0].set_ylabel("Output [ADU]")
    axes[1].set_ylabel("Slope [ADU/s]")
    axes[1].set_xlabel("Time [s]")

    log.info('Saving plot into file "{0}"'.format(output_file_name))
    plt.savefig(output_file_name, bbox_inches="tight")


def parse_arguments():
    """Return a class containing the values of the command-line arguments.

    The field accessible from the object returned by this function are the following:

    - ``polarimeter_name``
    - ``input_url``
    - ``output_path``
    """
    parser = ArgumentParser(description=__doc__)
    parser.add_argument(
        "polarimeter_name",
        type=str,
        help="""Name of the polarimeter (any text string
                                is ok, it is used in the reports)""",
    )
    parser.add_argument(
        "input_url", type=str, help="""File/URL containing the data being saved"""
    )
    parser.add_argument(
        "output_path",
        type=str,
        help="""Path to the directory that will contain the
                        report. If the path does not exist, it will be created""",
    )
    parser.add_argument(
        "--first-region-length",
        type=float,
        default=None,
        help="""Specify that the first region to be analyzed should
                        cover the first N seconds of data in the test. This is
                        useful when the first step lasted for a very short period of
                        time before the user changed the temperature of the load, and
                        the code is not able to detect that the instrument was stable at
                        the beginning.""",
    )
    parser.add_argument(
        "--regions",
        type=str,
        default=None,
        help="""Specify the time intervals where the signal was stable
                        enough to be used for the characterization of the noise temperature.
                        Normally the code is able to detect such intervals, but there are
                        a few cases where unwanted features in the signal (e.g., spikes,
                        high white noise...) might make the code fail. Intervals are to be
                        specified as a comma-separated list, where each element contains
                        the start and end times in seconds, separated by an hyphen. For
                        example, "10-20,60-120" specifies two intervals: the first one starts
                        at time 10 s and ends at time 20 s, the second one starts at
                        60 s and ends at 120 s.""",
    )
    return parser.parse_args()


# This structure is a superset of "Region", which holds information
# useful to produce the report
RegionInformation = namedtuple(
    "RegionInformation",
    [
        "time0_s",
        "time1_s",
        "index0",
        "index1",
        "mean_output_adu",
        "rms_output_adu",
    ],
)


def assemble_region_info(time, value, regions: List[Region]) -> List[RegionInformation]:
    result = []

    for cur_r in regions:
        # We do not want NumPy types here (they make JSON serialization difficult)
        index0 = int(np.argmin(np.abs(time - cur_r.time0_s)))
        index1 = int(np.argmin(np.abs(time - cur_r.time1_s)))

        result.append(
            RegionInformation(
                time0_s=cur_r.time0_s,
                time1_s=cur_r.time1_s,
                index0=index0,
                index1=index1,
                mean_output_adu=float(np.mean(value[index0 : index1 + 1])),
                rms_output_adu=float(np.std(value[index0 : index1 + 1])),
            )
        )

    return result


def build_dict_from_results(
    pol_name: str, blind_channel: int, time, data, regions: Dict[int, List[Region]]
) -> Dict[str, Any]:
    return {
        "polarimeter": pol_name,
        "title": "Noise temperature analysis for polarimeter {0}".format(pol_name),
        "blind_channel": blind_channel,
        "sampling_frequency": SAMPLING_FREQUENCY_HZ,
        "regions": dict(
            [
                (idx, assemble_region_info(time, data[:, idx], regions[idx]))
                for idx, region in regions.items()
            ]
        ),
    }


def parse_region_list_str(s: str) -> List[Tuple[float, float]]:
    """Parse a string passed to the --regions command line parameter.

    Example:
    > parse_region_list_str('10-20,60-120')
    [(10.0, 20.0), (60.0, 120.0)]
    """

    if not s:
        return []

    result = []
    for interval in s.split(","):
        t0, t1 = interval.split("-")
        result.append((float(t0), float(t1)))

    return result


def main():

    log.basicConfig(format="[%(asctime)s %(levelname)s] %(message)s", level=log.DEBUG)
    args = parse_arguments()

    log.info('Tuning radiometer "%s"', args.polarimeter_name)
    log.info('Reading data from file "%s"', args.input_url)
    log.info('Writing the report into "%s"', args.output_path)

    # Create the directory that will contain the report
    os.makedirs(args.output_path, exist_ok=True)

    # Load from the text file only the columns containing the output of the four detectors
    log.info('Loading data from "{0}"'.format(args.input_url))
    timestream = load_timestream(args.input_url)[1]
    power = timestream.power

    log.info("File loaded, {0} samples found".format(len(power[:, 0])))

    time = np.arange(len(power[:, 0])) / SAMPLING_FREQUENCY_HZ
    slopes = [
        slope(time, power[:, i], chunk_len=25 * 60, step=25 * 3) for i in range(4)
    ]

    # Find the blind channel
    blind_channel, slope_threshold = find_blind_channel(slopes)
    log.info("The blind channel is PWR{0}".format(blind_channel))
    log.info(
        "The maximum threshold on the slope is {0:.1f} ADU/s".format(slope_threshold)
    )

    # Look for the places where the signal seems to be stable enough
    regions = {}
    num_of_regions = None
    for curve_idx in range(4):
        if curve_idx == blind_channel:
            continue

        regions[curve_idx] = find_stable_regions(
            slopes=slopes[curve_idx],
            slope_threshold_adu_s=1.5 * slope_threshold,
            duration_threshold_s=60.0,
            clipping_s=15.0,
            first_region_length=args.first_region_length,
            regions=parse_region_list_str(args.regions),
        )
        if not num_of_regions:
            num_of_regions = len(regions[curve_idx])
        else:
            if num_of_regions != len(regions[curve_idx]):
                log.warning(
                    "Mismatch in the number of quiet regions across the detectors: %d against %d",
                    num_of_regions,
                    len(regions[curve_idx]),
                )

    # Produce the plots
    for curve_idx in range(4):
        output_file_name = os.path.join(
            args.output_path, "plot_pwr{0}.svg".format(curve_idx)
        )
        if curve_idx == blind_channel:
            curve_regions = []
        else:
            curve_regions = regions[curve_idx]

        save_plot(
            time,
            power[:, curve_idx],
            slopes[curve_idx],
            curve_regions,
            slope_threshold,
            output_file_name,
        )

    params = build_dict_from_results(
        pol_name=args.polarimeter_name,
        blind_channel=blind_channel,
        time=time,
        data=power,
        regions=regions,
    )
    params["data_url"] = args.input_url

    region_str = []
    for curve_idx in range(4):
        if curve_idx in regions and regions[curve_idx]:
            region_str.append(
                ",".join(
                    [
                        "{0:.0f}-{1:.0f}".format(x.time0_s, x.time1_s)
                        for x in regions[curve_idx]
                    ]
                )
            )
        else:
            region_str.append("")
    params["region_str"] = region_str

    save_parameters_to_json(
        params=dict(params, **get_code_version_params()),
        output_file_name=os.path.join(args.output_path, "tnoise_step1_results.json"),
    )

    create_report(
        params=params,
        md_template_file="tnoise_step1.md",
        md_report_file="tnoise_step1_report.md",
        html_report_file="tnoise_step1_report.html",
        output_path=args.output_path,
    )


if __name__ == "__main__":
    main()
