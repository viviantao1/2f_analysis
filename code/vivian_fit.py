import numpy as np
from datetime import datetime
import os

from Dpipe import repkg, pipe, timeseries_util as tsu, startup
from Dpipe import mss_util as util
from Dpipe import cuts
from Dpipe import tau_fit
from Dpipe import filtering

# Adapted from Dpipe.mss_util so that the frequency to fit can be modified. The original function only fits 2f amplitudes.
def filter_span_data(span, freq_mult=2, verbose=False):
    """Apply bandpass filter to the entire span's detector data.

    Parameters
    ----------
    span : Span
        The Dpipe Span object with loaded TOD data
    freq_mult : float
        The frequency multiplier for the modulation frequency
    verbose : bool
        Whether to print progress information

    Returns
    -------
    dict
        Dictionary containing filtered data and associated metadata
    """

    if not hasattr(span, 'tod') or span.tod is None:
        if verbose:
            print("No TOD data found in span")
        return None

    det_data = span.tod.data
    ctime = span.tod.ctime

    encoder_data = span.tod.mod_enc

    if hasattr(span.tod, 'dm') and span.tod.dm.has_hwp():
        modulator_type = 'hwp'
    elif hasattr(span.tod, 'dm') and span.tod.dm.has_vpm():
        modulator_type = 'vpm'
    else:
        if verbose:
            print("No modulator found for this span")
        return None

    mod_freq = span.mod_freq if hasattr(span, 'mod_freq') else None

    if mod_freq is None:
        raise ValueError("Modulation frequency not found in span")

    sample_rate = 1.0 / np.median(np.diff(ctime))
    n_samps = len(ctime)

    if modulator_type.lower() == 'hwp':
        mod_type = 'hwp'
        # For HWP: Bandpass filter around 2f
        freq = freq_mult * mod_freq  # fit any multiple of mod_freq
        fc = (max(0.05, freq * 0.8), freq * 1.2)  # (low, high) cutoffs
    else:  # VPM
        mod_type = 'vpm'
        # For VPM: Bandpass filter around modulation frequency
        fc = (max(0.05, mod_freq * 0.8), mod_freq * 1.2)  # (low, high) cutoffs

    tw = (0.2, 0.2)  # Transition width for both edges
    ntimes = 3

    if mod_type == 'vpm':
        if span.taus is None:
            span.get_taus()
        taus = span.taus
        readout = True
    else:
        taus = None
        readout = False

    filt = filtering.WindowBlackFilter(nsamp=n_samps, samp_freq=float(sample_rate), fc=fc, tw=tw,
                                       ntimes=ntimes, stride=1)

    tod_filter = filtering.get_tod_filter(span, filt=filt, readout=readout, taus=taus)
    tod_filter(span.tod.data, do_stride=False, do_detrend=True, do_retrend=False)

    # Use span.cuts consistently (not span.tod.cuts)
    if hasattr(span, 'cuts') and span.cuts is not None:
        uncut_dets = span.cuts.get_detectors_with_uncut_samples()
    else:
        uncut_dets = np.arange(det_data.shape[0])

    det_uids = [span.tod.det_uid[i] for i in uncut_dets]

    cut_masks = []

    for i, det_idx in enumerate(uncut_dets):
        if hasattr(span, 'cuts') and span.cuts is not None:
            mask = span.cuts.cuts[det_idx].get_mask()
        else:
            mask = np.zeros(n_samps, dtype=bool)
        cut_masks.append(mask)

    return {'filtered_data': span.tod.data, 'uncut_dets': uncut_dets, 'det_uids': det_uids,
            'encoder_data': encoder_data, 'ctime': ctime, 'modulator_type': modulator_type,
            'mod_freq': mod_freq, 'sample_rate': sample_rate,
            'filter_params': {'fc': fc, 'tw': tw, 'ntimes': ntimes}, 'cut_masks': cut_masks}

def extract_templates_from_span(span, filtered_span_data,freq_mult=2, binning=True, bin_med=False, n=1, verbose=False):
    """Extract fit templates from a Dpipe Span object, optionally applying phase folding.

    Parameters
    ----------
    span : Span
        The Dpipe Span object with loaded TOD data
    freq_mult : float
        The frequency multiplier for the modulation frequency
    binning : bool
        Whether to apply phase folding
    bin_med : bool
        Use median for binning if True, otherwise use mean
    n : int
        The number of segments to further split a datapkg into (default is 1, meaning each segment is one datapkg)
    verbose : bool
        Whether to print progress information

    Returns
    -------
    dict
        Dictionary containing timestamp and detector fit results for each segment in the span
    """

    results = {}

    if not hasattr(span, 'tod') or span.tod is None:
        if verbose:
            print("No TOD data found in span")
        return results

    # if verbose:
    #     print("Filtering span data...")
    # filtered_span_data = filter_span_data(span, freq_mult=freq_mult, verbose=verbose)
    # if filtered_span_data is None:
    #     if verbose:
    #         print("Failed to filter span data")
    #     return {}

    filtered_data = filtered_span_data['filtered_data']
    uncut_dets = filtered_span_data['uncut_dets']
    det_uids = filtered_span_data['det_uids']
    encoder_data = filtered_span_data['encoder_data']
    modulator_type = filtered_span_data['modulator_type']
    mod_freq = filtered_span_data['mod_freq']
    full_cut_masks = filtered_span_data['cut_masks']

    # split into segments by datapkg
    slices = span.get_segments(unit='dpkg')
    # slices = span.get_segments(unit='sec', n = 60)
    if n != 1:
        small_slices = []
        for dpkg in slices:
            start = dpkg.start
            stop = dpkg.stop

            edges = np.linspace(start, stop, n + 1).astype(int)

            for i in range(n):
                small_slices.append(slice(edges[i], edges[i+1]))

        slices = small_slices

    k = 0
    if verbose:
        print(f"Extracting templates from {len(slices)} segments...")
    for seg in slices:
        if binning:
            seg_results = extract_templates_phase_folding(seg, filtered_data, uncut_dets, det_uids, encoder_data, k, bin_med=bin_med)
        else:
            seg_results = extract_templates_raw_fit(seg, filtered_data, uncut_dets, det_uids, encoder_data, k)
        if seg_results:
            pkg_timestamp = span.tod.ctime[seg.start]
            results[str(pkg_timestamp)] = seg_results
        k+=1

    return results

def extract_templates_phase_folding(seg, filtered_data, uncut_dets, det_uids, encoder_data, k, bin_med=False):
    """Extract fit templates from a segment of a span with phase folding, either using bin mean or median.

    Parameters
    ----------
    seg : slice
        The slice object representing the segment of the span to process
    filtered_data : np.ndarray
        The filtered detector data for the entire span
    uncut_dets : np.ndarray
        The TOD indices of detectors with at least one uncut sample
    det_uids : np.ndarray
        The detector UID corresponding to the uncut detectors and TOD indexing.
    encoder_data : np.ndarray
        The encoder data for the entire span
    k : int
        The current segment ID
    bin_med : bool
        Use median for binning if True, otherwise use mean

    Returns
    -------
    dict
        Dictionary containing detector fit results for the segment, by detector UID
    """

    seg_results = {}
    theta = np.deg2rad(encoder_data[seg])

    # divide the phase into 360 bins - each 0.5 degrees wide
    # bin_count = min(360, int(180.0 / max(0.5, np.median(np.diff(np.sort(np.unique(encoder_data)))))))
    bin_count = 360
    phase = encoder_data[seg] % 180
    bins = np.linspace(0, 180, bin_count + 1) # bin edges
    bin_idx = np.digitize(phase, bins) - 1 # index of which bin each piece of data goes into 

    bin_centers = (bins[1:] + bins[:-1]) / 2 # find middle of bin edges
            
    for i in range(len(uncut_dets)): #uncut_dets stores tod indices of detectors with at least 1 uncut sample
        signal = filtered_data[uncut_dets[i],seg]

        # phase fold signal into each of the bins by calcuating mean or median of the signal in each bin
        template = np.zeros(bin_count) # mean signal at each bin
        template_errors = np.zeros(bin_count)
        bin_size = np.zeros(bin_count, dtype=int)

        for j in range(bin_count):
            samples = signal[bin_idx == j]

            bin_size[j] = len(samples)

            if len(samples) > 0:
                if bin_med:
                    template[j] = np.median(samples)
                else:
                    template[j] = np.mean(samples)

                if len(samples) > 1:
                    template_errors[j] = np.std(samples, ddof=1) / np.sqrt(len(samples))

        # Check if we have enough binned data
        valid_bins = bin_size > 0
        if np.sum(valid_bins) < 0.5 * bin_count:
            template_result = {'amplitude': None, 'scaling_factor': None, 'quality': None, 'A':None, 'B': None, 'C':None, 'segment_id': k}
        else:
            # fit the binned data to a 2f model using least squares
            theta_bin = np.deg2rad(bin_centers[valid_bins])
            matrix = np.column_stack([
                np.cos(2 * theta_bin),
                np.sin(2 * theta_bin),
                np.ones_like(theta_bin)
            ])

            coef, *_ = np.linalg.lstsq(matrix, template[valid_bins], rcond=None)

            template_result = get_fit_info(coef, theta, signal, k)

            det_uid = det_uids[i] # det uid in order of indexing of tod
            seg_results[det_uid] = template_result

    return seg_results

def extract_templates_raw_fit(seg, filtered_data, uncut_dets, det_uids, encoder_data, k):
    """Extract fit templates from a segment of a span with unbinned, raw fitting.

    Parameters
    ----------
    seg : slice
        The slice object representing the segment of the span to process
    filtered_data : np.ndarray
        The filtered detector data for the entire span
    uncut_dets : np.ndarray
        The TOD indices of detectors with at least one uncut sample
    det_uids : np.ndarray
        The detector UID corresponding to the uncut detectors and TOD indexing.
    encoder_data : np.ndarray
        The encoder data for the entire span
    k : int
        The current segment ID

    Returns
    -------
    dict
        Dictionary containing detector fit results for the segment, by detector UID
    """

    seg_results = {}
    theta = np.deg2rad(encoder_data[seg])

    for i in range(len(uncut_dets)): #uncut_dets stores tod indices of detectors with at least 1 uncut sample
        signal = filtered_data[uncut_dets[i],seg]

        # fit the filtered data to a 2f model using least squares
        matrix = np.column_stack([
            np.cos(2 * theta),
            np.sin(2 * theta),
            np.ones_like(theta)
        ])

        coef, *_ = np.linalg.lstsq(matrix, signal, rcond=None)

        template_result = get_fit_info(coef, theta, signal, k)

        det_uid = det_uids[i] # det uid in order of indexing of tod
        seg_results[det_uid] = template_result
    return seg_results

def get_fit_info(coef, theta, signal, k):
    """Get amplitude and r-squared quality of fit for a given detector's signal and least squaresfitted coefficients.

    Parameters
    ----------
    coef : np.ndarray
        The least squares fitted coefficients A,B,C for the model signal = A*cos(2*theta) + B*sin(2*theta) + C
    theta : np.ndarray
        The phase data in radians
    signal : np.ndarray
        The detector signal data
    k : int
        The current segment ID

    Returns
    -------
    dict
        Dictionary containing fit amplitude and r-squared results for one detector and segment
    """

    A, B, C = coef

    matrix = np.column_stack([
        np.cos(2 * theta),
        np.sin(2 * theta),
        np.ones_like(theta)
    ])
    
    fit_2f = matrix @ np.array([A, B, C])
    ss_res = np.sum((signal - fit_2f)**2)
    ss_tot = np.sum((signal - np.mean(signal))**2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0

    amplitude = np.sqrt(A**2 + B**2)

    template_result = {'amplitude': amplitude, 'scaling_factor': 1.0, 'quality': r2, 'A':A, 'B': B, 'C':C, 'segment_id': k}

    return template_result