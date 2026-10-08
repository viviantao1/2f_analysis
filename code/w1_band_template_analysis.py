"""
Helper functions for W1-band HWP template database analysis.

MODIFIED FROM THE Q-BAND VPM TEMPLATE ANALYSIS MODULE (Dpipe/examples/q_band_template_analysis.py)

This module provides utilities for querying and visualizing VPM modulation templates
stored in the Q-band template database. The database contains template information
extracted from processed spans, including amplitudes, phases, quality metrics, and
environmental conditions.

Database Schema:
----------------
- config_info: Configuration details and hashing for reproducibility
- spans: Top-level span information (path, boresight, elevation)
- segments: Time segments within spans (timestamp, mod_freq, modulator_type, site_pwv)
- templates: Per-detector templates (det_uid, amplitude, phase, quality, scaling_factor)

Functions are organized into:
1. Database queries
2. Data processing and statistics
3. Visualization functions
"""

import os
import sqlite3
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.dates import DateFormatter, YearLocator, MonthLocator
from datetime import datetime


# =============================================================================
# DATABASE QUERY FUNCTIONS
# =============================================================================

def query_database(db_path, det_uid=None, start_time=None, end_time=None,
                   span_path=None, quality_threshold=None, boresight=None,
                   pwv_range=None):
    """Query the Q-band template database for template information.

    This function provides a flexible interface to extract template data from
    the database with various filtering options. Results include amplitude,
    phase, quality, and metadata for each template.

    Parameters
    ----------
    db_path : str
        Path to the Q-band template database file
    det_uid : int or list of int, optional
        Detector UID(s) to query. If None, returns all detectors.
    start_time : float, optional
        Start time (ctime) for filtering segments
    end_time : float, optional
        End time (ctime) for filtering segments
    span_path : str, optional
        Specific span path to query (e.g., '2024-07-01-21-30-00_71')
    quality_threshold : float, optional
        Minimum quality threshold for templates (0-1)
    boresight : int or list of int, optional
        Boresight angle(s) in degrees to filter by
    pwv_range : tuple of (float, float), optional
        PWV range (min, max) in mm for filtering

    Returns
    -------
    dict
        Dictionary with 2D grid format optimized for vectorized operations:

        **2D Measurement Arrays** [ndet, ntime]:
        - 'amplitudes': template amplitudes (normalized)
        - 'amplitude_errors': amplitude fit errors
        - 'phases': template phases
        - 'phase_errors': phase fit errors
        - 'quality': template quality (R² values)
        - 'scaling_factors': normalization factors
        - 'original_amplitudes': unnormalized amplitudes

        **1D Coordinate Arrays**:
        - 'timestamps': unique timestamps [ntime]
        - 'det_uids': unique detector UIDs [ndet]

        **1D Per-Time Metadata** [ntime]:
        - 'spans': span IDs (identifies which original span each timestamp belongs to)
        - 'span_paths': span path strings (for reference)
        - 'boresights': boresight angles
        - 'elevations': elevation angles
        - 'pwv': PWV values
        - 'mod_freq': modulation frequencies
        - 'modulator_type': modulator types ('vpm' or 'hwp')
        - 'subdivision_index': segment subdivision indices

        Missing detector-time combinations are filled with NaN.

    Examples
    --------
    >>> # Query all detectors with quality > 0.8
    >>> data = query_database(db_path, quality_threshold=0.8)

    >>> # Query specific detector at specific boresight
    >>> data = query_database(db_path, det_uid=516, boresight=45)

    >>> # Query PWV range
    >>> data = query_database(db_path, pwv_range=(0.5, 2.0))
    """
    if not os.path.exists(db_path):
        print(f"Database not found: {db_path}")
        return {}

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    try:
        # Build query
        query = """
            SELECT
                segments.timestamp,
                templates.det_uid,
                templates.amplitude,
                templates.amplitude_error,
                templates.phase,
                templates.phase_error,
                templates.quality,
                templates.scaling_factor,
                templates.original_amplitude,
                segments.span_id,
                spans.path,
                spans.boresight,
                spans.elevation,
                segments.site_pwv,
                segments.mod_freq,
                segments.modulator_type,
                segments.subdivision_index
            FROM templates
            JOIN segments ON templates.segment_id = segments.id
            JOIN spans ON segments.span_id = spans.id
            WHERE 1=1
        """

        params = []

        # Add filters
        if det_uid is not None:
            if isinstance(det_uid, (list, tuple, np.ndarray)):
                placeholders = ','.join(['?'] * len(det_uid))
                query += f" AND templates.det_uid IN ({placeholders})"
                params.extend(det_uid)
            else:
                query += " AND templates.det_uid = ?"
                params.append(det_uid)

        if start_time is not None:
            query += " AND segments.timestamp >= ?"
            params.append(start_time)

        if end_time is not None:
            query += " AND segments.timestamp <= ?"
            params.append(end_time)

        if span_path is not None:
            query += " AND spans.path = ?"
            params.append(span_path)

        if quality_threshold is not None:
            query += " AND templates.quality >= ?"
            params.append(quality_threshold)

        if boresight is not None:
            if isinstance(boresight, (list, tuple)):
                placeholders = ','.join(['?'] * len(boresight))
                query += f" AND spans.boresight IN ({placeholders})"
                params.extend(boresight)
            else:
                query += " AND spans.boresight = ?"
                params.append(boresight)

        if pwv_range is not None:
            query += " AND segments.site_pwv >= ? AND segments.site_pwv <= ?"
            params.extend([pwv_range[0], pwv_range[1]])
            query += " AND segments.site_pwv IS NOT NULL"

        query += " ORDER BY segments.timestamp, templates.det_uid"

        # Execute query
        cursor.execute(query, params)
        rows = cursor.fetchall()

        if len(rows) == 0:
            return {}

        # Convert to flat 1D arrays first
        timestamps_flat = np.array([row['timestamp'] for row in rows])
        det_uids_flat = np.array([row['det_uid'] for row in rows])
        amplitudes_flat = np.array([row['amplitude'] for row in rows])
        amplitude_errors_flat = np.array(
            [row['amplitude_error'] if row['amplitude_error'] is not None else np.nan for row in rows])
        phases_flat = np.array([row['phase'] for row in rows])
        phase_errors_flat = np.array([row['phase_error'] if row['phase_error']
                                     is not None else np.nan for row in rows])
        quality_flat = np.array([row['quality'] for row in rows])
        scaling_factors_flat = np.array([row['scaling_factor'] for row in rows])
        original_amplitudes_flat = np.array([row['original_amplitude'] for row in rows])
        span_ids_flat = np.array([row['span_id'] for row in rows])
        spans_flat = np.array([row['path'] for row in rows])
        boresights_flat = np.array([row['boresight'] if row['boresight']
                                   is not None else np.nan for row in rows])
        elevations_flat = np.array([row['elevation'] if row['elevation']
                                   is not None else np.nan for row in rows])
        pwv_flat = np.array([row['site_pwv'] if row['site_pwv'] is not None else np.nan for row in rows])
        mod_freq_flat = np.array([row['mod_freq'] if row['mod_freq'] is not None else np.nan for row in rows])
        modulator_type_flat = np.array([row['modulator_type'] for row in rows])
        subdivision_index_flat = np.array([row['subdivision_index'] for row in rows])

        # Reshape to 2D grid format [ndet, ntime]
        unique_times = np.unique(timestamps_flat)
        unique_dets = np.unique(det_uids_flat)

        n_dets = len(unique_dets)
        n_times = len(unique_times)

        # Create mapping dictionaries
        time_to_idx = {t: i for i, t in enumerate(unique_times)}
        det_to_idx = {d: i for i, d in enumerate(unique_dets)}

        # Initialize 2D arrays for measurements
        amplitudes_2d = np.full((n_dets, n_times), np.nan)
        amplitude_errors_2d = np.full((n_dets, n_times), np.nan)
        phases_2d = np.full((n_dets, n_times), np.nan)
        phase_errors_2d = np.full((n_dets, n_times), np.nan)
        quality_2d = np.full((n_dets, n_times), np.nan)
        scaling_factors_2d = np.full((n_dets, n_times), np.nan)
        original_amplitudes_2d = np.full((n_dets, n_times), np.nan)

        # Initialize 1D arrays for per-time metadata (each timestamp has one value)
        spans_1d = np.empty(n_times, dtype=object)
        span_ids_1d = np.full(n_times, -1, dtype=int)
        boresights_1d = np.full(n_times, np.nan)
        elevations_1d = np.full(n_times, np.nan)
        pwv_1d = np.full(n_times, np.nan)
        mod_freq_1d = np.full(n_times, np.nan)
        modulator_type_1d = np.empty(n_times, dtype=object)
        subdivision_index_1d = np.full(n_times, -1, dtype=int)

        # Fill in data
        for i in range(len(rows)):
            t_idx = time_to_idx[timestamps_flat[i]]
            d_idx = det_to_idx[det_uids_flat[i]]

            # Measurement data (2D)
            amplitudes_2d[d_idx, t_idx] = amplitudes_flat[i]
            amplitude_errors_2d[d_idx, t_idx] = amplitude_errors_flat[i]
            phases_2d[d_idx, t_idx] = phases_flat[i]
            phase_errors_2d[d_idx, t_idx] = phase_errors_flat[i]
            quality_2d[d_idx, t_idx] = quality_flat[i]
            scaling_factors_2d[d_idx, t_idx] = scaling_factors_flat[i]
            original_amplitudes_2d[d_idx, t_idx] = original_amplitudes_flat[i]

            # Per-time metadata (1D) - same for all detectors at this time
            if span_ids_1d[t_idx] == -1:
                spans_1d[t_idx] = spans_flat[i]
                span_ids_1d[t_idx] = span_ids_flat[i]
                boresights_1d[t_idx] = boresights_flat[i]
                elevations_1d[t_idx] = elevations_flat[i]
                pwv_1d[t_idx] = pwv_flat[i]
                mod_freq_1d[t_idx] = mod_freq_flat[i]
                modulator_type_1d[t_idx] = modulator_type_flat[i]
                subdivision_index_1d[t_idx] = subdivision_index_flat[i]

        # Package results in 2D grid format
        results = {
            # 2D measurement arrays [ndet, ntime]
            'amplitudes': amplitudes_2d,
            'amplitude_errors': amplitude_errors_2d,
            'phases': phases_2d,
            'phase_errors': phase_errors_2d,
            'quality': quality_2d,
            'scaling_factors': scaling_factors_2d,
            'original_amplitudes': original_amplitudes_2d,
            # 1D coordinate arrays
            'timestamps': unique_times,  # [ntime]
            'det_uids': unique_dets,     # [ndet]
            # 1D per-time metadata [ntime]
            'spans': span_ids_1d,        # Span IDs (for CM algorithm)
            'span_paths': spans_1d,      # Span paths (for reference)
            'boresights': boresights_1d,
            'elevations': elevations_1d,
            'pwv': pwv_1d,
            'mod_freq': mod_freq_1d,
            'modulator_type': modulator_type_1d,
            'subdivision_index': subdivision_index_1d,
        }

        return results

    except Exception as e:
        print(f"Database query error: {e}")
        import traceback
        traceback.print_exc()
        return {}

    finally:
        conn.close()


def get_unique_detector_uids(db_path):
    """Get sorted list of all unique detector UIDs in the database.

    Parameters
    ----------
    db_path : str
        Path to the database file

    Returns
    -------
    list of int
        Sorted list of unique detector UIDs
    """
    if not os.path.exists(db_path):
        print(f"Database not found: {db_path}")
        return []

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    try:
        cursor.execute("SELECT DISTINCT det_uid FROM templates ORDER BY det_uid")
        uids = [row[0] for row in cursor.fetchall()]
        return uids
    except Exception as e:
        print(f"Database error: {e}")
        return []
    finally:
        conn.close()


def get_unique_boresights(db_path):
    """Get sorted list of all unique boresight angles in the database.

    Parameters
    ----------
    db_path : str
        Path to the database file

    Returns
    -------
    list of int
        Sorted list of unique boresight angles in degrees
    """
    if not os.path.exists(db_path):
        print(f"Database not found: {db_path}")
        return []

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    try:
        cursor.execute("""
            SELECT DISTINCT boresight
            FROM spans
            WHERE boresight IS NOT NULL
            ORDER BY boresight
        """)
        boresights = [row[0] for row in cursor.fetchall()]
        return boresights
    except Exception as e:
        print(f"Database error: {e}")
        return []
    finally:
        conn.close()


# =============================================================================
# DATA PROCESSING FUNCTIONS
# =============================================================================

def compute_detector_statistics(data, group_by_boresight=False, time_mask=None):
    """Compute per-detector statistics from query results.

    This function calculates median, variance, and other statistics for each
    detector from the raw template data. Optionally groups by boresight angle
    or filters by time mask.

    Parameters
    ----------
    data : dict
        Query results from query_database()
    group_by_boresight : bool, optional
        If True, compute statistics separately for each boresight angle.
        Default is False (combine all boresights).
    time_mask : array of bool, optional
        Boolean mask [ntime] to select subset of time points.
        If None, uses all time points. Default is None.

    Returns
    -------
    dict
        If group_by_boresight=False:
            Dictionary keyed by detector UID with values containing:
            - 'median_amplitude': median of amplitude × scaling_factor
            - 'std_amplitude': standard deviation
            - 'var_amplitude': variance
            - 'n_samples': number of measurements
            - 'median_quality': median quality score
            - 'median_pwv': median PWV (if available)

        If group_by_boresight=True:
            Nested dictionary: {boresight: {det_uid: stats}}

    Examples
    --------
    >>> data = query_database(db_path, quality_threshold=0.8)
    >>> stats = compute_detector_statistics(data)
    >>> print(stats[516]['median_amplitude'])

    >>> # Nighttime only
    >>> night_mask = data['sun_elevation'] < 0
    >>> stats_night = compute_detector_statistics(data, time_mask=night_mask)

    >>> # Group by boresight
    >>> stats_by_bs = compute_detector_statistics(data, group_by_boresight=True)
    >>> print(stats_by_bs[45][516]['median_amplitude'])
    """
    if not data or len(data['det_uids']) == 0:
        return {}

    # Compute physical amplitudes (denormalized) [ndet, ntime]
    phys_amps = np.abs(data['amplitudes'] * data['scaling_factors'])

    # Apply time mask if provided
    if time_mask is not None:
        phys_amps = phys_amps[:, time_mask]
        # Also need to mask time-dependent arrays used later
        quality_masked = data['quality'][:, time_mask]
        pwv_masked = data['pwv'][time_mask] if 'pwv' in data else None
        boresights_masked = data['boresights'][time_mask] if group_by_boresight else None
    else:
        quality_masked = data['quality']
        pwv_masked = data['pwv'] if 'pwv' in data else None
        boresights_masked = data['boresights'] if group_by_boresight else None

    if not group_by_boresight:
        # Compute stats for all boresights combined
        # Data format: phys_amps is [ndet, ntime], det_uids is [ndet], pwv is [ntime]
        stats = {}

        for d_idx, uid in enumerate(data['det_uids']):
            # Extract this detector's row [ntime]
            uid_amps = phys_amps[d_idx, :]
            uid_quality = quality_masked[d_idx, :]

            # Remove NaN values
            valid_mask = ~np.isnan(uid_amps)
            uid_amps_valid = uid_amps[valid_mask]
            uid_quality_valid = uid_quality[valid_mask]

            # PWV is per-time, so get values for valid times
            uid_pwv_valid = pwv_masked[valid_mask] if pwv_masked is not None else np.array([])

            # Calculate median PWV
            median_pwv = (np.nanmedian(uid_pwv_valid)
                          if len(uid_pwv_valid) > 0 and not np.all(np.isnan(uid_pwv_valid))
                          else np.nan)

            stats[int(uid)] = {
                'median_amplitude': np.nanmedian(uid_amps_valid) if len(uid_amps_valid) > 0 else np.nan,
                'std_amplitude': np.nanstd(uid_amps_valid) if len(uid_amps_valid) > 0 else np.nan,
                'var_amplitude': np.nanvar(uid_amps_valid) if len(uid_amps_valid) > 0 else np.nan,
                'n_samples': len(uid_amps_valid),
                'median_quality': np.nanmedian(uid_quality_valid) if len(uid_quality_valid) > 0 else np.nan,
                'median_pwv': median_pwv
            }

        return stats

    else:
        # Group by boresight
        # Data format: boresights is [ntime], det_uids is [ndet], phys_amps is [ndet, ntime]
        stats_by_bs = {}
        unique_boresights = np.unique(boresights_masked[~np.isnan(boresights_masked)])

        for bs in unique_boresights:
            # Create time mask for this boresight [ntime]
            bs_time_mask = np.abs(boresights_masked - bs) < 0.1  # tolerance

            stats_by_bs[int(bs)] = {}

            for d_idx, uid in enumerate(data['det_uids']):
                # Extract this detector's data [ntime]
                uid_amps = phys_amps[d_idx, :]
                uid_quality = quality_masked[d_idx, :]

                # Apply boresight mask and remove NaNs
                uid_amps_bs = uid_amps[bs_time_mask]
                uid_quality_bs = uid_quality[bs_time_mask]

                valid_mask = ~np.isnan(uid_amps_bs)
                uid_amps_valid = uid_amps_bs[valid_mask]
                uid_quality_valid = uid_quality_bs[valid_mask]

                # Get PWV for this boresight
                uid_pwv_bs = pwv_masked[bs_time_mask] if pwv_masked is not None else np.array([])
                uid_pwv_valid = uid_pwv_bs[valid_mask] if len(uid_pwv_bs) > 0 else np.array([])

                if len(uid_amps_valid) > 0:
                    # Calculate median PWV
                    median_pwv_bs = (np.nanmedian(uid_pwv_valid)
                                     if len(uid_pwv_valid) > 0 and not np.all(np.isnan(uid_pwv_valid))
                                     else np.nan)

                    stats_by_bs[int(bs)][int(uid)] = {
                        'median_amplitude': np.nanmedian(uid_amps_valid),
                        'std_amplitude': np.nanstd(uid_amps_valid),
                        'var_amplitude': np.nanvar(uid_amps_valid),
                        'n_samples': len(uid_amps_valid),
                        'median_quality': np.nanmedian(uid_quality_valid),
                        'median_pwv': median_pwv_bs
                    }

        return stats_by_bs


def identify_outlier_detectors(stats, zscore_threshold=2.0, min_samples=10):
    """Identify detectors with anomalous amplitudes or high variance.

    This uses z-score analysis within detector modules to find outliers.

    Parameters
    ----------
    stats : dict
        Output from compute_detector_statistics()
    zscore_threshold : float, optional
        Z-score threshold for flagging outliers. Default is 2.0.
    min_samples : int, optional
        Minimum number of samples required for a detector to be analyzed.
        Default is 10.

    Returns
    -------
    dict
        Dictionary with keys:
        - 'low_amplitude': list of UIDs with unusually low amplitudes
        - 'high_variance': list of UIDs with unusually high variance
        - 'low_samples': list of UIDs with too few samples
    """
    if not stats:
        return {'low_amplitude': [], 'high_variance': [], 'low_samples': []}

    # Extract arrays
    uids = np.array(list(stats.keys()))
    medians = np.array([stats[uid]['median_amplitude'] for uid in uids])
    variances = np.array([stats[uid]['var_amplitude'] for uid in uids])
    n_samples = np.array([stats[uid]['n_samples'] for uid in uids])

    # Filter by minimum samples
    good_sample_mask = n_samples >= min_samples
    low_sample_uids = uids[~good_sample_mask].tolist()

    uids_filtered = uids[good_sample_mask]
    medians_filtered = medians[good_sample_mask]
    variances_filtered = variances[good_sample_mask]

    # Compute z-scores
    median_zscore = (medians_filtered - np.median(medians_filtered)) / np.std(medians_filtered)
    var_zscore = (variances_filtered - np.median(variances_filtered)) / np.std(variances_filtered)

    # Identify outliers
    low_amp_uids = uids_filtered[median_zscore < -zscore_threshold].tolist()
    high_var_uids = uids_filtered[var_zscore > zscore_threshold].tolist()

    return {
        'low_amplitude': low_amp_uids,
        'high_variance': high_var_uids,
        'low_samples': low_sample_uids
    }


# =============================================================================
# COMMON-MODE REMOVAL FUNCTIONS
# =============================================================================

def compute_common_mode(data, method='robust_median', n_components=1, sigma_clip=3.0,
                        min_detectors=3, quality_threshold=None, normalize_detectors=False,
                        smooth_hours=None):
    """Compute common-mode signal across detectors.

    The common-mode represents systematic variations that affect all detectors
    similarly (e.g., atmospheric loading, thermal variations, day-night cycles).

    Parameters
    ----------
    data : dict
        Query results from query_database()
    method : str, optional
        Method to compute common mode:
        - 'robust_median': Median across detectors at each time (default)
        - 'pca': First principal component of detector time series
        Default is 'robust_median'.
    n_components : int, optional
        Number of PCA components to compute (only for method='pca'). Default is 1.
    sigma_clip : float, optional
        Sigma threshold for outlier rejection (only for method='robust_median'). Default is 3.0.
        Set to None to disable sigma clipping.
    min_detectors : int, optional
        Minimum number of detectors required at each time point. Default is 3.
    quality_threshold : float, optional
        If provided, only use templates with quality >= threshold
    normalize_detectors : bool, optional
        If True, normalize each detector's time series to zero mean and unit variance
        before computing common mode. Default is False.
    smooth_hours : float, optional
        If provided, apply Gaussian smoothing with this FWHM window size (in hours).
        This extracts smooth, differentiable low-frequency variations (e.g., day-night cycles).
        Recommended: 2-6 hours for day-night cycles. Default is None (no smoothing).

    Returns
    -------
    dict
        Dictionary with keys:
        - 'timestamps': Array of unique timestamps
        - 'common_mode': Common-mode amplitude at each timestamp
        - 'common_mode_raw': (if smoothed) Raw common mode before smoothing
        - 'n_detectors': Number of detectors contributing at each time
        - 'method': Method used
        - Additional keys depending on method
    """
    # Apply quality filter if requested
    if quality_threshold is not None:
        mask = data['quality'] >= quality_threshold
        filtered_data = {k: v[mask] if isinstance(v, np.ndarray) else v
                         for k, v in data.items()}
    else:
        filtered_data = data

    # Compute physical amplitudes
    phys_amps = np.abs(filtered_data['amplitudes'] * filtered_data['scaling_factors'])

    # Get unique timestamps and detectors
    unique_times = np.unique(filtered_data['timestamps'])
    det_uids = np.unique(filtered_data['det_uids'])

    print("\nCommon-mode computation:")
    print(f"  Method: {method}")
    print(f"  Unique detectors: {len(det_uids)}")
    print(f"  Unique times: {len(unique_times)}")
    print(f"  Total measurements: {len(phys_amps)}")

    if method == 'robust_median':
        common_mode = []
        n_dets = []
        valid_times = []

        for t in unique_times:
            time_mask = filtered_data['timestamps'] == t
            amps_at_time = phys_amps[time_mask]

            if len(amps_at_time) < min_detectors:
                continue

            # Compute median and optionally sigma-clip
            if sigma_clip is not None and len(amps_at_time) >= 5:
                # Iterative sigma clipping
                amps_clip = amps_at_time.copy()
                for _ in range(3):  # Max 3 iterations
                    median_amp = np.median(amps_clip)
                    std_amp = np.std(amps_clip)
                    if std_amp > 0:
                        good = np.abs(amps_clip - median_amp) < sigma_clip * std_amp
                        if np.sum(good) >= min_detectors:
                            amps_clip = amps_clip[good]
                        else:
                            break
                    else:
                        break
                cm_value = np.median(amps_clip)
            else:
                cm_value = np.median(amps_at_time)

            common_mode.append(cm_value)
            n_dets.append(len(amps_at_time))
            valid_times.append(t)

        cm_array = np.array(common_mode)
        times_array = np.array(valid_times)
        n_dets_array = np.array(n_dets)

        print(f"  Valid time points: {len(valid_times)}")
        print(f"  Median detectors per time: {np.median(n_dets_array):.0f}")
        print(f"  Raw CM range: [{np.min(cm_array):.6f}, {np.max(cm_array):.6f}]")
        print(f"  Raw CM mean: {np.mean(cm_array):.6f}")
        print(f"  Raw CM std: {np.std(cm_array):.6f}")
        print(f"  Raw CM variation: {100*np.std(cm_array)/np.mean(cm_array):.1f}%")

        # Apply smoothing to extract low-frequency component if requested
        cm_raw = cm_array.copy()
        if smooth_hours is not None:
            from scipy.ndimage import gaussian_filter1d

            # Convert time window to sigma in samples
            # For Gaussian filter: FWHM ≈ 2.355 * sigma
            # We want FWHM = smooth_hours, so sigma = smooth_hours / 2.355
            if len(times_array) > 1:
                time_diffs = np.diff(times_array)
                median_dt = np.median(time_diffs)  # in seconds

                # sigma in seconds
                sigma_seconds = (smooth_hours * 3600) / 2.355
                # sigma in samples
                sigma_samples = sigma_seconds / median_dt

                print("\nSmoothing (Gaussian filter for differentiable result):")
                print(f"  Window (FWHM): {smooth_hours} hours")
                print(f"  Sigma: {sigma_samples:.1f} samples ({sigma_seconds/3600:.2f} hours)")
                print(f"  Median time spacing: {median_dt:.1f} seconds")

                # Apply Gaussian filter - produces smooth, differentiable curve
                cm_array = gaussian_filter1d(cm_array, sigma=sigma_samples, mode='nearest')

                print(f"  Smoothed CM range: [{np.min(cm_array):.6f}, {np.max(cm_array):.6f}]")
                print(f"  Smoothed CM std: {np.std(cm_array):.6f}")
                print(f"  Smoothed CM variation: {100*np.std(cm_array)/np.mean(cm_array):.1f}%")
            else:
                print("\n  Warning: Not enough points to smooth")

        result = {
            'timestamps': times_array,
            'common_mode': cm_array,
            'n_detectors': n_dets_array,
            'method': 'robust_median',
            'sigma_clip': sigma_clip,
            'smooth_hours': smooth_hours
        }

        if smooth_hours is not None:
            result['common_mode_raw'] = cm_raw

        return result

    elif method == 'pca':
        # Build detector × time matrix (rows=detectors, cols=times)
        amp_matrix = np.full((len(det_uids), len(unique_times)), np.nan)

        for i, det_uid in enumerate(det_uids):
            det_mask = filtered_data['det_uids'] == det_uid
            det_times = filtered_data['timestamps'][det_mask]
            det_amps = phys_amps[det_mask]

            for t_idx, t in enumerate(unique_times):
                matching = det_times == t
                if np.any(matching):
                    amp_matrix[i, t_idx] = np.mean(det_amps[matching])

        # Filter out times and detectors with too much missing data
        time_coverage = np.sum(~np.isnan(amp_matrix), axis=0) / len(det_uids)
        det_coverage = np.sum(~np.isnan(amp_matrix), axis=1) / len(unique_times)

        good_times = time_coverage >= (min_detectors / len(det_uids))
        good_dets = det_coverage >= 0.3  # Detector must have at least 30% coverage

        amp_matrix_filt = amp_matrix[good_dets][:, good_times]
        clean_times = unique_times[good_times]
        clean_det_uids = det_uids[good_dets]

        print(f"  Detectors after filtering: {len(clean_det_uids)} / {len(det_uids)}")
        print(f"  Times after filtering: {len(clean_times)} / {len(unique_times)}")

        if amp_matrix_filt.size == 0:
            raise ValueError("No data remaining after filtering")

        # Normalize each detector's time series if requested
        if normalize_detectors:
            for i in range(amp_matrix_filt.shape[0]):
                row_mean = np.nanmean(amp_matrix_filt[i])
                row_std = np.nanstd(amp_matrix_filt[i])
                if row_std > 0:
                    amp_matrix_filt[i] = (amp_matrix_filt[i] - row_mean) / row_std

        # Impute missing values with row median
        for i in range(amp_matrix_filt.shape[0]):
            row_median = np.nanmedian(amp_matrix_filt[i])
            amp_matrix_filt[i, np.isnan(amp_matrix_filt[i])] = row_median

        # Check for remaining NaNs
        if np.any(np.isnan(amp_matrix_filt)):
            print(f"  Warning: {np.sum(np.isnan(amp_matrix_filt))} NaNs remaining after imputation")
            amp_matrix_filt = np.nan_to_num(amp_matrix_filt, nan=0.0)

        # Perform PCA on detector time series
        # Each detector is a sample, each time point is a feature
        from sklearn.decomposition import PCA
        pca = PCA(n_components=min(n_components, amp_matrix_filt.shape[0]))

        # Fit PCA (samples=detectors, features=times)
        pca.fit(amp_matrix_filt)

        # The first principal component in time-space is the common mode
        # This is a vector of length = n_times
        common_mode_pattern = pca.components_[0]  # Shape: (n_times,)

        # Scale it to have the same units as the original amplitudes
        # Project all detectors onto this component and take the mean
        projection_coeffs = pca.transform(amp_matrix_filt)[:, 0]  # Shape: (n_detectors,)
        mean_coeff = np.mean(projection_coeffs)

        # Reconstruct common mode signal in original amplitude units
        common_mode_signal = mean_coeff * common_mode_pattern

        # If we normalized, the common mode is in z-score units
        # Scale back to approximate amplitude units using overall mean
        if normalize_detectors:
            overall_mean = np.mean(phys_amps)
            overall_std = np.std(phys_amps)
            common_mode_signal = common_mode_signal * overall_std + overall_mean

        print(f"  PCA variance explained: {pca.explained_variance_ratio_[0]:.1%}")
        print(f"  Common-mode range: [{np.min(common_mode_signal):.6f}, {np.max(common_mode_signal):.6f}]")
        print(f"  Common-mode mean: {np.mean(common_mode_signal):.6f}")
        print(f"  Common-mode std: {np.std(common_mode_signal):.6f}")

        return {
            'timestamps': clean_times,
            'common_mode': common_mode_signal,
            'n_detectors': np.sum(good_times).astype(int),  # Detectors per time is variable
            'method': 'pca',
            'pca_object': pca,
            'variance_explained': pca.explained_variance_ratio_,
            'projection_coefficients': projection_coeffs,
            'detector_uids': clean_det_uids
        }

    else:
        raise ValueError(f"Unknown method: {method}. Use 'robust_median' or 'pca'.")


def remove_common_mode(data, common_mode_result, interpolate=True):
    """Remove common-mode signal from detector amplitudes.

    Works with 2D grid format [ndet, ntime].

    Parameters
    ----------
    data : dict
        Query results from query_database() in 2D grid format
    common_mode_result : dict
        Output from compute_common_mode()
    interpolate : bool, optional
        If True, interpolate common mode to match data timestamps. Default is True.

    Returns
    -------
    dict
        New data dictionary with common-mode removed from amplitudes.
        Adds 'amplitudes_cm_removed' and 'scaling_factors' fields.
    """
    # Physical amplitudes [ndet, ntime]
    phys_amps = data['amplitudes'] * data['scaling_factors']

    # Interpolate common mode to data timestamps if needed [ntime]
    if interpolate:
        common_mode_interp = np.interp(
            data['timestamps'],
            common_mode_result['timestamps'],
            common_mode_result['common_mode'],
            left=np.nan,
            right=np.nan
        )
    else:
        # Find nearest timestamp
        common_mode_interp = np.zeros(len(data['timestamps']))
        for i, t in enumerate(data['timestamps']):
            idx = np.argmin(np.abs(common_mode_result['timestamps'] - t))
            common_mode_interp[i] = common_mode_result['common_mode'][idx]

    # Subtract common mode (broadcast [ntime] across all detectors)
    # phys_amps is [ndet, ntime], common_mode_interp is [ntime]
    # Broadcasting: each detector gets same CM subtracted at each time
    phys_amps_corrected = phys_amps - common_mode_interp[np.newaxis, :]

    # Create new data dict
    corrected_data = data.copy()
    corrected_data['amplitudes_cm_removed'] = phys_amps_corrected / data['scaling_factors']
    corrected_data['common_mode_signal'] = common_mode_interp

    return corrected_data


# =============================================================================
# VISUALIZATION FUNCTIONS
# =============================================================================

def plot_detector_amplitudes_vs_time(data, det_uids, db_path=None,
                                     figsize=None, y_lim=None,
                                     quality_threshold=None,
                                     show_violin=True,
                                     time_range=None):
    """Plot amplitude time series for multiple detectors in a multi-panel figure.

    This creates a stacked subplot for each detector showing how the amplitude
    (amplitude × scaling_factor) varies over time. Useful for identifying
    detector drift, instabilities, or systematic variations.

    Parameters
    ----------
    data : dict
        Query results from query_database()
    det_uids : list of int
        List of detector UIDs to plot
    db_path : str, optional
        Database path (used for title only)
    figsize : tuple, optional
        Figure size (width, height). Default is (12, 2*len(det_uids)).
    y_lim : tuple, optional
        Y-axis limits (ymin, ymax). If None, auto-scales.
    quality_threshold : float, optional
        If provided, filters data to quality >= threshold
    show_violin : bool, optional
        If True, shows violin plot of amplitude distribution next to each time series.
        Default is True.
    time_range : tuple of datetime, optional
        Time range (start_date, end_date) to plot. If None, shows all data.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The generated figure

    Examples
    --------
    >>> data = query_database(db_path, det_uid=[1, 33, 516])
    >>> fig = plot_detector_amplitudes_vs_time(data, [1, 33, 516])
    >>> plt.savefig('detector_timeseries.png', dpi=150)

    >>> # Zoom to specific date range
    >>> from datetime import datetime
    >>> fig = plot_detector_amplitudes_vs_time(data, [1, 33, 516],
    ...     time_range=(datetime(2024, 6, 1), datetime(2024, 8, 1)))
    """
    if figsize is None:
        figsize = (14, max(2, 1.5 * len(det_uids)))

    # Apply quality filter if requested (filter in 2D)
    if quality_threshold is not None:
        quality_mask_2d = data['quality'] >= quality_threshold
        # For 2D arrays, set low-quality measurements to NaN
        filtered_data = data.copy()
        filtered_data['amplitudes'] = np.where(quality_mask_2d, data['amplitudes'], np.nan)
        filtered_data['scaling_factors'] = np.where(quality_mask_2d, data['scaling_factors'], np.nan)
    else:
        filtered_data = data

    # Apply time range filter if requested
    if time_range is not None:
        start_ctime = time_range[0].timestamp()
        end_ctime = time_range[1].timestamp()
        time_mask = (filtered_data['timestamps'] >= start_ctime) & (filtered_data['timestamps'] <= end_ctime)

        # Filter 2D arrays along time axis, 1D time arrays normally
        filtered_data = {
            **{k: filtered_data[k][:, time_mask] for k in ['amplitudes', 'amplitude_errors', 'phases',
                                                           'phase_errors', 'quality', 'scaling_factors',
                                                           'original_amplitudes']},
            **{k: filtered_data[k][time_mask] for k in ['timestamps', 'spans', 'span_paths',
                                                        'boresights', 'elevations', 'pwv',
                                                        'mod_freq', 'modulator_type', 'subdivision_index']
               if k in filtered_data},
            'det_uids': filtered_data['det_uids']  # Keep detector array unchanged
        }

    # Compute physical amplitudes [ndet, ntime]
    phys_amps = np.abs(filtered_data['amplitudes'] * filtered_data['scaling_factors'])

    # Convert timestamps to datetime objects
    dates = [datetime.fromtimestamp(t) for t in filtered_data['timestamps']]

    # Setup figure
    fig = plt.figure(figsize=figsize)
    if show_violin:
        gs = GridSpec(len(det_uids), 2, width_ratios=[10, 1], wspace=0.0, hspace=0)
    else:
        gs = GridSpec(len(det_uids), 1, hspace=0)

    first_ax = None

    # Determine time span for axis formatting
    all_dates = np.array(dates)
    if len(all_dates) > 0:
        time_span_days = (all_dates.max() - all_dates.min()).total_seconds() / 86400
    else:
        time_span_days = 365

    for i, uid in enumerate(det_uids):
        # Get data for this detector (2D format: find detector index, extract row)
        det_idx = np.where(filtered_data['det_uids'] == uid)[0]

        if len(det_idx) == 0:
            print(f"Warning: Detector {uid} not found in data")
            continue

        det_idx = det_idx[0]

        # Extract detector row [ntime]
        uid_amps = phys_amps[det_idx, :]
        uid_quality = filtered_data['quality'][det_idx, :]

        # Filter out NaN values for this detector
        valid_mask = ~np.isnan(uid_amps)
        uid_dates = [dates[j] for j in range(len(dates)) if valid_mask[j]]
        uid_amps = uid_amps[valid_mask]
        uid_quality = uid_quality[valid_mask]

        if len(uid_dates) == 0:
            continue

        # Create time series axis
        if i == 0:
            ax_time = fig.add_subplot(gs[i, 0])
            first_ax = ax_time
        else:
            ax_time = fig.add_subplot(gs[i, 0], sharex=first_ax)

        # Plot time series
        ax_time.scatter(uid_dates, uid_amps, s=4, alpha=0.6, color='C0')

        # Compute statistics
        med = np.median(uid_amps)
        std = np.std(uid_amps)

        # Add annotation
        ax_time.text(0.02, 0.85, f"UID {uid}: med={med:.3f}, σ={std:.3f}, n={len(uid_amps)}",
                     transform=ax_time.transAxes, fontsize=9, fontweight='bold')

        ax_time.grid(True, alpha=0.3)

        if y_lim is not None:
            ax_time.set_ylim(y_lim)

        # Violin plot
        if show_violin and len(uid_amps) > 1:
            ax_violin = fig.add_subplot(gs[i, 1], sharey=ax_time)
            violin_parts = ax_violin.violinplot([uid_amps], positions=[0],
                                                vert=True, widths=0.3,
                                                showmeans=True, showextrema=True)
            for pc in violin_parts['bodies']:
                pc.set_facecolor('C0')
                pc.set_alpha(0.4)

            ax_violin.grid(False)
            ax_violin.tick_params(left=False, labelleft=False,
                                  bottom=False, labelbottom=False)
            ax_violin.set_xlim(-0.5, 0.5)

        # X-axis labels only on bottom plot
        if i < len(det_uids) - 1:
            ax_time.tick_params(labelbottom=False)

        # Y-axis label in middle
        if i == len(det_uids) // 2:
            ax_time.set_ylabel('Amplitude × Scaling Factor')

    # Format x-axis with dates
    if first_ax is not None:
        # Choose formatter based on time span
        if time_span_days > 365:
            # More than a year: use years
            ax_time.xaxis.set_major_locator(YearLocator())
            ax_time.xaxis.set_major_formatter(DateFormatter('%Y'))
            ax_time.set_xlabel('Year')
        elif time_span_days > 60:
            # 2 months to 1 year: use months and years
            ax_time.xaxis.set_major_locator(MonthLocator(interval=2))
            ax_time.xaxis.set_major_formatter(DateFormatter('%b %Y'))
            ax_time.set_xlabel('Date')
            plt.setp(ax_time.xaxis.get_majorticklabels(), rotation=45, ha='right')
        elif time_span_days > 7:
            # 1 week to 2 months: use months and days
            ax_time.xaxis.set_major_locator(MonthLocator())
            ax_time.xaxis.set_major_formatter(DateFormatter('%b %d'))
            ax_time.set_xlabel('Date')
            plt.setp(ax_time.xaxis.get_majorticklabels(), rotation=45, ha='right')
        else:
            # Less than 1 week: use days and hours
            from matplotlib.dates import DayLocator
            ax_time.xaxis.set_major_locator(DayLocator())
            ax_time.xaxis.set_major_formatter(DateFormatter('%b %d\n%H:%M'))
            ax_time.set_xlabel('Date and Time')
            plt.setp(ax_time.xaxis.get_majorticklabels(), rotation=45, ha='right')

    # Title
    title = "VPM Template Amplitudes vs. Time"
    if quality_threshold is not None:
        title += f" (quality ≥ {quality_threshold})"
    fig.suptitle(title, fontsize=12, y=0.995)

    fig.subplots_adjust(top=0.96, bottom=0.08, hspace=0.0, wspace=0.0,
                        right=0.98 if show_violin else 0.99, left=0.1)

    return fig


def plot_detector_amplitudes_summary(stats, array_data, figsize=(12, 7),
                                     color_by='auto', normalize_by_group=False,
                                     outlier_threshold=None):
    """Plot summary of median amplitudes for all detectors.

    This creates a scatter plot showing the median amplitude for each detector,
    colored by grouping (readout column for Q-band, wafer for W/G-band) with
    group-level horizontal lines.

    Parameters
    ----------
    stats : dict
        Output from compute_detector_statistics()
    array_data : pandas.DataFrame
        Array data with 'col' (Q-band readout column) and/or 'wafer' columns for grouping
    figsize : tuple, optional
        Figure size (width, height). Default is (12, 7).
    color_by : str, optional
        How to group detectors: 'auto' (col for Q-band, wafer for W/G-band),
        'column' (uses 'col' field), or 'wafer'. Default is 'auto'.
    normalize_by_group : bool, optional
        If True, normalize each group's amplitudes by its median.
        Default is False.
    outlier_threshold : float, optional
        If provided, highlights detectors with |z-score| > threshold in red

    Returns
    -------
    fig : matplotlib.figure.Figure
        The generated figure
    outlier_uids : list
        List of detector UIDs flagged as outliers (if threshold provided)

    Examples
    --------
    >>> data = query_database(db_path, quality_threshold=0.8)
    >>> stats = compute_detector_statistics(data)
    >>> # Auto-detect grouping (uses 'col' for Q-band)
    >>> fig, outliers = plot_detector_amplitudes_summary(stats, array_data,
    ...                                                   color_by='auto',
    ...                                                   outlier_threshold=2.0)
    >>> # Force column grouping
    >>> fig, outliers = plot_detector_amplitudes_summary(stats, array_data,
    ...                                                   color_by='column',
    ...                                                   outlier_threshold=2.0)
    """
    if not stats:
        print("No statistics to plot")
        return None, []

    # Extract data
    uids = sorted(stats.keys())
    medians = np.array([stats[uid]['median_amplitude'] for uid in uids])
    stds = np.array([stats[uid]['std_amplitude'] for uid in uids])

    # Determine grouping column
    if array_data is None:
        group_col = None
        group_label = 'all'
    else:
        # Auto-detect grouping based on available columns
        if color_by == 'auto':
            # Prefer 'col' (readout column) for Q-band, 'wafer' for W/G-band
            if 'col' in array_data.columns:
                group_col = 'col'
                group_label = 'Column'
            elif 'wafer' in array_data.columns:
                group_col = 'wafer'
                group_label = 'Wafer'
            else:
                group_col = None
                group_label = 'all'
        elif color_by == 'column':
            # Try 'col' first (Q-band readout column), then 'column'
            if 'col' in array_data.columns:
                group_col = 'col'
            elif 'column' in array_data.columns:
                group_col = 'column'
            else:
                group_col = None
            group_label = 'Column'
        elif color_by == 'wafer':
            group_col = 'wafer' if 'wafer' in array_data.columns else None
            group_label = 'Wafer'
        else:
            group_col = None
            group_label = 'all'

    # Get group assignments
    groups = []
    has_grouping = group_col is not None

    if has_grouping:
        for uid in uids:
            if int(uid) in array_data.index:
                group_val = array_data[group_col][int(uid)]
                # Handle NaN values
                if isinstance(group_val, float) and np.isnan(group_val):
                    groups.append('unknown')
                else:
                    groups.append(str(group_val))
            else:
                groups.append('unknown')
    else:
        # No grouping available - group all together
        groups = ['all'] * len(uids)

    unique_groups = sorted(set(g for g in groups if g != 'unknown'))

    # Setup colors
    colors = plt.cm.tab10(np.linspace(0, 1, len(unique_groups)))
    group_colors = {grp: colors[i] for i, grp in enumerate(unique_groups)}

    fig, ax = plt.subplots(figsize=figsize)

    outlier_uids = []

    # Plot each group
    for group in unique_groups:
        group_mask = [g == group for g in groups]
        group_uids = np.array(uids)[group_mask]
        group_medians = medians[group_mask]
        group_stds = stds[group_mask]

        if len(group_medians) == 0:
            continue

        group_median = np.median(group_medians)
        group_std_global = np.std(group_medians)

        if normalize_by_group:
            plot_vals = group_medians / group_median
            plot_errs = group_stds / group_median
            target_line = 1.0
        else:
            plot_vals = group_medians
            plot_errs = group_stds
            target_line = group_median

        # Compute z-scores for outlier detection
        if group_std_global > 0:
            zscores = (group_medians - group_median) / group_std_global
        else:
            zscores = np.zeros_like(group_medians)

        # Determine point colors
        point_colors = []
        for i, z in enumerate(zscores):
            if outlier_threshold is not None and abs(z) > outlier_threshold:
                point_colors.append('red')
                outlier_uids.append(group_uids[i])
            else:
                point_colors.append(group_colors[group])

        # Plot points
        for i in range(len(group_uids)):
            # Format label based on group type
            if i == 0:
                if group_label == 'Column':
                    label = f'Col {group}'
                elif group_label == 'Wafer':
                    label = f'Wafer {group}'
                else:
                    label = f'{group_label} {group}'
            else:
                label = ''

            ax.errorbar([group_uids[i]], [plot_vals[i]], yerr=[plot_errs[i]],
                        fmt='o', color=point_colors[i], ecolor='gray', capsize=3,
                        label=label, zorder=1)

        # Plot group median line
        if len(group_uids) > 0:
            ax.hlines(target_line, min(group_uids), max(group_uids),
                      colors=group_colors[group], linestyles='--',
                      linewidth=2, alpha=0.7, zorder=2)

    ax.set_xlabel('Detector UID')
    ylabel = 'Median Amplitude'
    if normalize_by_group:
        ylabel = f'Normalized Median Amplitude (relative to {group_label.lower()})'
    ax.set_ylabel(ylabel)

    title = 'Detector Amplitude Summary'
    if outlier_threshold is not None:
        title += f' (red points: |z| > {outlier_threshold})'
    if not has_grouping:
        title += ' (no grouping available)'
    else:
        title += f' (by {group_label})'
    ax.set_title(title)

    ax.grid(True, alpha=0.3)
    if has_grouping and len(unique_groups) > 1:
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

    plt.tight_layout()

    return fig, sorted(outlier_uids)


def plot_amplitude_distribution(data, det_uid, array_data,
                                figsize=(10, 6), n_bins=50,
                                group_by_boresight=False,
                                quality_threshold=None):
    """Plot histogram/KDE of amplitude distribution for a single detector.

    Shows the distribution of template amplitudes for a specific detector,
    optionally separated by boresight angle.

    Parameters
    ----------
    data : dict
        Query results from query_database()
    det_uid : int
        Detector UID to plot
    array_data : pandas.DataFrame
        Array data for module info
    figsize : tuple, optional
        Figure size. Default is (10, 6).
    n_bins : int, optional
        Number of histogram bins. Default is 50.
    group_by_boresight : bool, optional
        If True, separate distributions by boresight angle. Default is False.
    quality_threshold : float, optional
        If provided, only include templates with quality >= threshold

    Returns
    -------
    fig : matplotlib.figure.Figure
        The generated figure
    ax : matplotlib.axes.Axes
        The axes object
    """
    # Filter by quality
    if quality_threshold is not None:
        mask = data['quality'] >= quality_threshold
        filtered_data = {k: v[mask] if isinstance(v, np.ndarray) else v
                         for k, v in data.items()}
    else:
        filtered_data = data

    # Get detector data
    det_mask = filtered_data['det_uids'] == det_uid
    det_amps = np.abs(filtered_data['amplitudes'][det_mask]
                      * filtered_data['scaling_factors'][det_mask])
    det_bs = filtered_data['boresights'][det_mask]

    if len(det_amps) == 0:
        print(f"No data found for detector {det_uid}")
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, f"No data for detector {det_uid}",
                ha='center', va='center', transform=ax.transAxes)
        return fig, ax

    fig, ax = plt.subplots(figsize=figsize)

    if not group_by_boresight or np.all(np.isnan(det_bs)):
        # Single distribution
        ax.hist(det_amps, bins=n_bins, alpha=0.6, density=True, label='All data')
        med = np.median(det_amps)
        ax.axvline(med, color='r', linestyle='--',
                   label=f'Median = {med:.4f}')
    else:
        # Separate by boresight
        unique_bs = np.unique(det_bs[~np.isnan(det_bs)])
        colors = plt.cm.viridis(np.linspace(0, 1, len(unique_bs)))

        for i, bs in enumerate(unique_bs):
            bs_mask = np.abs(det_bs - bs) < 0.1
            bs_amps = det_amps[bs_mask]

            if len(bs_amps) > 0:
                ax.hist(bs_amps, bins=n_bins, alpha=0.5, density=True,
                        color=colors[i], label=f'BS {int(bs)}°')
                med = np.median(bs_amps)
                ax.axvline(med, color=colors[i], linestyle='--', alpha=0.7)

    ax.set_xlabel('Amplitude × Scaling Factor')
    ax.set_ylabel('Density')

    # Get grouping info (if available)
    title = f'Amplitude Distribution: Detector {det_uid}'
    if array_data is not None and int(det_uid) in array_data.index:
        # Try col (readout column) first, then column, then wafer
        if 'col' in array_data.columns:
            col_val = array_data.col[int(det_uid)]
            if not (isinstance(col_val, float) and np.isnan(col_val)):
                title = f'Amplitude Distribution: Detector {det_uid} (Col {int(col_val)})'
        elif 'column' in array_data.columns:
            col_val = array_data.column[int(det_uid)]
            if not (isinstance(col_val, float) and np.isnan(col_val)):
                title = f'Amplitude Distribution: Detector {det_uid} (Col {int(col_val)})'
        elif 'wafer' in array_data.columns:
            wafer_val = array_data.wafer[int(det_uid)]
            if not (isinstance(wafer_val, float) and np.isnan(wafer_val)):
                title = f'Amplitude Distribution: Detector {det_uid} (Wafer {wafer_val})'

    if quality_threshold is not None:
        title += f'\n(quality ≥ {quality_threshold})'

    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    return fig, ax


def plot_quality_vs_time(data, figsize=(14, 6), time_range=None, quality_threshold=None,
                         bin_hours=1, show_histogram=True):
    """Plot template fit quality vs time.

    Shows how fit quality (R²) varies over time, useful for identifying periods
    of poor template fits that may correlate with amplitude variations.

    Parameters
    ----------
    data : dict
        Query results from query_database()
    figsize : tuple, optional
        Figure size (width, height). Default is (14, 6).
    time_range : tuple of datetime, optional
        Time range (start_date, end_date) to plot
    quality_threshold : float, optional
        Horizontal line to mark quality threshold
    bin_hours : float, optional
        Bin width in hours for computing median quality. Default is 1.
    show_histogram : bool, optional
        If True, shows quality histogram on right side. Default is True.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The generated figure
    """
    # Apply time range filter if requested
    if time_range is not None:
        start_ctime = time_range[0].timestamp()
        end_ctime = time_range[1].timestamp()
        time_mask = (data['timestamps'] >= start_ctime) & (data['timestamps'] <= end_ctime)
        filtered_data = {k: v[time_mask] if isinstance(v, np.ndarray) else v
                         for k, v in data.items()}
    else:
        filtered_data = data

    # Convert to dates
    dates = np.array([datetime.fromtimestamp(t) for t in filtered_data['timestamps']])
    quality = filtered_data['quality']

    # Create figure
    if show_histogram:
        fig = plt.figure(figsize=figsize)
        gs = GridSpec(1, 2, width_ratios=[3, 1], wspace=0.3)
        ax_time = fig.add_subplot(gs[0, 0])
        ax_hist = fig.add_subplot(gs[0, 1])
    else:
        fig, ax_time = plt.subplots(figsize=figsize)

    # Scatter plot
    ax_time.scatter(dates, quality, s=1, alpha=0.3, color='C0')

    # Bin and plot median
    if bin_hours > 0:
        bin_seconds = bin_hours * 3600
        time_bins = np.arange(filtered_data['timestamps'].min(),
                              filtered_data['timestamps'].max() + bin_seconds,
                              bin_seconds)

        bin_medians = []
        bin_centers = []

        for i in range(len(time_bins) - 1):
            mask = (filtered_data['timestamps'] >= time_bins[i]) & \
                   (filtered_data['timestamps'] < time_bins[i + 1])
            if np.sum(mask) > 0:
                bin_medians.append(np.median(quality[mask]))
                bin_centers.append((time_bins[i] + time_bins[i + 1]) / 2)

        if bin_centers:
            bin_dates = [datetime.fromtimestamp(t) for t in bin_centers]
            ax_time.plot(bin_dates, bin_medians, 'r-', linewidth=2,
                         label=f'{bin_hours}h median', zorder=10)
            ax_time.legend()

    # Quality threshold line
    if quality_threshold is not None:
        ax_time.axhline(quality_threshold, color='k', linestyle='--',
                        label=f'Threshold = {quality_threshold}')
        ax_time.legend()

    ax_time.set_ylabel('Template Quality (R²)')
    ax_time.grid(True, alpha=0.3)

    # Format x-axis
    time_span_days = (dates.max() - dates.min()).total_seconds() / 86400
    if time_span_days > 365:
        ax_time.xaxis.set_major_locator(YearLocator())
        ax_time.xaxis.set_major_formatter(DateFormatter('%Y'))
        ax_time.set_xlabel('Year')
    elif time_span_days > 60:
        ax_time.xaxis.set_major_locator(MonthLocator(interval=2))
        ax_time.xaxis.set_major_formatter(DateFormatter('%b %Y'))
        ax_time.set_xlabel('Date')
        plt.setp(ax_time.xaxis.get_majorticklabels(), rotation=45, ha='right')
    else:
        ax_time.xaxis.set_major_locator(MonthLocator())
        ax_time.xaxis.set_major_formatter(DateFormatter('%b %d'))
        ax_time.set_xlabel('Date')
        plt.setp(ax_time.xaxis.get_majorticklabels(), rotation=45, ha='right')

    # Histogram
    if show_histogram:
        ax_hist.hist(quality, bins=50, orientation='horizontal', alpha=0.6)
        ax_hist.set_xlabel('Count')
        ax_hist.set_ylabel('')
        ax_hist.yaxis.tick_right()
        ax_hist.grid(True, alpha=0.3, axis='x')
        ax_hist.set_ylim(ax_time.get_ylim())

    plt.suptitle('Template Fit Quality vs. Time')
    plt.tight_layout()

    return fig


def plot_common_mode_signal(common_mode_result, figsize=(14, 5), time_range=None):
    """Plot the common-mode signal.

    Parameters
    ----------
    common_mode_result : dict
        Output from compute_common_mode()
    figsize : tuple, optional
        Figure size (width, height). Default is (14, 5).
    time_range : tuple of datetime, optional
        Time range (start_date, end_date) to plot

    Returns
    -------
    fig : matplotlib.figure.Figure
        The generated figure
    """
    # Apply time range filter if requested
    if time_range is not None:
        start_ctime = time_range[0].timestamp()
        end_ctime = time_range[1].timestamp()
        time_mask = (common_mode_result['timestamps'] >= start_ctime) & \
            (common_mode_result['timestamps'] <= end_ctime)
        timestamps = common_mode_result['timestamps'][time_mask]
        common_mode = common_mode_result['common_mode'][time_mask]
    else:
        timestamps = common_mode_result['timestamps']
        common_mode = common_mode_result['common_mode']

    # Check if we have data
    if len(timestamps) == 0:
        print("Warning: No data to plot in time range")
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, 'No data in selected time range',
                ha='center', va='center', transform=ax.transAxes)
        return fig

    # Convert to dates
    dates = np.array([datetime.fromtimestamp(t) for t in timestamps])

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    # Plot common mode
    ax.plot(dates, common_mode, 'k-', linewidth=1, alpha=0.7)

    print(f"Plotting {len(dates)} points, CM range: [{np.min(common_mode):.6f}, {np.max(common_mode):.6f}]")
    ax.set_ylabel('Common-Mode Amplitude')
    ax.grid(True, alpha=0.3)

    # Format x-axis
    time_span_days = (dates.max() - dates.min()).total_seconds() / 86400
    if time_span_days > 365:
        ax.xaxis.set_major_locator(YearLocator())
        ax.xaxis.set_major_formatter(DateFormatter('%Y'))
        ax.set_xlabel('Year')
    elif time_span_days > 60:
        ax.xaxis.set_major_locator(MonthLocator(interval=2))
        ax.xaxis.set_major_formatter(DateFormatter('%b %Y'))
        ax.set_xlabel('Date')
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
    elif time_span_days > 7:
        ax.xaxis.set_major_locator(MonthLocator())
        ax.xaxis.set_major_formatter(DateFormatter('%b %d'))
        ax.set_xlabel('Date')
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
    else:
        from matplotlib.dates import DayLocator
        ax.xaxis.set_major_locator(DayLocator())
        ax.xaxis.set_major_formatter(DateFormatter('%b %d\n%H:%M'))
        ax.set_xlabel('Date and Time')
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

    # Title
    method_str = common_mode_result['method']
    if method_str == 'pca':
        var_explained = common_mode_result['variance_explained'][0] * 100
        title = f'Common-Mode Signal (PCA, {var_explained:.1f}% variance explained)'
    else:
        title = f'Common-Mode Signal ({method_str})'

    ax.set_title(title)
    plt.tight_layout()

    return fig


def plot_common_mode_comparison(data, det_uids, common_mode_result, figsize=(14, 10),
                                time_range=None, quality_threshold=None):
    """Plot detector amplitudes before and after common-mode removal.

    Shows side-by-side comparison of original and corrected amplitudes for
    selected detectors to demonstrate common-mode removal effectiveness.

    Parameters
    ----------
    data : dict
        Query results from query_database()
    det_uids : list of int
        List of detector UIDs to plot
    common_mode_result : dict
        Output from compute_common_mode()
    figsize : tuple, optional
        Figure size (width, height). Default is (14, 10).
    time_range : tuple of datetime, optional
        Time range (start_date, end_date) to plot
    quality_threshold : float, optional
        If provided, filters data to quality >= threshold

    Returns
    -------
    fig : matplotlib.figure.Figure
        The generated figure
    """
    # Apply quality filter if requested (filter in 2D)
    if quality_threshold is not None:
        quality_mask_2d = data['quality'] >= quality_threshold
        filtered_data = data.copy()
        filtered_data['amplitudes'] = np.where(quality_mask_2d, data['amplitudes'], np.nan)
        filtered_data['scaling_factors'] = np.where(quality_mask_2d, data['scaling_factors'], np.nan)
    else:
        filtered_data = data

    # Remove common mode
    corrected_data = remove_common_mode(filtered_data, common_mode_result)

    # Apply time range filter if requested
    if time_range is not None:
        start_ctime = time_range[0].timestamp()
        end_ctime = time_range[1].timestamp()
        time_mask = (filtered_data['timestamps'] >= start_ctime) & \
            (filtered_data['timestamps'] <= end_ctime)

        # Filter 2D arrays along time axis, 1D time arrays normally
        filtered_data = {
            **{k: filtered_data[k][:, time_mask] for k in ['amplitudes', 'amplitude_errors', 'phases',
                                                           'phase_errors', 'quality', 'scaling_factors',
                                                           'original_amplitudes']},
            **{k: filtered_data[k][time_mask] for k in ['timestamps', 'spans', 'span_paths',
                                                        'boresights', 'elevations', 'pwv',
                                                        'mod_freq', 'modulator_type', 'subdivision_index']
               if k in filtered_data},
            'det_uids': filtered_data['det_uids']
        }

        # List of 2D array keys that need time filtering
        array_2d_keys = ['amplitudes_cm_removed', 'amplitudes', 'scaling_factors',
                         'amplitude_errors', 'phases', 'phase_errors',
                         'quality', 'original_amplitudes']

        corrected_data = {
            **{k: (corrected_data[k][:, time_mask] if k in array_2d_keys
                   else corrected_data[k])
               for k in corrected_data.keys()
               if k != 'det_uids' and k != 'common_mode_signal'},
            'det_uids': corrected_data['det_uids'],
            'common_mode_signal': (corrected_data['common_mode_signal'][time_mask]
                                   if 'common_mode_signal' in corrected_data
                                   else None)
        }

        # Add back timestamp and other 1D time arrays
        for k in ['timestamps', 'spans', 'span_paths', 'boresights', 'elevations', 'pwv',
                  'mod_freq', 'modulator_type', 'subdivision_index']:
            if k in filtered_data:
                corrected_data[k] = filtered_data[k]

    # Create figure
    fig, axes = plt.subplots(len(det_uids), 2, figsize=figsize, sharex=True, sharey='row')
    if len(det_uids) == 1:
        axes = axes.reshape(1, -1)

    # Convert timestamps to datetime
    dates = [datetime.fromtimestamp(t) for t in filtered_data['timestamps']]

    for i, uid in enumerate(det_uids):
        # Get data for this detector (2D format)
        det_idx = np.where(filtered_data['det_uids'] == uid)[0]

        if len(det_idx) == 0:
            print(f"Warning: Detector {uid} not found in data")
            continue

        det_idx = det_idx[0]

        # Extract detector rows [ntime]
        uid_amps_orig = np.abs(filtered_data['amplitudes'][det_idx, :]
                               * filtered_data['scaling_factors'][det_idx, :])
        uid_amps_corr = np.abs(corrected_data['amplitudes_cm_removed'][det_idx, :]
                               * corrected_data['scaling_factors'][det_idx, :])

        # Filter out NaN values
        valid_mask = ~np.isnan(uid_amps_orig)
        uid_dates = [dates[j] for j in range(len(dates)) if valid_mask[j]]
        uid_amps_orig_valid = uid_amps_orig[valid_mask]
        uid_amps_corr_valid = uid_amps_corr[valid_mask]

        if len(uid_dates) == 0:
            continue

        # Original
        axes[i, 0].scatter(uid_dates, uid_amps_orig_valid, s=4, alpha=0.6, color='C0')
        axes[i, 0].set_ylabel(f'Det {uid}\nAmplitude')
        axes[i, 0].grid(True, alpha=0.3)
        if i == 0:
            axes[i, 0].set_title('Original')

        # Corrected
        axes[i, 1].scatter(uid_dates, uid_amps_corr_valid, s=4, alpha=0.6, color='C1')
        axes[i, 1].grid(True, alpha=0.3)
        if i == 0:
            axes[i, 1].set_title('Common-Mode Removed')

        # Compute RMS for comparison
        rms_orig = np.std(uid_amps_orig_valid)
        rms_corr = np.std(uid_amps_corr_valid)
        improvement = (1 - rms_corr / rms_orig) * 100

        # Add text box with stats
        axes[i, 1].text(0.98, 0.98, f'RMS reduction: {improvement:.1f}%',
                        transform=axes[i, 1].transAxes,
                        ha='right', va='top',
                        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # Format x-axis on bottom plots
    time_span_days = (uid_dates.max() - uid_dates.min()).total_seconds() / \
        86400 if len(uid_dates) > 0 else 365
    for ax in axes[-1, :]:
        if time_span_days > 365:
            ax.xaxis.set_major_locator(YearLocator())
            ax.xaxis.set_major_formatter(DateFormatter('%Y'))
        elif time_span_days > 60:
            ax.xaxis.set_major_locator(MonthLocator(interval=2))
            ax.xaxis.set_major_formatter(DateFormatter('%b %Y'))
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
        else:
            ax.xaxis.set_major_locator(MonthLocator())
            ax.xaxis.set_major_formatter(DateFormatter('%b %d'))
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

    plt.suptitle('Common-Mode Removal Comparison')
    plt.tight_layout()

    return fig
