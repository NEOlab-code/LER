import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks
import imageio.v2 as imageio


# Anchor-based height-threshold edge extraction.
# Anchors come from the row-averaged gradient profile (same idea as the ERF
# workflow). The edge position is then read off by linearly interpolating the
# AFM height profile at a local threshold - not from a binary pixel transition.
# The binary image at the end is only a LACERM-friendly representation.


# load xyz text into a z(x, y) grid
def load_xyz(path, skiprows=0, delimiter=None, usecols=(0, 1, 2)):
    data = np.loadtxt(path, skiprows=skiprows, delimiter=delimiter, usecols=usecols)
    if data.ndim != 2 or data.shape[1] < 3:
        raise ValueError("txt file must contain at least 3 columns (x, y, z).")

    x = data[:, 0]
    y = data[:, 1]
    z = data[:, 2]

    x_unique = np.unique(x)
    y_unique = np.unique(y)

    nx = len(x_unique)
    ny = len(y_unique)

    x_to_ix = {v: i for i, v in enumerate(x_unique)}
    y_to_iy = {v: i for i, v in enumerate(y_unique)}

    z_map = np.full((ny, nx), np.nan, dtype=float)
    for xi, yi, zi in zip(x, y, z):
        z_map[y_to_iy[yi], x_to_ix[xi]] = zi

    return x_unique, y_unique, z_map


def interp_nans(z):
    zz = z.copy()
    idx = np.arange(len(zz))
    m = np.isfinite(zz)
    if m.sum() < 2:
        return zz
    zz[~m] = np.interp(idx[~m], idx[m], zz[m])
    return zz


# anchor detection: gradient peaks of the row-averaged profile (same as ERF)
def merge_anchors(anchors, min_sep_um):
    if anchors is None or len(anchors) == 0:
        return np.asarray(anchors)

    a = np.sort(np.asarray(anchors, float))
    kept = []
    group = [a[0]]

    for v in a[1:]:
        if v - group[-1] < min_sep_um:
            group.append(v)
        else:
            kept.append(float(np.median(group)))
            group = [v]

    kept.append(float(np.median(group)))
    return np.array(kept, float)


def find_anchors(
    x,
    z_map,
    smooth_sigma=3.0,
    min_distance_px=120,
    prominence=None,
    merge_min_sep_um=None,
):
    z_mean = np.nanmean(z_map, axis=0)
    z_mean = gaussian_filter1d(z_mean, sigma=smooth_sigma)
    dzdx = np.gradient(z_mean, x)

    if prominence is None:
        p = np.nanpercentile(np.abs(dzdx), 90)
        prominence = max(p * 0.2, 0.0)

    rise_idx, _ = find_peaks(dzdx, distance=min_distance_px, prominence=prominence)
    fall_idx, _ = find_peaks(-dzdx, distance=min_distance_px, prominence=prominence)

    anchors_rise = np.sort(x[rise_idx])
    anchors_fall = np.sort(x[fall_idx])

    if merge_min_sep_um is not None and merge_min_sep_um > 0:
        anchors_rise = merge_anchors(anchors_rise, merge_min_sep_um)
        anchors_fall = merge_anchors(anchors_fall, merge_min_sep_um)

    return anchors_rise, anchors_fall, prominence


# local plateau levels around an edge -> threshold height
def local_plateaus(
    x,
    z_row,
    x_ref,
    mode="rising",
    window_half_width=0.04,
    exclude_half_width=0.010,
    min_points_each_side=5,
):
    """
    Local lower/upper plateau levels around one edge.

    x_ref (anchor or tracked position) only sets the local ROI, it is not the
    edge itself. For a rising edge the left side is the lower plateau and the
    right side the upper one; for a falling edge it is the other way round.
    Each level is the median of its side after dropping the central transition.
    """
    in_win = (x >= x_ref - window_half_width) & (x <= x_ref + window_half_width)
    left = in_win & (x <= x_ref - exclude_half_width)
    right = in_win & (x >= x_ref + exclude_half_width)

    if np.sum(left) < min_points_each_side or np.sum(right) < min_points_each_side:
        return None

    z_left = z_row[left]
    z_right = z_row[right]
    z_left = z_left[np.isfinite(z_left)]
    z_right = z_right[np.isfinite(z_right)]

    if z_left.size < min_points_each_side or z_right.size < min_points_each_side:
        return None

    left_level = float(np.nanmedian(z_left))
    right_level = float(np.nanmedian(z_right))

    if mode == "rising":
        z_low = left_level
        z_high = right_level
    elif mode == "falling":
        z_low = right_level
        z_high = left_level
    else:
        raise ValueError("mode must be 'rising' or 'falling'")

    # skip if the local contrast is too small to be a real edge
    if not np.isfinite(z_low) or not np.isfinite(z_high) or np.isclose(z_low, z_high):
        return None

    return z_low, z_high


def find_crossing(
    x,
    z_row,
    h_thr,
    x_ref,
    search_half_width=0.020,
    min_slope_abs=1e-12,
):
    """
    Threshold crossing by linear interpolation between the two neighboring
    samples that straddle h_thr. With several crossings in the window, pick the
    one nearest to x_ref.
    """
    in_search = (x >= x_ref - search_half_width) & (x <= x_ref + search_half_width)
    idx = np.where(in_search)[0]

    if idx.size < 2:
        return np.nan

    crossings = []

    for i0, i1 in zip(idx[:-1], idx[1:]):
        z0 = z_row[i0]
        z1 = z_row[i1]
        if not (np.isfinite(z0) and np.isfinite(z1)):
            continue

        d0 = z0 - h_thr
        d1 = z1 - h_thr

        if d0 == 0:
            crossings.append(float(x[i0]))
            continue

        if d0 * d1 < 0:
            dz = z1 - z0
            if abs(dz) < min_slope_abs:
                continue
            frac = (h_thr - z0) / dz
            x_cross = x[i0] + frac * (x[i1] - x[i0])
            crossings.append(float(x_cross))

    if len(crossings) == 0:
        return np.nan

    crossings = np.array(crossings, dtype=float)
    return float(crossings[np.argmin(np.abs(crossings - x_ref))])


# per-row edge contours by height threshold
def extract_edges(
    x,
    y,
    z_map,
    anchors,
    mode="rising",
    threshold_ratio=0.5,
    # anchor/ROI parameters
    window_half_width_base=0.04,
    track_half_width=0.020,
    search_half_width=0.020,
    plateau_exclude_half_width=0.010,
    max_jump=0.04,
    # optional cross-line smoothing for numerical stability
    # Set to 0.0 to use the measured row directly for the threshold crossing.
    crossing_smooth_sigma=0.0,
):
    """
    Returns:
        x_edge_map:  shape (n_edges, ny), sub-pixel threshold edge coordinates in um
        zlow_map:    local lower plateau level used for threshold
        zhigh_map:   local upper plateau level used for threshold
        hthr_map:    local threshold height
    """
    ny, nx = z_map.shape
    n_edges = len(anchors)

    x_edge_map = np.full((n_edges, ny), np.nan, dtype=float)
    zlow_map = np.full((n_edges, ny), np.nan, dtype=float)
    zhigh_map = np.full((n_edges, ny), np.nan, dtype=float)
    hthr_map = np.full((n_edges, ny), np.nan, dtype=float)

    prev_x = anchors.astype(float).copy()

    for iy in range(ny):
        z = interp_nans(z_map[iy, :])

        if crossing_smooth_sigma and crossing_smooth_sigma > 0:
            z_for_crossing = gaussian_filter1d(z, sigma=crossing_smooth_sigma)
        else:
            z_for_crossing = z

        for k, anchor in enumerate(anchors):
            x_ref = prev_x[k] if np.isfinite(prev_x[k]) else float(anchor)

            plateaus = local_plateaus(
                x=x,
                z_row=z_for_crossing,
                x_ref=x_ref,
                mode=mode,
                window_half_width=window_half_width_base,
                exclude_half_width=plateau_exclude_half_width,
            )

            if plateaus is None:
                continue

            z_low, z_high = plateaus
            h_thr = z_low + threshold_ratio * (z_high - z_low)

            x_edge = find_crossing(
                x=x,
                z_row=z_for_crossing,
                h_thr=h_thr,
                x_ref=x_ref,
                search_half_width=search_half_width,
            )

            if not np.isfinite(x_edge):
                continue

            if np.isfinite(prev_x[k]) and abs(x_edge - prev_x[k]) > max_jump:
                continue

            x_edge_map[k, iy] = x_edge
            zlow_map[k, iy] = z_low
            zhigh_map[k, iy] = z_high
            hthr_map[k, iy] = h_thr
            prev_x[k] = x_edge

    return x_edge_map, zlow_map, zhigh_map, hthr_map


def fill_gaps(contours):
    if contours is None or contours.size == 0:
        return contours
    out = contours.copy()
    ny = out.shape[1]
    idx = np.arange(ny)

    for k in range(out.shape[0]):
        xs = out[k]
        m = np.isfinite(xs)
        if m.sum() < 2:
            continue
        out[k, ~m] = np.interp(idx[~m], idx[m], xs[m])

    return out


# binary image from the sub-pixel contours
def to_binary(x, y, contours_rise, contours_fall, start_black=True):
    """Render the extracted contours as a binary image (for visualization / software input)."""
    nx = len(x)
    ny = len(y)
    dx = x[1] - x[0]

    def xpos_to_ix(xpos):
        return int(np.clip(np.round((xpos - x[0]) / dx), 0, nx - 1))

    bin_img = np.zeros((ny, nx), dtype=np.uint8)

    for iy in range(ny):
        events = []

        if contours_rise is not None and contours_rise.size > 0:
            xs = contours_rise[:, iy]
            events.extend(xs[np.isfinite(xs)].tolist())

        if contours_fall is not None and contours_fall.size > 0:
            xs = contours_fall[:, iy]
            events.extend(xs[np.isfinite(xs)].tolist())

        if len(events) == 0:
            bin_img[iy, :] = 0 if start_black else 255
            continue

        events.sort()
        state = 0 if start_black else 1
        last_ix = 0

        for xpos in events:
            ix = xpos_to_ix(xpos)
            if ix > last_ix:
                bin_img[iy, last_ix:ix] = 255 if state else 0
            state = 1 - state
            last_ix = ix

        bin_img[iy, last_ix:] = 255 if state else 0

    return bin_img


# overlay plot and a rough CD
def plot_overlay(x, y, z_map, x_rise, x_fall, title="Anchor-based height-threshold contours"):
    plt.figure(figsize=(7, 6))
    plt.imshow(z_map, extent=[x.min(), x.max(), y.max(), y.min()], cmap="gray", aspect="auto")

    for contours in (x_rise, x_fall):
        if contours is None or contours.size == 0:
            continue
        for k in range(contours.shape[0]):
            xs = contours[k]
            m = np.isfinite(xs)
            if m.any():
                plt.plot(xs[m], y[m], "-", color="red", linewidth=1.5)

    plt.xlabel("x (um)")
    plt.ylabel("y (um)")
    plt.title(title)
    plt.tight_layout()
    plt.show()


def compute_cd(x_rise, x_fall):
    """
    Spacing between adjacent rise/fall contours, as a rough CD.
    Line/space polarity depends on the image, so check against the overlay
    before trusting this number.
    """
    if x_rise is None or x_fall is None or x_rise.size == 0 or x_fall.size == 0:
        return np.array([])

    contours = []
    for C, label in [(x_rise, "rise"), (x_fall, "fall")]:
        for k in range(C.shape[0]):
            xs = C[k]
            if np.isfinite(xs).sum() > 2:
                contours.append((float(np.nanmean(xs)), label, xs))

    contours.sort(key=lambda v: v[0])
    cds = []

    for (_, label0, xs0), (_, label1, xs1) in zip(contours[:-1], contours[1:]):
        if label0 == label1:
            continue
        m = np.isfinite(xs0) & np.isfinite(xs1)
        if m.sum() < 2:
            continue
        widths_um = np.abs(xs1[m] - xs0[m])
        cds.extend(widths_um * 1000.0)  # um -> nm

    return np.array(cds, dtype=float)


# process a single file
def process_one_file(
    txt_path,
    threshold_ratio=0.50,
    skiprows=7,
    delimiter=None,
    save_binary=True,
    start_black=True,
    show_plot=True,
    # Anchor detection
    mean_smooth_sigma=3.0,
    min_distance_um=0.05,
    merge_min_sep_um=0.01,
    # Contour extraction
    window_half_width_base=0.04,
    search_half_width=0.020,
    plateau_exclude_half_width=0.010,
    max_jump=0.04,
    crossing_smooth_sigma=0.0,
):
    import os
    txt_path = str(txt_path)
    stem = os.path.splitext(os.path.basename(txt_path))[0]
    out_dir = os.path.dirname(txt_path)

    out_bin    = os.path.join(out_dir, f"{stem}_thr{threshold_ratio:.2f}.png")
    out_coords = os.path.join(out_dir, f"{stem}_thr{threshold_ratio:.2f}_coords.npz")

    print(f"\n{'='*60}")
    print(f"[FILE] {os.path.basename(txt_path)}")
    print(f"[THRESHOLD] {threshold_ratio:.2f}")

    x, y, z_map = load_xyz(txt_path, skiprows=skiprows, delimiter=delimiter)
    nx, ny = len(x), len(y)
    dx = float(x[1] - x[0])
    dy = float(y[1] - y[0]) if ny > 1 else float("nan")
    print(f"nx={nx}, ny={ny}, dx~{dx*1000:.2f} nm/px, dy~{dy*1000:.2f} nm/px")

    min_distance_px = max(1, int(np.round(min_distance_um / dx)))
    print(f"[AUTO] MIN_DISTANCE: {min_distance_um:.3f} um -> {min_distance_px} px", flush=True)

    anchors_rise, anchors_fall, _ = find_anchors(
        x, z_map,
        smooth_sigma=mean_smooth_sigma,
        min_distance_px=min_distance_px,
        prominence=None,
        merge_min_sep_um=merge_min_sep_um,
    )
    print(f"anchors(after merge): rising={len(anchors_rise)}, falling={len(anchors_fall)}")

    shared_kw = dict(
        threshold_ratio=threshold_ratio,
        window_half_width_base=window_half_width_base,
        search_half_width=search_half_width,
        plateau_exclude_half_width=plateau_exclude_half_width,
        max_jump=max_jump,
        crossing_smooth_sigma=crossing_smooth_sigma,
    )

    x_thr_rise, zlo_rise, zhi_rise, hthr_rise = extract_edges(
        x, y, z_map, anchors=anchors_rise, mode="rising", **shared_kw
    )
    x_thr_fall, zlo_fall, zhi_fall, hthr_fall = extract_edges(
        x, y, z_map, anchors=anchors_fall, mode="falling", **shared_kw
    )

    print(f"[DEBUG] valid rise={np.isfinite(x_thr_rise).sum()}/{x_thr_rise.size}  "
          f"fall={np.isfinite(x_thr_fall).sum()}/{x_thr_fall.size}")

    x_thr_rise_f = fill_gaps(x_thr_rise)
    x_thr_fall_f = fill_gaps(x_thr_fall)

    if show_plot:
        plot_overlay(x, y, z_map, x_thr_rise_f, x_thr_fall_f,
                     title=f"{stem}  t={threshold_ratio:.2f}")

    np.savez(
        out_coords,
        x=x, y=y,
        anchors_rise=anchors_rise, anchors_fall=anchors_fall,
        x_threshold_rise=x_thr_rise, x_threshold_fall=x_thr_fall,
        x_threshold_rise_filled=x_thr_rise_f, x_threshold_fall_filled=x_thr_fall_f,
        zlo_rise=zlo_rise, zhi_rise=zhi_rise, hthr_rise=hthr_rise,
        zlo_fall=zlo_fall, zhi_fall=zhi_fall, hthr_fall=hthr_fall,
        threshold_ratio=threshold_ratio,
    )
    print(f"[Saved] {out_coords}")

    cd_nm = compute_cd(x_thr_rise_f, x_thr_fall_f)
    if cd_nm.size > 0:
        print(f"CD: N={cd_nm.size}  mean={np.mean(cd_nm):.2f} nm  "
              f"3σ={3*np.std(cd_nm):.2f} nm  min/max={np.min(cd_nm):.2f}/{np.max(cd_nm):.2f} nm")

    if save_binary:
        bin_img = to_binary(x, y, x_thr_rise_f, x_thr_fall_f, start_black=start_black)
        imageio.imwrite(out_bin, bin_img)
        print(f"[Saved] {out_bin}")

    return cd_nm


# batch mode over a folder
if __name__ == "__main__":
    import argparse
    import os
    import glob

    parser = argparse.ArgumentParser(description="Anchor-based height-threshold edge extraction")
    parser.add_argument("folder",         type=str,   help="folder path (process all *.txt files)")
    parser.add_argument("--threshold",    type=float, default=0.50, help="height threshold ratio (0-1, default 0.50)")
    parser.add_argument("--skiprows",     type=int,   default=7,    help="header rows to skip in txt (default 7)")
    parser.add_argument("--no-binary",    action="store_true",      help="do not save PNG")
    parser.add_argument("--no-plot",      action="store_true",      help="do not show overlay plot")
    parser.add_argument("--start-white",  action="store_true",      help="start binary image with white")
    args = parser.parse_args()

    txt_files = sorted(glob.glob(os.path.join(args.folder, "*.txt")))
    if not txt_files:
        print(f"[ERROR] no .txt files found in '{args.folder}'.")
        raise SystemExit(1)

    print(f"[BATCH] folder: {args.folder}")
    print(f"[BATCH] threshold={args.threshold:.2f}  file count={len(txt_files)}")

    all_cd = []
    for path in txt_files:
        cd = process_one_file(
            txt_path=path,
            threshold_ratio=args.threshold,
            skiprows=args.skiprows,
            save_binary=not args.no_binary,
            start_black=not args.start_white,
            show_plot=not args.no_plot,
        )
        all_cd.append(cd)

    combined = np.concatenate([c for c in all_cd if c.size > 0]) if all_cd else np.array([])
    if combined.size > 0:
        print(f"\n{'='*60}")
        print(f"[BATCH SUMMARY] all CD (threshold={args.threshold:.2f})")
        print(f"  file count    : {len(txt_files)}")
        print(f"  total points  : {combined.size}")
        print(f"  mean CD (nm)  : {np.mean(combined):.2f}")
        print(f"  CD 3σ  (nm)   : {3*np.std(combined):.2f}")
        print(f"  min/max (nm)  : {np.min(combined):.2f} / {np.max(combined):.2f}")
