import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks, savgol_filter
from scipy.optimize import curve_fit
from scipy.special import erf
import imageio.v2 as imageio


# -----------------------------
# 1) Load
# -----------------------------
def load_xyz_txt(path, skiprows=0, delimiter=None, usecols=(0, 1, 2)):
    data = np.loadtxt(path, skiprows=skiprows, delimiter=delimiter, usecols=usecols)
    if data.ndim != 2 or data.shape[1] < 3:
        raise ValueError("txt 파일은 최소 3열(x,y,z)을 포함해야 합니다.")

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


def fill_nan_1d_linear(z):
    zz = z.copy()
    idx = np.arange(len(zz))
    m = np.isfinite(zz)
    if m.sum() < 2:
        return zz
    zz[~m] = np.interp(idx[~m], idx[m], zz[m])
    return zz


# -----------------------------
# 2) Anchors + merge close peaks
# -----------------------------
def merge_close_anchors(anchors, min_sep_um):
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


def find_edge_anchors_both(
    x, z_map,
    smooth_sigma=3.0,
    min_distance_px=120,
    prominence=None,
    merge_min_sep_um=None
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
        anchors_rise = merge_close_anchors(anchors_rise, merge_min_sep_um)
        anchors_fall = merge_close_anchors(anchors_fall, merge_min_sep_um)

    return anchors_rise, anchors_fall, prominence


# -----------------------------
# 3) Edge model (ERF)
# -----------------------------
def erf_edge(x, zlo, zhi, x0, sigma):
    """
    z(x)=zlo + (zhi-zlo)/2 * [1 + erf((x-x0)/(sqrt(2)*sigma))]
    sigma>0
    """
    return zlo + 0.5 * (zhi - zlo) * (1.0 + erf((x - x0) / (np.sqrt(2.0) * sigma)))


def fit_edge_x0_erf(x_seg, z_seg, x0_init, sigma_init, zlo_init, zhi_init):
    bounds_lower = [np.nanmin(z_seg) - 1e9, np.nanmin(z_seg) - 1e9, x_seg.min(), 1e-6]
    bounds_upper = [np.nanmax(z_seg) + 1e9, np.nanmax(z_seg) + 1e9, x_seg.max(), (x_seg.max() - x_seg.min())]
    p0 = [zlo_init, zhi_init, x0_init, sigma_init]

    popt, _ = curve_fit(
        erf_edge, x_seg, z_seg,
        p0=p0,
        bounds=(bounds_lower, bounds_upper),
        maxfev=12000
    )
    zlo, zhi, x0, sigma = popt
    return float(x0), float(sigma), float(zlo), float(zhi)


# -----------------------------
# 4) x0 안정화 도구: Hampel + Savitzky-Golay
# -----------------------------
def hampel_filter_1d(x, k=7, t0=3.0):
    x = x.copy()
    n = len(x)
    for i in range(n):
        i0 = max(0, i - k)
        i1 = min(n, i + k + 1)
        w = x[i0:i1]
        m = np.isfinite(w)
        if m.sum() < 5:
            continue
        wv = w[m]
        med = np.median(wv)
        mad = np.median(np.abs(wv - med)) + 1e-12
        if np.isfinite(x[i]):
            if np.abs(x[i] - med) > t0 * 1.4826 * mad:
                x[i] = med
    return x


def smooth_contour_1d(x, win=31, poly=2):

    xx = x.copy()
    idx = np.arange(len(xx))
    m = np.isfinite(xx)
    if m.sum() < 10:
        return xx
    xx[~m] = np.interp(idx[~m], idx[m], xx[m])

    # win 조정(데이터 길이에 맞춤)
    win = int(win)
    if win % 2 == 0:
        win += 1
    win = min(win, len(xx) - (1 - len(xx) % 2))  # 길이보다 작고 홀수
    if win < 5:
        return xx

    return savgol_filter(xx, window_length=win, polyorder=poly, mode="interp")


# -----------------------------
# 5) Per-row erf fit (sigma-aware adaptive ROI + tracking + retry)
# -----------------------------
def extract_contours_erf_fit_sigma_aware(
    x, y, z_map,
    anchors,
    mode="rising",
    smooth_sigma=1.0,
    # base ROI (최소값)
    window_half_width_base=0.05,
    track_half_width_base=0.02,
    # sigma 기반 확장 계수 (핵심)
    k_window=6.0,      # window_half_width >= k_window * sigma_prev
    k_track=3.0,       # track_half_width  >= k_track  * sigma_prev
    # 나머지
    max_jump=0.04,
    sigma_init=0.003,
    retry_with_anchor_window=True,
    # after-fit stabilize
    do_hampel=True,
    hampel_k=7,
    hampel_t0=3.0,
    do_savgol=True,
    savgol_win=31,
    savgol_poly=2
):
    ny, nx = z_map.shape
    n_edges = len(anchors)

    x0_map    = np.full((n_edges, ny), np.nan, float)
    sigma_map = np.full((n_edges, ny), np.nan, float)
    zlo_map   = np.full((n_edges, ny), np.nan, float)
    zhi_map   = np.full((n_edges, ny), np.nan, float)

    prev_x0 = anchors.astype(float).copy()
    prev_sig = np.full(n_edges, float(sigma_init))

    for iy in range(ny):
        z = fill_nan_1d_linear(z_map[iy, :])
        z_s = gaussian_filter1d(z, sigma=smooth_sigma)

        for k, a in enumerate(anchors):
            # sigma-aware ROI sizes
            sig_ref = prev_sig[k] if np.isfinite(prev_sig[k]) else sigma_init
            win_hw = max(window_half_width_base, k_window * sig_ref)
            trk_hw = max(track_half_width_base,  k_track  * sig_ref)

            xmin0 = a - win_hw
            xmax0 = a + win_hw

            if np.isfinite(prev_x0[k]):
                xmin_t = max(xmin0, prev_x0[k] - trk_hw)
                xmax_t = min(xmax0, prev_x0[k] + trk_hw)
            else:
                xmin_t, xmax_t = xmin0, xmax0

            def try_fit(xmin, xmax):
                i0 = np.searchsorted(x, xmin, side="left")
                i1 = np.searchsorted(x, xmax, side="right")
                i0 = max(0, i0)
                i1 = min(nx, i1)
                if i1 - i0 < 25:
                    return None

                x_seg = x[i0:i1]
                z_seg = z_s[i0:i1]

                if mode == "rising":
                    zlo_init = np.nanpercentile(z_seg, 10)
                    zhi_init = np.nanpercentile(z_seg, 90)
                else:
                    zlo_init = np.nanpercentile(z_seg, 90)
                    zhi_init = np.nanpercentile(z_seg, 10)

                x0_init = prev_x0[k] if np.isfinite(prev_x0[k]) else a
                sig_init = prev_sig[k] if np.isfinite(prev_sig[k]) else sigma_init

                try:
                    x0, sig, zlo, zhi = fit_edge_x0_erf(x_seg, z_seg, x0_init, sig_init, zlo_init, zhi_init)
                except Exception:
                    return None

                if np.isfinite(prev_x0[k]) and abs(x0 - prev_x0[k]) > max_jump:
                    return None

                return (x0, sig, zlo, zhi)

            res = try_fit(xmin_t, xmax_t)
            if res is None and retry_with_anchor_window:
                res = try_fit(xmin0, xmax0)

            if res is None:
                continue

            x0, sig, zlo, zhi = res
            x0_map[k, iy] = x0
            sigma_map[k, iy] = sig
            zlo_map[k, iy] = zlo
            zhi_map[k, iy] = zhi
            prev_x0[k] = x0
            prev_sig[k] = sig

    for k in range(n_edges):
        xs = x0_map[k, :]
        if np.isfinite(xs).sum() < 20:
            continue
        if do_hampel:
            xs = hampel_filter_1d(xs, k=hampel_k, t0=hampel_t0)
        if do_savgol:
            xs = smooth_contour_1d(xs, win=savgol_win, poly=savgol_poly)
        x0_map[k, :] = xs

    return x0_map, sigma_map, zlo_map, zhi_map


# -----------------------------
# 6) Interpolate contours (for overlay/binary)
# -----------------------------
def interpolate_contours_linear(contours):
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


# -----------------------------
# 7) Overlay plot (contours on height map)
# -----------------------------
def plot_overlay(x, y, z_map, x0_rise, x0_fall, title="ERF fit edge contours (sigma-aware)", contour_lw=1.8):
    plt.figure(figsize=(7, 6))
    plt.imshow(z_map, extent=[x.min(), x.max(), y.max(), y.min()], cmap="gray", aspect="auto")

    def draw(C):
        if C is None or C.size == 0:
            return
        for k in range(C.shape[0]):
            xs = C[k]
            m = np.isfinite(xs)
            if m.any():
                plt.plot(xs[m], y[m], "-", color="red", linewidth=contour_lw)

    draw(x0_rise)
    draw(x0_fall)
    plt.xlabel("x (um)")
    plt.ylabel("y (um)")
    plt.title(title)
    plt.tight_layout()
    plt.show()


# -----------------------------
# 8) Choose a y-row robustly
# -----------------------------
def choose_profile_row(y, profile_row_list=None, profile_y_list_um=None):
    ny = len(y)
    rows = []

    if profile_row_list is not None:
        if isinstance(profile_row_list, (int, np.integer)):
            profile_row_list = [int(profile_row_list)]
        for r in profile_row_list:
            r = int(r)
            r = max(0, min(ny - 1, r))
            rows.append(r)

    if profile_y_list_um is not None:
        for y_um in profile_y_list_um:
            rows.append(int(np.argmin(np.abs(y - y_um))))

    if len(rows) == 0:
        rows = [ny // 2]

    rows = sorted(set(rows))
    return rows[0]


# -----------------------------
# 9) Print sigma stats (whole pattern)
# -----------------------------
def print_sigma_stats(label, sigma_map):
    s = sigma_map[np.isfinite(sigma_map)]
    if s.size == 0:
        print(f"[{label}] sigma: no valid fits")
        return None
    s_nm = s * 1000.0
    p10, p50, p90 = np.percentile(s_nm, [10, 50, 90])
    mean = float(np.mean(s_nm))
    std = float(np.std(s_nm))
    print(f"[{label}] sigma(nm)  N={s_nm.size}")
    print(f"  mean={mean:.3f}, std={std:.3f}, p10={p10:.3f}, median={p50:.3f}, p90={p90:.3f}")
    print(f"  10-90% transition width ≈ 2.563*sigma  (median 기준 ≈ {2.563*p50:.3f} nm)")
    return {"mean_nm": mean, "std_nm": std, "p10_nm": p10, "median_nm": p50, "p90_nm": p90}


# -----------------------------
# 10) Equations + representative (for one row)
# -----------------------------
def _collect_edges_for_row(iy, anchors, x0, sig, zlo, zhi, kind):
    out = []
    if anchors is None or len(anchors) == 0:
        return out
    n = len(anchors)
    for k in range(n):
        if not (np.isfinite(x0[k, iy]) and np.isfinite(sig[k, iy]) and np.isfinite(zlo[k, iy]) and np.isfinite(zhi[k, iy])):
            continue
        out.append({
            "kind": kind,
            "k": k,
            "anchor": float(anchors[k]),
            "x0": float(x0[k, iy]),
            "sigma": float(sig[k, iy]),
            "zlo": float(zlo[k, iy]),
            "zhi": float(zhi[k, iy]),
        })
    return out


def print_erf_equations_for_row(
    iy, y_um,
    anchors_rise, x0_rise, sig_rise, zlo_rise, zhi_rise,
    anchors_fall, x0_fall, sig_fall, zlo_fall, zhi_fall,
    max_edges_to_print=30,
    decimals=6
):
    edges = []
    edges += _collect_edges_for_row(iy, anchors_rise, x0_rise, sig_rise, zlo_rise, zhi_rise, "rising")
    edges += _collect_edges_for_row(iy, anchors_fall, x0_fall, sig_fall, zlo_fall, zhi_fall, "falling")
    edges.sort(key=lambda d: d["x0"])

    print(f"\n[ERF equations] y-row iy={iy} (y={y_um:.6f} um)")
    if len(edges) == 0:
        print("  (No valid fitted edges on this row)")
        return None

    edge_count = 0
    for e in edges:
        if edge_count >= max_edges_to_print:
            break
        edge_count += 1
        sig_nm = e["sigma"] * 1000.0
        print(f"#{edge_count:02d} {e['kind']:7s} anchor={e['anchor']:.{decimals}f} um | x0={e['x0']:.{decimals}f} um | sigma={sig_nm:.3f} nm")
        print(f"    z(x) = {e['zlo']:.4f} + 0.5*({e['zhi']:.4f}-{e['zlo']:.4f})*(1 + erf((x-{e['x0']:.{decimals}f})/(sqrt(2)*{e['sigma']:.{decimals}f})))")

    sigs = np.array([e["sigma"] for e in edges], float)
    sig_med = np.nanmedian(sigs)
    rep = min(edges, key=lambda e: abs(e["sigma"] - sig_med))

    print("\n[Representative ERF (one)]  (closest to median sigma on this row)")
    print(f"  kind={rep['kind']}, anchor={rep['anchor']:.{decimals}f} um, x0={rep['x0']:.{decimals}f} um, sigma={rep['sigma']*1000:.3f} nm")
    print(f"  z(x) = {rep['zlo']:.4f} + 0.5*({rep['zhi']:.4f}-{rep['zlo']:.4f})*(1 + erf((x-{rep['x0']:.{decimals}f})/(sqrt(2)*{rep['sigma']:.{decimals}f})))")

    return rep


# -----------------------------
# 11) Profile plot (black) + fits (gray dashed)
# -----------------------------
def plot_profile_with_erf_fits(
    x, y, z_map,
    iy,
    anchors_rise, x0_rise, sig_rise, zlo_rise, zhi_rise,
    anchors_fall, x0_fall, sig_fall, zlo_fall, zhi_fall,
    max_edges_to_plot=10,
    view_smooth_sigma=1.0,
    lw_profile=1.6,
    lw_fit=1.2
):
    z = fill_nan_1d_linear(z_map[iy, :])
    z_view = gaussian_filter1d(z, sigma=view_smooth_sigma)

    edges = []
    edges += _collect_edges_for_row(iy, anchors_rise, x0_rise, sig_rise, zlo_rise, zhi_rise, "rising")
    edges += _collect_edges_for_row(iy, anchors_fall, x0_fall, sig_fall, zlo_fall, zhi_fall, "falling")
    edges.sort(key=lambda d: d["x0"])
    edges = edges[:max_edges_to_plot]

    plt.figure(figsize=(10, 4))
    plt.plot(x, z_view, color="black", linewidth=lw_profile, label="AFM profile")

    for e in edges:
        z_fit = erf_edge(x, e["zlo"], e["zhi"], e["x0"], e["sigma"])
        plt.plot(x, z_fit, color="0.5", linestyle="--", linewidth=lw_fit)

    plt.xlabel("x (um)")
    plt.ylabel("Height (a.u.)")
    plt.yticks([])
    plt.title(f"AFM profile (black) + ERF fits (gray dashed) | iy={iy}, y={y[iy]:.6f} um")
    plt.tight_layout()
    plt.show()


# -----------------------------
# 12) Contours -> binary + PNG save
# -----------------------------
def contours_to_binary(x, y, contours_rise, contours_fall, start_black=True):
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
            xs = xs[np.isfinite(xs)]
            events.extend(xs.tolist())
        if contours_fall is not None and contours_fall.size > 0:
            xs = contours_fall[:, iy]
            xs = xs[np.isfinite(xs)]
            events.extend(xs.tolist())

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


# -----------------------------
# 13) Run
# -----------------------------
if __name__ == "__main__":
    # ===== 입출력 =====
    TXT_PATH = r"input path.txt"
    SKIPROWS = 7
    DELIMITER = None

    PROFILE_ROW_LIST = None
    PROFILE_Y_LIST_UM = None

    # anchor params
    MEAN_SMOOTH_SIGMA = 3.0
    MIN_DISTANCE_UM   = 0.15
    MERGE_MIN_SEP_UM  = 0.06

    # erf fit smoothing (per-row)
    LINE_SMOOTH_SIGMA = 1.2

    # sigma-aware ROI (parameter 건들지 말것)
    WINDOW_HALF_WIDTH_BASE = 0.04   # um (최소)
    TRACK_HALF_WIDTH_BASE  = 0.015  # um (최소)
    K_WINDOW = 6.0                  # win >= K_WINDOW*sigma_prev
    K_TRACK  = 3.0                  # trk >= K_TRACK*sigma_prev

    MAX_JUMP    = 0.04
    SIGMA_INIT  = 0.003   # um (≈3 nm)
    RETRY_WITH_ANCHOR_WINDOW = True

    # x0 안정화 후처리(너무 과하면 LER이 작아져서 bias일 수 있음)
    DO_HAMPEL = False
    HAMPEL_K = 7
    HAMPEL_T0 = 3.0
    DO_SAVGOL = False
    SAVGOL_WIN = 31
    SAVGOL_POLY = 2

    # plot/print
    CONTOUR_LW = 1.8
    MAX_EDGES_TO_PRINT = 40
    MAX_EDGES_TO_PLOT_PROFILE = 12
    PROFILE_VIEW_SMOOTH_SIGMA = 1.0

    # binary save
    SAVE_BINARY = True
    OUT_BIN = r"output path\test.png"
    START_BLACK = True
    # =======================

    x, y, z_map = load_xyz_txt(TXT_PATH, skiprows=SKIPROWS, delimiter=DELIMITER)
    nx, ny = len(x), len(y)
    dx = float(x[1] - x[0])
    dy = float(y[1] - y[0]) if ny > 1 else np.nan

    MIN_DISTANCE_PX = max(1, int(np.round(MIN_DISTANCE_UM / dx)))
    print(f"nx={nx}, ny={ny}, dx≈{dx*1000:.2f} nm/px, dy≈{dy*1000:.2f} nm/px")
    print(f"[AUTO] MIN_DISTANCE: {MIN_DISTANCE_UM:.3f} um -> {MIN_DISTANCE_PX} px")

    anchors_rise, anchors_fall, prom_used = find_edge_anchors_both(
        x, z_map,
        smooth_sigma=MEAN_SMOOTH_SIGMA,
        min_distance_px=MIN_DISTANCE_PX,
        prominence=None,
        merge_min_sep_um=MERGE_MIN_SEP_UM
    )
    print(f"anchors(after merge): rising={len(anchors_rise)}, falling={len(anchors_fall)}")

    x0_rise, sig_rise, zlo_rise, zhi_rise = extract_contours_erf_fit_sigma_aware(
        x, y, z_map,
        anchors=anchors_rise,
        mode="rising",
        smooth_sigma=LINE_SMOOTH_SIGMA,
        window_half_width_base=WINDOW_HALF_WIDTH_BASE,
        track_half_width_base=TRACK_HALF_WIDTH_BASE,
        k_window=K_WINDOW,
        k_track=K_TRACK,
        max_jump=MAX_JUMP,
        sigma_init=SIGMA_INIT,
        retry_with_anchor_window=RETRY_WITH_ANCHOR_WINDOW,
        do_hampel=DO_HAMPEL,
        hampel_k=HAMPEL_K,
        hampel_t0=HAMPEL_T0,
        do_savgol=DO_SAVGOL,
        savgol_win=SAVGOL_WIN,
        savgol_poly=SAVGOL_POLY
    )

    x0_fall, sig_fall, zlo_fall, zhi_fall = extract_contours_erf_fit_sigma_aware(
        x, y, z_map,
        anchors=anchors_fall,
        mode="falling",
        smooth_sigma=LINE_SMOOTH_SIGMA,
        window_half_width_base=WINDOW_HALF_WIDTH_BASE,
        track_half_width_base=TRACK_HALF_WIDTH_BASE,
        k_window=K_WINDOW,
        k_track=K_TRACK,
        max_jump=MAX_JUMP,
        sigma_init=SIGMA_INIT,
        retry_with_anchor_window=RETRY_WITH_ANCHOR_WINDOW,
        do_hampel=DO_HAMPEL,
        hampel_k=HAMPEL_K,
        hampel_t0=HAMPEL_T0,
        do_savgol=DO_SAVGOL,
        savgol_win=SAVGOL_WIN,
        savgol_poly=SAVGOL_POLY
    )

    print("[DEBUG] valid x0_rise =", np.isfinite(x0_rise).sum(), "/", x0_rise.size)
    print("[DEBUG] valid x0_fall =", np.isfinite(x0_fall).sum(), "/", x0_fall.size)

    # (1) 전체 패턴 sigma 통계 출력
    print_sigma_stats("RISING", sig_rise)
    print_sigma_stats("FALLING", sig_fall)
    sig_all = np.concatenate([sig_rise[np.isfinite(sig_rise)], sig_fall[np.isfinite(sig_fall)]]) if (
        np.isfinite(sig_rise).any() or np.isfinite(sig_fall).any()
    ) else np.array([])
    if sig_all.size > 0:
        sig_all_nm = sig_all * 1000.0
        print(f"[ALL] sigma(nm) N={sig_all_nm.size}  median={np.median(sig_all_nm):.3f}  p10={np.percentile(sig_all_nm,10):.3f}  p90={np.percentile(sig_all_nm,90):.3f}")

    # overlay용 보간
    x0_rise_f = interpolate_contours_linear(x0_rise)
    x0_fall_f = interpolate_contours_linear(x0_fall)

    # 2) contour overlay plot
    plot_overlay(
        x, y, z_map,
        x0_rise_f, x0_fall_f,
        title="ERF fit edge contours (sigma-aware ROI + x0 smoothing)",
        contour_lw=CONTOUR_LW
    )

    # 3) 대표 y-row 선택 + 식 출력 + 대표식
    iy = choose_profile_row(y, profile_row_list=PROFILE_ROW_LIST, profile_y_list_um=PROFILE_Y_LIST_UM)
    rep = print_erf_equations_for_row(
        iy=iy, y_um=float(y[iy]),
        anchors_rise=anchors_rise, x0_rise=x0_rise, sig_rise=sig_rise, zlo_rise=zlo_rise, zhi_rise=zhi_rise,
        anchors_fall=anchors_fall, x0_fall=x0_fall, sig_fall=sig_fall, zlo_fall=zlo_fall, zhi_fall=zhi_fall,
        max_edges_to_print=MAX_EDGES_TO_PRINT
    )

    # 4) profile plot
    plot_profile_with_erf_fits(
        x, y, z_map,
        iy=iy,
        anchors_rise=anchors_rise, x0_rise=x0_rise, sig_rise=sig_rise, zlo_rise=zlo_rise, zhi_rise=zhi_rise,
        anchors_fall=anchors_fall, x0_fall=x0_fall, sig_fall=sig_fall, zlo_fall=zlo_fall, zhi_fall=zhi_fall,
        max_edges_to_plot=MAX_EDGES_TO_PLOT_PROFILE,
        view_smooth_sigma=PROFILE_VIEW_SMOOTH_SIGMA
    )

    # 5) binary PNG 저장
    if SAVE_BINARY:
        bin_img = contours_to_binary(x, y, x0_rise_f, x0_fall_f, start_black=START_BLACK)
        imageio.imwrite(OUT_BIN, bin_img)
        print(f"[Saved] {OUT_BIN}")
