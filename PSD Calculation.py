import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from scipy import signal
from scipy.stats import chi2

# ============================================================
# 0) Helpers: edge extraction
# ============================================================
def detrend_linear(y_idx, x):
    a, b = np.polyfit(y_idx, x, 1)
    return x - (a * y_idx + b)

def row_crossings_subpixel(row, level=0.5):
    r = np.asarray(row, dtype=np.float64)
    s = r - level
    x_list = []

    # exact hits
    zero_idx = np.where(s == 0)[0]
    for i in zero_idx:
        x_list.append(float(i))

    # sign changes
    s0 = s[:-1]
    s1 = s[1:]
    idx = np.where((s0 * s1) < 0)[0]
    for i in idx:
        denom = (s1[i] - s0[i])
        if denom == 0:
            continue
        t = -s0[i] / denom
        x_list.append(float(i + t))

    return sorted(set(x_list))

def extract_all_edge_series_from_grayscale(img_gray_01, level=0.5, remove_double_edges=True):
    H, W = img_gray_01.shape
    crossings_per_row = []
    y_kept = []

    for y in range(H):
        xs = row_crossings_subpixel(img_gray_01[y, :], level=level)
        if remove_double_edges:
            # line/space patterns: usually even number of crossings
            if len(xs) < 2 or (len(xs) % 2 != 0):
                continue
        crossings_per_row.append(xs)
        y_kept.append(y)

    if len(y_kept) < 32:
        raise ValueError("Too few valid rows after filtering. Check ROI/threshold/image quality.")

    max_edges = max(len(xs) for xs in crossings_per_row)
    Ny = len(y_kept)
    edges = [np.full(Ny, np.nan, dtype=np.float64) for _ in range(max_edges)]

    for i, xs in enumerate(crossings_per_row):
        for k, x in enumerate(xs):
            edges[k][i] = x

    return np.array(y_kept, dtype=int), edges

# ============================================================
# 1) PSD / ACF / stats
# ============================================================
def psd_fft_density(x_nm, dy_nm):
    """
    One-sided PSD via FFT (no smoothing).
    f: um^-1 (cycles/um)
    P: nm^2 / (um^-1)  (equivalently nm^2*um)
    """
    x = np.asarray(x_nm, dtype=np.float64)
    x = x - np.mean(x)

    dy_um = dy_nm / 1000.0
    N = len(x)

    f = np.fft.rfftfreq(N, d=dy_um)  # cycles/um
    X = np.fft.rfft(x)

    P = (dy_um / N) * (np.abs(X) ** 2)
    if N % 2 == 0:
        P[1:-1] *= 2.0
    else:
        P[1:] *= 2.0

    m = (f > 0) & np.isfinite(P) & (P > 0)
    return f[m], P[m]

def autocorr_1d(x):
    """Normalized ACF: C[k] = R[k]/R[0], k>=0."""
    x = np.asarray(x, dtype=np.float64)
    x = x - np.mean(x)
    acf_full = signal.correlate(x, x, mode="full", method="auto")
    acf = acf_full[len(x) - 1 :]
    if acf[0] != 0:
        acf = acf / acf[0]
    return acf

def integral_correlation_length_nm(acf, dy_nm, cutoff=0.0, max_lag_nm=None):
    """
    xi_int = ∫_0^{lc} C(l) dl
    lc = first lag where C(l) <= cutoff (default cutoff=0),
    optionally capped by max_lag_nm.
    """
    C = np.asarray(acf, dtype=np.float64)
    lags_nm = np.arange(len(C), dtype=np.float64) * dy_nm

    stop_idx = len(C)
    if len(C) > 2:
        hits = np.where(C[1:] <= cutoff)[0]
        if hits.size > 0:
            stop_idx = min(stop_idx, int(hits[0] + 1))

    if max_lag_nm is not None:
        cap_idx = int(np.searchsorted(lags_nm, max_lag_nm, side="right"))
        stop_idx = min(stop_idx, max(2, cap_idx))

    stop_idx = max(2, stop_idx)
    return float(np.trapz(C[:stop_idx], lags_nm[:stop_idx]))

def correlation_length_threshold_nm(acf, dy_nm, target, max_lag_nm=None):
    """
    xi_target: first lag where C(l) decays to 'target' (e.g., exp(-2)),
    using linear interpolation between adjacent discrete lags.
    """
    C = np.asarray(acf, dtype=np.float64)
    lags_nm = np.arange(len(C), dtype=np.float64) * dy_nm

    stop = len(C)
    if max_lag_nm is not None:
        stop = min(stop, int(np.searchsorted(lags_nm, max_lag_nm, side="right")))
        stop = max(stop, 2)

    C = C[:stop]
    lags_nm = lags_nm[:stop]

    idx = np.where(C[1:] <= target)[0]
    if idx.size == 0:
        return np.nan

    k = int(idx[0] + 1)
    y0, y1 = C[k - 1], C[k]
    x0, x1 = lags_nm[k - 1], lags_nm[k]
    if y1 == y0:
        return float(x1)

    t = (target - y0) / (y1 - y0)
    return float(x0 + t * (x1 - x0))

def psd_integrated_sigma_nm(f, P):
    """sigma^2 = ∫ PSD(f) df"""
    f = np.asarray(f, dtype=np.float64)
    P = np.asarray(P, dtype=np.float64)
    m = np.isfinite(f) & np.isfinite(P) & (f > 0) & (P > 0)
    if np.sum(m) < 2:
        return np.nan
    var = np.trapz(P[m], f[m])
    return float(np.sqrt(var))

def neff_from_xi(L_total_nm, xi_nm, factor=2.0):
    """N_eff ≈ L_total / (factor * xi)"""
    if xi_nm <= 0 or not np.isfinite(xi_nm):
        return np.nan
    return float(L_total_nm / (factor * xi_nm))

def chi2_ci_for_sigma(s_nm, neff, alpha=0.05):
    """Chi-square CI for true sigma given sample std s and dof=neff-1."""
    if (not np.isfinite(neff)) or neff <= 2:
        return (np.nan, np.nan)
    dof = neff - 1.0
    lo = np.sqrt((dof * s_nm**2) / chi2.ppf(1 - alpha/2, dof))
    hi = np.sqrt((dof * s_nm**2) / chi2.ppf(alpha/2, dof))
    return float(lo), float(hi)

# ============================================================
# 2) H band from xi and Nyquist + linear regression
# ============================================================
def compute_H_band_from_xi(xi_nm, dy_nm, fmax_fraction_of_nyq=0.10):
    """
    fmin = 1/xi
    fmax = fmax_fraction_of_nyq * fNyq
    """
    dy_um = dy_nm / 1000.0
    fNyq = 1.0 / (2.0 * dy_um)

    xi_um = xi_nm / 1000.0
    fmin = 1.0 / xi_um if xi_um > 0 else np.nan
    fmax = float(fmax_fraction_of_nyq) * fNyq
    return float(fmin), float(fmax), float(fNyq)

def linear_regression_H(f, P, fmin, fmax):
    """
    log10(P) = slope*log10(f) + intercept
    H = (-slope - 1)/2   for 1D edge PSD ~ f^{-(2H+1)}
    """
    f = np.asarray(f, float)
    P = np.asarray(P, float)

    m = np.isfinite(f) & np.isfinite(P) & (f > 0) & (P > 0) & (f >= fmin) & (f <= fmax)
    ff = f[m]
    PP = P[m]
    if ff.size < 10:
        return np.nan, np.nan, np.nan, np.nan, 0, ff, PP

    x = np.log10(ff)
    y = np.log10(PP)
    slope, intercept = np.polyfit(x, y, 1)

    yhat = slope * x + intercept
    ss_res = np.sum((y - yhat) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan

    H = (-slope - 1.0) / 2.0
    return float(H), float(slope), float(intercept), float(r2), int(ff.size), ff, PP

# ============================================================
# 3) Main analysis
#    - H band fmin uses xi_int (ACF integral)
#    - CI Neff uses xi_1e2 (threshold exp(-2))
#    - Plot only PSD (no lag/ACF subplot)
# ============================================================
def analyze_edges_psd_xi_mixed_H_CI(
    png_path,
    nm_per_px,
    roi=None,
    contour_level=0.5,
    remove_double_edges=True,
    detrend="linear",

    # H band control
    fmax_fraction_of_nyq=0.15,

    # xi settings
    xi_int_cutoff=0.0,       # for xi_int: usually 0
    xi_max_lag_nm=None,      # cap for both xi computations (optional)
    neff_factor=1.0,
    alpha=0.05,

    # fallback for CI xi if exp(-2) not reached: "xi_1e" or "xi_int"
    ci_fallback="xi_1e",

    # plot
    plot=True,
    title="Avg PSD + H band from ξ_int(ACF integral) + CI from ξ_1/e²",
    legend_outside=False,
    show_fitline=True,
):
    # load grayscale
    img = Image.open(png_path).convert("L")
    g = np.asarray(img, dtype=np.float64) / 255.0
    if roi is not None:
        y0, y1, x0, x1 = roi
        g = g[y0:y1, x0:x1]

    dy_nm = nm_per_px

    # edge extraction
    y_idx, edge_series_px = extract_all_edge_series_from_grayscale(
        g, level=contour_level, remove_double_edges=remove_double_edges
    )
    Ny = len(y_idx)

    # residual per edge
    edge_residuals_nm = []
    edge_sigmas_nm = []
    for xs in edge_series_px:
        m = np.isfinite(xs)
        if np.sum(m) < 64:
            continue
        yk = y_idx[m]
        xk = xs[m]

        if detrend == "linear":
            rk = detrend_linear(yk, xk)
        elif detrend == "mean":
            rk = xk - np.mean(xk)
        else:
            raise ValueError("detrend must be 'linear' or 'mean'")

        rk_nm = rk * nm_per_px
        edge_residuals_nm.append(rk_nm)
        edge_sigmas_nm.append(float(np.std(rk_nm, ddof=1)))

    if len(edge_residuals_nm) < 2:
        raise ValueError("Not enough valid edges extracted. Check ROI/threshold.")

    # LER
    sigma_mean_nm = float(np.mean(edge_sigmas_nm))
    ler_3sigma_nm = 3.0 * sigma_mean_nm

    # PSD per edge -> geometric mean
    f_ref = None
    psd_list = []
    for rk_nm in edge_residuals_nm:
        f, P = psd_fft_density(rk_nm, dy_nm)
        if f_ref is None:
            f_ref = f
            psd_list.append(P)
        else:
            if len(f) != len(f_ref) or np.max(np.abs(f - f_ref)) > 1e-12:
                P = np.interp(f_ref, f, P, left=np.nan, right=np.nan)
            psd_list.append(P)

    psd_arr = np.vstack(psd_list)
    Pxx_avg = np.exp(np.nanmean(np.log(psd_arr), axis=0))  # geometric mean

    # pooled residuals -> ACF
    pooled = np.concatenate([r for r in edge_residuals_nm if len(r) >= 64])
    acf = autocorr_1d(pooled)

    # --- xi definitions ---
    # (A) H-band uses xi_int
    xi_int_nm = integral_correlation_length_nm(
        acf, dy_nm, cutoff=xi_int_cutoff, max_lag_nm=xi_max_lag_nm
    )

    # (B) CI uses xi_1/e^2 (threshold)
    xi_1e_nm = correlation_length_threshold_nm(
        acf, dy_nm, target=np.exp(-1), max_lag_nm=xi_max_lag_nm
    )
    xi_1e2_nm = correlation_length_threshold_nm(
        acf, dy_nm, target=np.exp(-2), max_lag_nm=xi_max_lag_nm
    )

    # choose xi for CI with fallback
    xi_for_CI = xi_1e2_nm
    ci_used = "xi_1e2"
    if not np.isfinite(xi_for_CI) or xi_for_CI <= 0:
        if ci_fallback == "xi_int":
            xi_for_CI = xi_int_nm
            ci_used = "xi_int(fallback)"
        else:
            xi_for_CI = xi_1e_nm
            ci_used = "xi_1e(fallback)"

    # H band from xi_int and Nyquist
    H_fmin, H_fmax, fNyq = compute_H_band_from_xi(
        xi_int_nm, dy_nm, fmax_fraction_of_nyq=fmax_fraction_of_nyq
    )

    # clamp band to PSD available range
    fm = np.isfinite(f_ref) & (f_ref > 0)
    if np.any(fm):
        H_fmin = max(H_fmin, float(np.min(f_ref[fm])))
        H_fmax = min(H_fmax, float(np.max(f_ref[fm])))

    # linear regression in the band
    H_est, slope, intercept, r2, npts, f_fit, P_fit = linear_regression_H(
        f_ref, Pxx_avg, H_fmin, H_fmax
    )

    # CI using Neff from xi_for_CI
    L_total_nm = float(len(pooled) * dy_nm)
    neff = neff_from_xi(L_total_nm, xi_for_CI, factor=neff_factor)
    sigma_ci_lo, sigma_ci_hi = chi2_ci_for_sigma(sigma_mean_nm, neff, alpha=alpha)
    ler_ci_lo, ler_ci_hi = 3.0 * sigma_ci_lo, 3.0 * sigma_ci_hi

    # sigma from PSD integral
    sigma_psd_nm = psd_integrated_sigma_nm(f_ref, Pxx_avg)

    # ------------------------------------------------------------
    # Plot PSD only
    # ------------------------------------------------------------
    if plot:
        plt.figure(figsize=(8.0, 6.0))
        m = np.isfinite(Pxx_avg) & (Pxx_avg > 0) & (f_ref > 0)
        plt.loglog(f_ref[m], Pxx_avg[m], linestyle='-', color="midnightblue", label="Avg PSD")

        # H band lines (from xi_int)
        #if np.isfinite(H_fmin) and np.isfinite(H_fmax):
        #    plt.axvline(H_fmin, linestyle="--", linewidth=1.0, color="gray")
        #    plt.axvline(H_fmax, linestyle="--", linewidth=1.0, color="gray")

        # regression line overlay
        #if show_fitline and np.isfinite(slope) and np.isfinite(intercept) and npts >= 10:
            #P_line = 10 ** (slope * np.log10(f_fit) + intercept)
            #plt.loglog(f_fit, P_line, color="red", linewidth=2.5, label="Linear fit (log–log)")

        plt.xlabel("Spatial frequency f (µm⁻¹)")
        plt.ylabel("PSD (nm² / µm⁻¹)")
        plt.title(title)

        def fmt(x):
            return f"{x:.2f}" if np.isfinite(x) else "nan"

        #txt = (
        #    f"ξ_int (for H band, ACF integral) = {fmt(xi_int_nm)} nm\n"
        #    f"ξ_1/e² (for CI Neff) = {fmt(xi_1e2_nm)} nm\n"
        #    f"CI uses: {ci_used},  ξ_CI = {fmt(xi_for_CI)} nm\n"
        #    f"H-band: {H_fmin:.2g}–{H_fmax:.2g} µm⁻¹   (fNyq={fNyq:.2g}, fmax={fmax_fraction_of_nyq:.2g}·fNyq)\n"
        #    f"H = {H_est:.3f} (R²={r2:.3f}, N={npts})\n"
        #    f"LER(3σ from edges) = {ler_3sigma_nm:.2f} nm\n"
        #    f"95% CI LER: [{ler_ci_lo:.2f}, {ler_ci_hi:.2f}] nm\n"
        #)
        #plt.plot([], [], ' ', label=txt)

        #if legend_outside:
        #    plt.legend(loc="upper left", bbox_to_anchor=(1.02, 1),
        #               frameon=True, fontsize=10, borderpad=0.25,
        #               labelspacing=0.25, handlelength=0, handletextpad=0)
        #    plt.subplots_adjust(right=0.75)
        #else:
        #    plt.legend(loc="upper right", frameon=True, fontsize=10, borderpad=0.25)
        plt.grid(True, which='major', axis='both', linestyle=':')
        plt.grid(True, which='minor', axis='both', linestyle=':')
        plt.tick_params(axis='both', which='major', length=5, width=1.2)
        plt.tick_params(axis='both', which='minor', length=3, width=1)
        plt.tight_layout()
        plt.show()

    results = {
        "nm_per_px": float(nm_per_px),
        "Ny_used_rows": int(Ny),
        "edges_used": int(len(edge_residuals_nm)),

        "LER_3sigma_nm": float(ler_3sigma_nm),
        "sigma_mean_nm": float(sigma_mean_nm),
        "sigma_psd_integrated_nm": float(sigma_psd_nm) if np.isfinite(sigma_psd_nm) else np.nan,

        # xi
        "xi_int_nm": float(xi_int_nm) if np.isfinite(xi_int_nm) else np.nan,
        "xi_1e_nm": float(xi_1e_nm) if np.isfinite(xi_1e_nm) else np.nan,
        "xi_1e2_nm": float(xi_1e2_nm) if np.isfinite(xi_1e2_nm) else np.nan,

        # CI chosen xi + Neff
        "xi_for_CI_nm": float(xi_for_CI) if np.isfinite(xi_for_CI) else np.nan,
        "CI_xi_source": str(ci_used),
        "Neff_CI": float(neff) if np.isfinite(neff) else np.nan,
        "LER_CI_low_nm": float(ler_ci_lo) if np.isfinite(ler_ci_lo) else np.nan,
        "LER_CI_high_nm": float(ler_ci_hi) if np.isfinite(ler_ci_hi) else np.nan,

        # H
        "H": float(H_est) if np.isfinite(H_est) else np.nan,
        "H_slope": float(slope) if np.isfinite(slope) else np.nan,
        "H_R2": float(r2) if np.isfinite(r2) else np.nan,
        "H_band_fmin_um_inv": float(H_fmin) if np.isfinite(H_fmin) else np.nan,
        "H_band_fmax_um_inv": float(H_fmax) if np.isfinite(H_fmax) else np.nan,
        "fNyq_um_inv": float(fNyq),

        "params": {
            "fmax_fraction_of_nyq": float(fmax_fraction_of_nyq),
            "xi_int_cutoff": float(xi_int_cutoff),
            "xi_max_lag_nm": float(xi_max_lag_nm) if xi_max_lag_nm is not None else None,
            "neff_factor": float(neff_factor),
            "alpha": float(alpha),
            "ci_fallback": str(ci_fallback),
        }
    }
    return f_ref, Pxx_avg, results


# ============================================================
# 4) Example usage
# ============================================================
if __name__ == "__main__":
    png_path = r"input path"
    nm_per_px = 1.220
    roi = None

    f, Pxx_avg, res = analyze_edges_psd_xi_mixed_H_CI(
        png_path=png_path,
        nm_per_px=nm_per_px,
        roi=roi,
        contour_level=0.5,
        remove_double_edges=True,
        detrend="linear",

        fmax_fraction_of_nyq=0.15,

        # xi settings
        xi_int_cutoff=0.0,        # xi_int uses first C<=0 by default
        xi_max_lag_nm=None,       # optional cap (e.g., 2000)
        neff_factor=1.0,
        alpha=0.05,

        # if xi_1/e^2 is not reached, fallback to xi_1/e (or "xi_int")
        ci_fallback="xi_1e",

        plot=True,
        legend_outside=False,
        show_fitline=True,
    )

    print("\n=== Results ===")
    for k, v in res.items():
        if k != "params":
            print(f"{k}: {v}")
    print("\n=== Params ===")
    for k, v in res["params"].items():
        print(f"{k}: {v}")