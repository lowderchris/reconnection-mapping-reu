import numpy as np
from scipy.interpolate import interp1d, PchipInterpolator
from scipy.ndimage import map_coordinates, gaussian_filter1d


def _smooth(x, y, xq, smooth):
    o = np.argsort(x)
    x, y = np.asarray(x, float)[o], np.asarray(y, float)[o]
    xu, inv = np.unique(x, return_inverse=True)
    yu = np.zeros(xu.size); cnt = np.zeros(xu.size)
    np.add.at(yu, inv, y); np.add.at(cnt, inv, 1.0)
    yu /= cnt
    if smooth > 0 and xu.size >= 3:
        yu = gaussian_filter1d(yu, sigma=smooth)
    if xu.size >= 2:
        return PchipInterpolator(xu, yu)(np.clip(xq, xu[0], xu[-1]))
    return np.interp(xq, xu, yu)


def unshear(image, coords, xarc, yarc, *, smooth=0.5, order=1):
    """
    Parameters
    ----------
    image      : (ny, nx) array to warp.
    coords     : (2N, 2) clicked points in arcsec, corresponding to conjugate footpoints
    xarc, yarc : 1-D arcsec coords of image columns / rows.
    smooth     : spline smoothing scale for the ribbon loci (0 = piecewise lin).
    order      : map_coordinates spline order for the final resample.

    Returns
    -------
    dict with keys: warped, x_dense, y_bot, y_top
    """
    image = np.asarray(image, float)
    ny, nx = image.shape

    # ── 1. Split clicks into bottom / top pairs ─────────────────────────────
    bottom, top = coords[0::2], coords[1::2]
    n_pairs = min(len(bottom), len(top))
    xb, yb = bottom[:n_pairs, 0].astype(float), bottom[:n_pairs, 1].astype(float)
    xt, yt = top[:n_pairs, 0].astype(float),    top[:n_pairs, 1].astype(float)

    # Auto-correct pairs clicked in wrong order (bottom above top)
    swap = yb > yt
    xb, xt = np.where(swap, xt, xb), np.where(swap, xb, xt)
    yb, yt = np.where(swap, yt, yb), np.where(swap, yb, yt)

    # ── 2. Per-pair geometry ────────────────────────────────────────────────
    t_pil_arr = -yb / (yt - yb)
    L_arr     = np.sqrt((xt - xb) ** 2 + (yt - yb) ** 2)
    x_pil_arr = xb + t_pil_arr * (xt - xb)

    # ── 3. Drop bad pairs ───────────────────────────────────────────────────
    valid = (np.abs(yt - yb) > 0.01) & (t_pil_arr > 0) & (t_pil_arr < 1)
    xb, yb    = xb[valid], yb[valid]
    xt, yt    = xt[valid], yt[valid]
    t_pil_arr = t_pil_arr[valid]
    L_arr     = L_arr[valid]
    x_pil_arr = x_pil_arr[valid]

    # ── 4. Sort by PIL crossing x ───────────────────────────────────────────
    idx       = np.argsort(x_pil_arr)
    xb, yb    = xb[idx], yb[idx]
    xt, yt    = xt[idx], yt[idx]
    t_pil_arr = t_pil_arr[idx]
    L_arr     = L_arr[idx]
    x_pil_arr = x_pil_arr[idx]
    x_ref     = x_pil_arr

    # ── 5. Interpolators ────────────────────────────────────────────────────
    kw = dict(kind='linear', fill_value='extrapolate')
    f_xb   = interp1d(x_ref, xb,        **kw)
    f_yb   = interp1d(x_ref, yb,        **kw)
    f_xt   = interp1d(x_ref, xt,        **kw)
    f_yt   = interp1d(x_ref, yt,        **kw)
    f_tpil = interp1d(x_ref, t_pil_arr, **kw)
    f_L    = interp1d(x_ref, L_arr,     **kw)
    f_xpil = interp1d(x_ref, x_pil_arr, **kw)

    # ── 6. Output grid (y=0 forced onto a grid line) ────────────────────────
    x_arc_out = np.linspace(xarc[0], xarc[-1], nx)
    dy        = (yarc[-1] - yarc[0]) / (len(yarc) - 1)
    n_below   = int(np.round(-yarc[0] / dy))
    n_above   = ny - 1 - n_below
    yarc_full = np.arange(-n_below, n_above + 1) * dy
    y_arc_out = yarc_full
    X_out, Y_out = np.meshgrid(x_arc_out, y_arc_out)

    # ── 7. Clamp to valid interpolation range ───────────────────────────────
    x_clamp = np.clip(X_out, x_ref[0], x_ref[-1])

    # ── 8. Inverse warp (PIL row hard-fixed) ────────────────────────────────
    t_pil_col = f_tpil(x_clamp)
    L_col     = f_L(x_clamp)
    y_bot_arc = -t_pil_col * L_col
    y_top_arc = (1 - t_pil_col) * L_col

    s = np.where(Y_out <= 0, Y_out / np.abs(y_bot_arc),
                             Y_out / np.abs(y_top_arc))
    t = t_pil_col + s * np.where(s <= 0, t_pil_col, 1 - t_pil_col)

    X_in_raw = f_xb(x_clamp) + t * (f_xt(x_clamp) - f_xb(x_clamp))
    Y_in     = f_yb(x_clamp) + t * (f_yt(x_clamp) - f_yb(x_clamp))

    x_pil_in = f_xpil(x_clamp)
    X_in     = X_in_raw + (X_out - x_pil_in)

    pil_row          = np.argmin(np.abs(y_arc_out))
    X_in[pil_row, :] = X_out[pil_row, :]
    Y_in[pil_row, :] = Y_out[pil_row, :]

    # ── 9. Convert arcsec → pixel indices ───────────────────────────────────
    xi = np.interp(X_in.ravel(), xarc,      np.arange(nx)).reshape(ny, nx)
    yi = np.interp(Y_in.ravel(), yarc_full, np.arange(ny)).reshape(ny, nx)

    # ── 10. Resample ────────────────────────────────────────────────────────
    warped = map_coordinates(image, [yi, xi], order=order,
                             mode='constant', cval=0)

    # ── 11. Smooth ribbon loci in output ────────────────────────────────────
    x_dense = np.linspace(x_ref[0], x_ref[-1], 500)
    y_bot   = _smooth(x_ref, -t_pil_arr * L_arr,        x_dense, smooth)
    y_top   = _smooth(x_ref, (1 - t_pil_arr) * L_arr,   x_dense, smooth)

    return warped, x_dense, y_bot, y_top
