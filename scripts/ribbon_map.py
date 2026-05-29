import numpy as np
from tqdm.auto import tqdm

import sunpy
from sunpy.physics.differential_rotation import differential_rotate
import astropy.units as u


from scipy.interpolate import splprep, splev
from scipy.ndimage import map_coordinates
from scipy.signal import savgol_filter


def iris_reproject(iris_map, ref_aia_map):
    """
    Reproject ref_map onto a WCS derived from amap but scaled to ref_map's pixel resolution.

    Parameters
    ----------
    iris_map : sunpy.map.GenericMap
        Map whose WCS defines the sky region and projection (e.g. IRIS/SJI).
    ref_aia_map : sunpy.map.GenericMap
        Map to reproject (e.g. AIA).

    Returns
    -------
    sunpy.map.GenericMap
        ref_map reprojected onto amap's sky footprint at ref_map's native pixel scale.
    """
    scale_ratio = (ref_aia_map.scale[0] / iris_map.scale[0]).decompose().value

    new_wcs = ref_aia_map.wcs.deepcopy()
    new_wcs.wcs.cdelt = new_wcs.wcs.cdelt / scale_ratio
    new_wcs.wcs.crpix = new_wcs.wcs.crpix * scale_ratio
    new_wcs.wcs.dateobs = iris_map.date.iso

    shape_out = (
        int(np.round(ref_aia_map.data.shape[0] * scale_ratio)),
        int(np.round(ref_aia_map.data.shape[1] * scale_ratio)),
    )

    return iris_map.reproject_to(new_wcs, shape_out=shape_out)


def _build_slice_geometry(sunpy_map, pil_coords, xoffsets, yoffsets):
    """
    Shared geometry kernel for RibbonMap and RibbonCoords.

    Returns
    -------
    sunpy_map : the (possibly diff-rotated) map used for pixel/world conversion
    x_all, y_all : ndarray, shape (num_offsets, num_points) — pixel coords of every slice
    along_arcsec : ndarray, shape (num_points,)
    cross_arcsec : ndarray, shape (num_offsets,)
    """
    scale_x = sunpy_map.scale.axis1.to(u.arcsec / u.pixel).value
    scale_y = sunpy_map.scale.axis2.to(u.arcsec / u.pixel).value

    pixels = sunpy_map.world_to_pixel(pil_coords)
    x_r, y_r = pixels.x.value, pixels.y.value

    tck, _ = splprep([x_r, y_r], s=24)

    u_fine = np.linspace(0, 1, 4096)
    x_fine, y_fine = splev(u_fine, tck)
    arc_pix = np.sum(np.sqrt(np.diff(x_fine)**2 + np.diff(y_fine)**2))
    num_points = max(int(np.round(arc_pix)), 2)

    arc_fine = np.concatenate([[0], np.cumsum(np.sqrt(np.diff(x_fine)**2 + np.diff(y_fine)**2))])
    arc_fine /= arc_fine[-1]
    u_new = np.interp(np.linspace(0, 1, num_points), arc_fine, u_fine)
    x_r, y_r = splev(u_new, tck)

    dx_arc = np.diff(x_r) * scale_x
    dy_arc = np.diff(y_r) * scale_y
    along_arcsec = np.linspace(0.0, np.sum(np.sqrt(dx_arc**2 + dy_arc**2)), num_points)

    if xoffsets is not None:
        xoffsets = np.asarray(xoffsets)
        if xoffsets.ndim == 1 and xoffsets.size == 2:
            xoffsets = np.arange(xoffsets[0], xoffsets[1], scale_x)
    if yoffsets is not None:
        yoffsets = np.asarray(yoffsets)
        if yoffsets.ndim == 1 and yoffsets.size == 2:
            yoffsets = np.arange(yoffsets[0], yoffsets[1], scale_y)

    if yoffsets is None:
        xoffsets_pix = xoffsets / scale_x
        slope, _ = np.polyfit(y_r, x_r, 1)
        mag = np.sqrt(1 + slope**2)
        xoff_pix = xoffsets_pix / mag
        yoff_pix = xoffsets_pix * (-slope) / mag
    else:
        yoffsets_pix = yoffsets / scale_y
        slope, _ = np.polyfit(x_r, y_r, 1)
        mag = np.sqrt(1 + slope**2)
        yoff_pix = yoffsets_pix / mag
        xoff_pix = yoffsets_pix * (-slope) / mag

    primary_arcsec = xoffsets if yoffsets is None else yoffsets
    cross_arcsec = np.sign(primary_arcsec) * np.sqrt((xoff_pix * scale_x)**2 + (yoff_pix * scale_y)**2)

    x_all = x_r[np.newaxis, :] + xoff_pix[:, np.newaxis]
    y_all = y_r[np.newaxis, :] + yoff_pix[:, np.newaxis]

    return sunpy_map, x_all, y_all, along_arcsec, cross_arcsec


def RibbonMap(sunpy_map, pil_coords, xoffsets=None, yoffsets=None, diff_rotate=False, ref_map=None):
    """
    Returns
    -------
    intensity_map : ndarray, shape (num_offsets, num_points)
    along_arcsec  : ndarray, shape (num_points,)
        Cumulative arc-length along the PIL in arcsec.
    cross_arcsec  : ndarray, shape (num_offsets,)
        Cross-ribbon displacement for each slice in arcsec.

    Notes
    -----
    xoffsets / yoffsets are in arcseconds and can be:
      - a 2-element [start, stop] range: slices are generated at every pixel
        step between start and stop (e.g. xoffsets=[-30, 40] → ~500 slices
        spaced by the map's arcsec/pixel scale)
      - an explicit array of arcsecond values for finer control
    num_points is derived automatically so that one sample ≈ one native map
    pixel along the PIL.
    """
    if diff_rotate:
        if ref_map is None:
            raise ValueError("ref_map is required when diff_rotate=True")
        sunpy_map = differential_rotate(sunpy_map, observer=ref_map.observer_coordinate)

    if xoffsets is not None and yoffsets is not None:
        raise ValueError("Only one of xoffsets or yoffsets can be provided, not both.")
    elif xoffsets is None and yoffsets is None:
        print("No offsets provided, returning None.")
        return

    sunpy_map, x_all, y_all, along_arcsec, cross_arcsec = _build_slice_geometry(
        sunpy_map, pil_coords, xoffsets, yoffsets
    )

    intensity_map = map_coordinates(sunpy_map.data, [y_all, x_all], order=1)

    return intensity_map, along_arcsec, cross_arcsec


def RibbonCoords(sunpy_map, pil_coords, xoffsets=None, yoffsets=None, diff_rotate=False, ref_map=None):
    """
    Returns the world coordinates of each slice as a list of SkyCoords.

    Parameters are identical to RibbonMap.

    Returns
    -------
    slice_coords : list of SkyCoord, length num_offsets
        Each element is the world-coordinate path of one slice along the PIL.
    cross_arcsec : ndarray, shape (num_offsets,)
        Cross-ribbon displacement in arcsec for each slice (useful for labelling).
    """
    if diff_rotate:
        if ref_map is None:
            raise ValueError("ref_map is required when diff_rotate=True")
        sunpy_map = differential_rotate(sunpy_map, observer=ref_map.observer_coordinate)

    if xoffsets is not None and yoffsets is not None:
        raise ValueError("Only one of xoffsets or yoffsets can be provided, not both.")
    elif xoffsets is None and yoffsets is None:
        raise ValueError("One of xoffsets or yoffsets must be provided.")

    sunpy_map, x_all, y_all, _, cross_arcsec = _build_slice_geometry(
        sunpy_map, pil_coords, xoffsets, yoffsets
    )

    slice_coords = [
        sunpy_map.pixel_to_world(x_all[i] * u.pixel, y_all[i] * u.pixel)
        for i in range(x_all.shape[0])
    ]

    return slice_coords, cross_arcsec
