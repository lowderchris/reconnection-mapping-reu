import numpy as np
import time
from datetime import datetime

def iris_prep_despike(data, sigmas=4.5, Niter=10, kernel=None, min_std=1.0, 
                     silent=False, mode='bright'):
    """
    Generalized data despiking tool for 1-4 dimensional data arrays.
    Rewritten for Python from IDL code: see IRIS_PREP_DESPIKE.pro 
        https://hesperia.gsfc.nasa.gov/ssw/iris/idl/lmsal/calibration/iris_prep_despike.pro

    
    
    Parameters:
    -----------
    data : ndarray
        Input data array with 1-4 dimensions
    sigmas : float, optional
        Threshold for designating a bad pixel, as a multiple of the 
        neighborhood standard deviation. Default is 4.5.
    Niter : int, optional
        Maximum number of iterations for identifying bad pixels. Default is 10.
    kernel : ndarray, optional
        Convolution kernel used for calculating neighborhood mean and std dev.
        If not provided, default kernels are used based on data dimensions.
    min_std : float, optional
        Minimum value for the local standard deviation. Default is 1.0.
    silent : bool, optional
        If True, suppress verbose output. Default is False.
    mode : str, optional
        Set to detect "bright" spikes, "dark" spikes, or "both". Default is "bright".
        
    Returns:
    --------
    result : ndarray
        Processed version of data with spikes removed
    goodmap : ndarray
        Map of good pixels (1.0 for good, 0.0 for bad)
    """
    if not silent:
        print(f"{datetime.now().strftime('%c')} IRIS_PREP_DESPIKE started on array of {data.size} elements.")
        print("Step (1): Iteratively identifying bad pixels.")
    
    # Deal with NaN and Inf
    where_bad = np.where(~np.isfinite(data))
    if where_bad[0].size > 0:
        bad_crap = data[where_bad]
        data_copy = data.copy()  # Make a copy to avoid modifying the input
        data_copy[where_bad] = 1e-6  # A small number << 1 DN
    else:
        data_copy = data.copy()
        bad_crap = None
    
    # Assess data size & dimensionality
    Ndim = data.ndim
    t_begin = time.time()
    
    # Set default kernel if not provided
    if kernel is None:
        if Ndim == 1:
            kernel = np.ones(11)
        elif Ndim == 2:
            kernel = np.ones((9, 9))
        elif Ndim == 3:
            kernel = np.ones((5, 5, 5))
        elif Ndim == 4:
            kernel = np.ones((3, 3, 3, 3))
        else:
            raise ValueError("Data dimensionality is too great. Cannot construct kernel.")
    
    # Helper function for n-dimensional convolution
    def nd_convol(array1, array2, edge_truncate=True):
        """
        N-dimensional convolution equivalent to IDL's convol with edge_truncate
        """
        from scipy import ndimage
        
        # For edge_truncate, we use 'reflect' mode in scipy.ndimage
        return ndimage.convolve(array1, array2, mode='reflect')
    
    # Identify bad pixels
    goodmap = np.ones_like(data_copy, dtype=float)  # Map of good pixels, initially all 1's
    
    for i in range(1, Niter + 1):
        neighborhood_mean = nd_convol(goodmap * data_copy, kernel) / \
                            nd_convol(goodmap, kernel)
        
        if mode == 'bright':
            deviation = data_copy - neighborhood_mean      # Find bright spikes only
        elif mode == 'dark':
            deviation = neighborhood_mean - data_copy      # Find dark spikes only
        elif mode == 'both':
            deviation = np.abs(data_copy - neighborhood_mean)  # Find both bright and dark spikes
        else:
            raise ValueError(f"Called with undefined mode: {mode}")
        
        neighborhood_std = np.sqrt(nd_convol(goodmap * deviation**2, kernel) / 
                                  nd_convol(goodmap, kernel))
        neighborhood_std = np.maximum(neighborhood_std, min_std)
        
        bad = np.where(deviation > (sigmas * neighborhood_std))
        
        if bad[0].size == 0:
            break
            
        newly_bad = np.where(goodmap[bad] > 0)[0]
        
        if newly_bad.size == 0:
            break
            
        if not silent:
            print(f"Iteration {i:4d} found {bad[0].size:12d} bad pixels, {newly_bad.size:12d} of them new.")
            
        # Update bad pixels in goodmap
        goodmap[tuple(bad_idx[newly_bad] for bad_idx in bad)] = 0.0
    
    if not silent:
        print("Step (2): Replacing bad pixels")
    
    # Construct kernel k2 of the form exp(-r)/(1+r^Ndim)
    Nk2 = 5  # Size of very-near-local smoothing kernel
    middle = (Nk2 - 1) // 2
    
    if Ndim == 1:
        k2 = np.zeros(Nk2)
        for i in range(Nk2):
            x = i - middle
            k2[i] = np.exp(-np.abs(x))
    elif Ndim == 2:
        k2 = np.zeros((Nk2, Nk2))
        for i in range(Nk2):
            x = i - middle
            for j in range(Nk2):
                y = j - middle
                r = np.sqrt(x**2 + y**2)
                k2[i, j] = np.exp(-r) / (1 + r**Ndim)
    elif Ndim == 3:
        k2 = np.zeros((Nk2, Nk2, Nk2))
        for i in range(Nk2):
            x = i - middle
            for j in range(Nk2):
                y = j - middle
                for k in range(Nk2):
                    z = k - middle
                    r = np.sqrt(x**2 + y**2 + z**2)
                    k2[i, j, k] = np.exp(-r) / (1 + r**Ndim)
    elif Ndim == 4:
        k2 = np.zeros((Nk2, Nk2, Nk2, Nk2))
        for i in range(Nk2):
            x = i - middle
            for j in range(Nk2):
                y = j - middle
                for k in range(Nk2):
                    z = k - middle
                    for m in range(Nk2):
                        t = m - middle
                        r = np.sqrt(x**2 + y**2 + z**2 + t**2)
                        k2[i, j, k, m] = np.exp(-r) / (1 + r**Ndim)
    else:
        raise ValueError("Data dimensionality is too great. Cannot construct k2.")
    
    # Calculate local mean for replacement
    neighborhood_mean = nd_convol(goodmap * data_copy, k2) / nd_convol(goodmap, k2)
    
    # Replace bad pixels
    bad = np.where(goodmap == 0)
    result = data_copy.copy()
    
    if bad[0].size > 0:
        result[bad] = neighborhood_mean[bad]
    
    # Restore the NaN and Inf values that were filtered out
    if where_bad[0].size > 0:
        data_copy[where_bad] = bad_crap
        result[where_bad] = bad_crap
    
    t_end = time.time()
    if not silent:
        print(f"{datetime.now().strftime('%c')} IRIS_PREP_DESPIKE finished, {t_end - t_begin:.1f} sec elapsed.")
    
    return result