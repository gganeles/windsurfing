import matplotlib.pyplot as plt
import pandas as pd
from scipy.signal import butter, filtfilt

import numpy as np


def lowpass_filter_filtfilt(data, cutoff, fs, order=5):

    nyquist = 0.5 * fs 
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    
    filtered_data = filtfilt(b, a, data)
    return filtered_data


def smooth_derivative2(vec, threshold, max_itr = 50):
    """
    Removes regions where the absolute derivative exceeds a threshold and replaces
    them with linear interpolation between the edges of the regions.

    Parameters:
    vec (np.ndarray): Input vector.
    threshold (float): Threshold for the absolute derivative.

    Returns:
    np.ndarray: The smoothed vector.
    """
    vec = np.asarray(vec)

    if len(vec) < 2:
        # If the vector is too short, return it as is
        return vec
    derivative = True
    n = 0
    while np.any(derivative>threshold) and n<max_itr:
        # Compute the derivative
        derivative = np.abs(np.diff(vec))

        # Find regions where the absolute derivative exceeds the threshold
        exceeds_threshold = derivative > threshold

        # Identify start and end indices of contiguous regions exceeding the threshold
        smoothed_vec = vec.copy()
        start = None

        for i in range(len(exceeds_threshold)):
            if exceeds_threshold[i]:
                if start is None:
                    start = i-1
            else:
                if start is not None:
                    end = i + 1

                    # Linear interpolation
                    x_start, x_end = start, end
                    y_start, y_end = vec[x_start], vec[x_end]
                    interp_x = np.arange(x_start + 1, x_end)
                    interp_y = np.linspace(y_start, y_end, len(interp_x) + 2)[1:-1]

                    # Replace the region with interpolated values
                    smoothed_vec[x_start + 1:x_end] = interp_y

                    start = None

        # Handle case where region extends to the end of the vector
        if start is not None:
            x_start, x_end = start, len(vec) - 1
            y_start, y_end = vec[x_start], vec[x_end]
            interp_x = np.arange(x_start + 1, x_end + 1)
            interp_y = np.linspace(y_start, y_end, len(interp_x) + 2)[1:-1]
            smoothed_vec[x_start + 1:x_end + 1] = interp_y
        n+=1
    return smoothed_vec

def smooth_derivative(vec, threshold):
    """
    Removes regions where the derivative first drops below -threshold and then exceeds
    +threshold, replacing them with linear interpolation between the edges of the regions.

    Parameters:
    vec (np.ndarray): Input vector.
    threshold (float): Threshold for the derivative.

    Returns:
    np.ndarray: The smoothed vector.
    """
    vec = np.asarray(vec)

    if len(vec) < 2:
        # If the vector is too short, return it as is
        return vec

    # Compute the derivative
    derivative = np.diff(vec)

    # Initialize result vector
    smoothed_vec = vec.copy()

    # State machine to track regions of interest
    state = "idle"  # States: "idle", "below", "above"
    start = None

    for i in range(len(derivative)):
        if state == "idle":
            if derivative[i] < -threshold:
                state = "below"
                start = i
        elif state == "below":
            if derivative[i] > threshold:
                state = "above"
        elif state == "above":
            if derivative[i] <= threshold or i == len(derivative) - 1:
                # End of the region
                end = i + 1

                if start is not None:
                    # Linear interpolation
                    x_start, x_end = start, end
                    y_start, y_end = vec[x_start], vec[x_end]
                    interp_x = np.arange(x_start + 1, x_end)
                    interp_y = np.linspace(y_start, y_end, len(interp_x) + 2)[1:-1]

                    # Replace the region with interpolated values
                    smoothed_vec[x_start + 1:x_end] = interp_y

                # Reset state
                state = "idle"
                start = None

    return smoothed_vec


df = pd.read_csv("horizonParams2.csv")
df.interpolate()



#plt.plot(df["frame"],df["theta1"])

outlierless=np.clip(df["theta1"],2,1)

plt.plot(df["frame"]/29.97,outlierless)
lowpassed = lowpass_filter_filtfilt(outlierless,4,30,2)

dt_thresh = .035
lowpassed2 = smooth_derivative2(outlierless,dt_thresh)

#plt.plot(df["frame"],np.abs(np.diff(lowpassed2,append=0)))

plt.plot(df["frame"]/29.97,lowpass_filter_filtfilt(lowpassed2,4,30,2))
plt.plot(df["frame"]/29.97,np.concatenate([np.abs(np.diff(lowpassed2)),[0]]))

plt.show()