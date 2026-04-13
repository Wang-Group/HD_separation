from matplotlib import pyplot as plt
import numpy as np
from scipy import signal
import scipy
import pandas as pd

def read_in_GC_MS_data(file_path):
    """
    Read in GC-MS data using pandas
    Args:
        file_path: str, the path of the file
    Returns:
        time: the time in seconds
        mass_data_total: [data_of_scan1, data_of_scan2, data_of_scan3]
    """
    data = pd.read_csv(file_path)
    # read in data 
    keys = data.keys()
    start_index = 24
    time = data[keys[1]][23:]
    time = time.values[1:].astype(float)/1000 #convert from ms to s 
    mass_data_total = [data[keys[i]][24:].astype(float).values for i in range(2,5)]
    return time, mass_data_total

def low_pass_filter(wavelengths,intensities,fc=15):
    """
    To do:
        Use a low pass filter to deal with the read_in data.
    Args:
        wavelengths: the input wavelength.
        intensities: the corresponding intensities of UV-Vis.
        fc: frequency of low pass filter.
    Returns:
        series: the intensities passing through the low-pass filter.
    """
    fs = wavelengths.shape[0]  # Sampling frequency
    w = fc / (fs / 2) # Normalize the frequency
    b, a = signal.butter(5, w, 'low')
    series = signal.filtfilt(b, a, intensities)
    return series

def naive_detect_step(dary,verbose = False):
    """
    Detect the step change of a time series signal. 
    Input:
        dary: a numpy array 
        verbose: if plot the data
    Returns:
        step_indx1: the step for increasing 
        step_indx2: the step for decreasing
    """
    step = np.hstack((np.ones(len(dary)), -1*np.ones(len(dary))))
    dary_step = np.convolve(dary, step, mode='valid')
    # Get the peaks of the convolution
    peaks = signal.find_peaks(dary_step, width=20)[0]
    # plots
    if verbose:
        plt.plot(dary)
        plt.plot(dary_step/10)

    return peaks

def thresholding_algo(y, lag, threshold, influence):
    """
    Random code from stackoverflow (https://stackoverflow.com/questions/22583391/peak-signal-detection-in-realtime-timeseries-data)
    """
    signals = np.zeros(len(y))
    filteredY = np.array(y)
    avgFilter = [0]*len(y)
    stdFilter = [0]*len(y)
    avgFilter[lag - 1] = np.mean(y[0:lag])
    stdFilter[lag - 1] = np.std(y[0:lag])
    for i in range(lag, len(y)):
        if abs(y[i] - avgFilter[i-1]) > threshold * stdFilter [i-1]:
            if y[i] > avgFilter[i-1]:
                signals[i] = 1
            else:
                signals[i] = -1

            filteredY[i] = influence * y[i] + (1 - influence) * filteredY[i-1]
            avgFilter[i] = np.mean(filteredY[(i-lag+1):i+1])
            stdFilter[i] = np.std(filteredY[(i-lag+1):i+1])
        else:
            signals[i] = 0
            filteredY[i] = y[i]
            avgFilter[i] = np.mean(filteredY[(i-lag+1):i+1])
            stdFilter[i] = np.std(filteredY[(i-lag+1):i+1])

    return dict(signals = np.asarray(signals),
                avgFilter = np.asarray(avgFilter),
                stdFilter = np.asarray(stdFilter))

def detect_via_thresholding_algo(y, lag = 30, threshold = 5, influence = 0):
    """
    Random code from stackoverflow using thresholding_algo
    """
    # Run algo with settings from above
    result = thresholding_algo(y, lag=lag, threshold=threshold, influence=influence)
    # Plot result
    plt.subplot(211)
    plt.plot(np.arange(1, len(y)+1), y)

    plt.plot(np.arange(1, len(y)+1),
            result["avgFilter"], color="cyan", lw=2)

    plt.plot(np.arange(1, len(y)+1),
            result["avgFilter"] + threshold * result["stdFilter"], color="green", lw=2)

    plt.plot(np.arange(1, len(y)+1),
            result["avgFilter"] - threshold * result["stdFilter"], color="green", lw=2)

    plt.subplot(212)
    plt.step(np.arange(1, len(y)+1), result["signals"], color="red", lw=2)
    plt.ylim(-1.5, 1.5)
    plt.show()
    return result

def detect_via_electrolysis_time(
        ms_data, 
        time, 
        start_time,
        end_time,
        int_start_delay = 8*60,
        int_end_delay = 0,
        interval_distance1 = 60,
        interval_length1 = 60,
        interval_distance2 = 300,
        interval_length2 = 60):
    """
    Determine the peak intensity via the integration of the peak. It relies on the time where electrolysis starts or ends. 
    Inputs:
        ms_data: 1D numpy array, GC-MS data
        time: 1D numpy array, the time data,
        start_time: float, when the electrolysis starts
        end_time: float, when the electrolysis ends
        int_start_delay: the delay of integration after electrolysis starts
        int_end_delay: the dealy of integration after electrolysis stops
        interval_distance1: baseline data collection. This indicates the time difference between starting electrolysis and the basline data. 
        interval_length1: the time period of baseline collection.
        interval_distance2: baseline data collection. This indicates the time difference between stopping electrolysis and the basline data. 
        interval_length2: the time period of baseline collection.
    Returns:
        peak_int: the average signal intensity after the integration
        figure: the figure 
    """

    # get the index of the start/end time
    start_index = abs(time - start_time).argmin() 
    end_index = abs(time - end_time).argmin()

    # get the indexes for base line fitting
    start_interval = [abs(time - (start_time - interval_distance1 - interval_length1)).argmin(),
                    abs(time - (start_time - interval_distance1)).argmin()]
    end_interval = [abs(time - (end_time + interval_distance2)).argmin(), 
                    abs(time - (end_time + interval_distance2 + interval_length2)).argmin()]


    # perform linear fitting on the collected baseline data
    x = np.concatenate((time[start_interval[0]:start_interval[1]],
                        time[end_interval[0]:end_interval[1]]))
    y = np.concatenate((ms_data[start_interval[0]:start_interval[1]],
                        ms_data[end_interval[0]:end_interval[1]]))
    results = scipy.stats.linregress(x,y)

    figure = plt.figure()
    
    plt.plot(time, ms_data, c = "black")
    plt.ylabel("signal")
    plt.xlabel("time (s)")
    # the segements that are used for linear fitting
    plt.scatter(time[start_interval],
                ms_data[start_interval], c = "red")
    
    plt.scatter(time[end_interval],
                ms_data[end_interval], c = "blue")
    # the fitted linear line
    plt.plot(time[start_interval[0]:end_interval[1]],
            results.intercept + results.slope*time[start_interval[0]:end_interval[1]],
            color = "green")
    
    # perform the integration
    int_start_index = abs(time - (start_time + int_start_delay)).argmin()
    int_end_index = abs(time - (end_time + int_end_delay)).argmin()
    print(int_start_index)
    print(int_end_index)
    x = time[int_start_index:int_end_index] 
    # perform baseline substraction
    y = ms_data[int_start_index:int_end_index] - (results.intercept + results.slope*time[int_start_index:int_end_index])
    # sanity check
    if (y<0).any():
        print("warning: after baseline subtraction, the signal is negative! adding all negative signals to 0 in the integration.")
        y[y<0] = 0
    # get the average peak signal
    peak_int = scipy.integrate.simps(y = y, x=x)/(x.max()-x.min())

    print(f"Residual of the lienar fitting: {results.rvalue**2}")
    print(f"the average peak intensity after the integration: {peak_int}")

    # get the block between start and end
    plt.fill_between(
        time[start_index:end_index],
        (results.intercept + results.slope*time[start_index:end_index]),
        ms_data[start_index:end_index],
        color = "blue",
        alpha = 0.2
    )
    # get the block of actual integration
    plt.fill_between(
        x,
        (results.intercept + results.slope*time[int_start_index:int_end_index]),
        ms_data[int_start_index:int_end_index],
        color = "red",
        alpha = 0.5
    )
    plt.show()
    plt.close()
    
    return peak_int, figure