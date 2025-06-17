import h5py
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats


def check_trailing_zeros(waveform, n_samples=100):
    """
    Check if trailing zeros exist and find where trailing zeros start,
    evaluated from the last sample backward.

    Parameters:
    waveform (array-like): The waveform to analyze
    n_samples (int): Number of trailing samples to check initially

    Returns:
    tuple: (has_trailing_zeros, trailing_zero_index) where:
           - has_trailing_zeros is True if trailing zeros exist
           - trailing_zero_index is the index where zeros start, or None if no trailing zeros
    """
    if len(waveform) < n_samples:
        return False, None

    # Calculate an adaptive threshold based on the signal amplitude
    # Use a small percentage of the max amplitude or a minimum threshold
    signal_max = np.max(np.abs(waveform))
    zero_threshold = max(1e-10, signal_max * 0.001)  # 0.1% of max or 1e-10, whichever is larger

    # First, check if we have any trailing zeros or very small values
    if not np.all(np.abs(waveform[-n_samples:]) < zero_threshold):
        return False, None

    # Start from the last sample and work backward to find the first non-zero value
    for i in range(len(waveform) - 1, -1, -1):
        if abs(waveform[i]) > zero_threshold:
            return True, i + 1  # Return the index where zeros start

    # If all values are zeros
    return True, 0


def check_small_range(waveform, threshold=1e-5):
    """
    Check if the waveform has a very small range (max-min < threshold).

    Parameters:
    waveform (array-like): The waveform to analyze
    threshold (float): Threshold for considering a range as small

    Returns:
    bool: True if the range is small, False otherwise
    """
    return np.max(waveform) - np.min(waveform) < threshold


def check_linear_trend(
    waveform, r_squared_threshold=0.95, segment_threshold=0.10, min_segment_length=300
):
    """
    Check if the waveform follows a linear trend, evaluating from the last sample.
    This function can detect both full waveform linear trends and segments with linear trends.

    Parameters:
    waveform (array-like): The waveform to analyze
    r_squared_threshold (float): Threshold for R² to consider a segment as linear
    segment_threshold (float): Minimum fraction of the waveform length that must be linear to flag as faulty
    min_segment_length (int): Minimum length of a segment to be considered for analysis

    Returns:
    bool: True if a significant portion of the waveform follows a linear trend, False otherwise
    tuple: (slope, intercept, r_squared, segment_info) of the linear fit
    """
    n_samples = len(waveform)

    # First, check if the entire waveform follows a linear trend
    x_full = np.arange(n_samples)
    slope_full, intercept_full, r_value_full, p_value, std_err = stats.linregress(x_full, waveform)
    r_squared_full = r_value_full**2

    # If the entire waveform is linear, return immediately
    if r_squared_full > r_squared_threshold:
        return True, (
            slope_full,
            intercept_full,
            r_squared_full,
            {"start": 0, "end": n_samples - 1, "length": n_samples},
        )

    # Now, check for linear segments starting from the end of the waveform
    # Try different sizes of trailing segments
    window_sizes = [min_segment_length, int(n_samples * 0.3), int(n_samples * 0.5)]

    for window_size in sorted(window_sizes, reverse=True):
        if window_size >= n_samples:
            continue

        # Start from the end and check backward
        end = n_samples
        start = max(0, end - window_size)

        x_window = np.arange(start, end)
        y_window = waveform[start:end]

        # Skip windows with very low variance (likely flat lines)
        if np.var(y_window) < 1e-10:
            continue

        slope_win, intercept_win, r_value_win, p_value, std_err = stats.linregress(
            x_window, y_window
        )
        r_squared_win = r_value_win**2

        # If window is linear and long enough
        if r_squared_win > r_squared_threshold and window_size / n_samples > segment_threshold:
            # Now find where the trend deviates from linear by moving forward from the start
            # We'll evaluate smaller chunks moving toward the beginning of the waveform
            chunk_size = min(window_size // 4, 100)
            deviation_point = start

            # If there's not enough data to check further, just return the current segment
            if start <= chunk_size:
                segment_info = {"start": start, "end": end - 1, "length": window_size}
                return True, (slope_win, intercept_win, r_squared_win, segment_info)

            # Check chunks before the start point to find where the trend deviates
            for check_start in range(start - chunk_size, 0, -chunk_size):
                check_end = min(check_start + chunk_size, start)
                x_check = np.arange(check_start, check_end)
                y_check = waveform[check_start:check_end]

                # Calculate expected values based on the linear trend
                y_expected = slope_win * x_check + intercept_win

                # Calculate mean squared error between expected and actual values
                mse = np.mean((y_check - y_expected) ** 2)

                # If the error exceeds a threshold, we've found where the trend deviates
                if mse > 1e-6:  # Adjust this threshold as needed
                    deviation_point = check_end
                    break

            segment_info = {
                "start": deviation_point,
                "end": end - 1,
                "length": end - deviation_point,
            }
            return True, (slope_win, intercept_win, r_squared_win, segment_info)

    # No significant linear trend found
    return False, (slope_full, intercept_full, r_squared_full, None)


def plot_waveform_analysis(waveform, results, title="Waveform Analysis"):
    """
    Plot the waveform and visualize analysis results.

    Parameters:
    waveform (array-like): The waveform to plot
    results (dict): Results from evaluate_waveform function
    title (str): Title for the plot
    """
    plt.figure(figsize=(12, 6))

    # Plot original waveform
    x = np.arange(len(waveform))
    plt.plot(x, waveform, label="Original Waveform", alpha=0.7)

    # Calculate an adaptive threshold based on the signal amplitude for visualization
    signal_max = np.max(np.abs(waveform))
    zero_threshold = max(1e-10, signal_max * 0.001)

    # Mark the last oscillating sample if available
    if "last_oscillating_sample" in results and results["last_oscillating_sample"] is not None:
        last_osc = results["last_oscillating_sample"]
        plt.axvline(
            x=last_osc, color="g", linestyle="--", label=f"Last Oscillating Sample ({last_osc})"
        )

        # Highlight oscillating segment
        plt.axvspan(0, last_osc, color="g", alpha=0.1, label="Oscillating Segment")

    # Create a combined view of all faults after the oscillating segment
    fault_segments = []

    # Mark trailing zeros if detected
    if results["has_trailing_zeros"] and results["trailing_zero_index"] is not None:
        trailing_idx = results["trailing_zero_index"]
        plt.axvline(
            x=trailing_idx,
            color="orange",
            linestyle="--",
            label=f"Trailing Zeros Start ({trailing_idx})",
        )
        fault_segments.append(("trailing_zero", trailing_idx, len(waveform) - 1))

    # If linear trend is detected, plot the trend line
    if results["is_linear_trend"]:
        slope, intercept, r_squared, segment_info = results["linear_params"]

        if segment_info:
            # If we have segment info, plot only the linear segment
            start = segment_info["start"]
            end = segment_info["end"]
            x_segment = np.arange(start, end + 1)
            trend_line = slope * x_segment + intercept

            # Plot the linear trend segment
            plt.plot(
                x_segment,
                trend_line,
                "r--",
                label=f"Linear Segment (R² = {r_squared:.4f})",
                linewidth=2,
            )

            # Mark start point
            plt.axvline(
                x=start, color="purple", linestyle="--", label=f"Linear Trend Start ({start})"
            )

            fault_segments.append(("linear_trend", start, end))
        else:
            # Otherwise plot the line for the entire waveform
            trend_line = slope * x + intercept
            plt.plot(
                x, trend_line, "r--", label=f"Linear Trend (R² = {r_squared:.4f})", linewidth=2
            )

    # Highlight all fault regions with appropriate colors
    fault_segments.sort(key=lambda x: x[1])  # Sort by start index

    for fault_type, start, end in fault_segments:
        if fault_type == "trailing_zero":
            plt.axvspan(start, end, color="red", alpha=0.2, label="Trailing Zeros")
        elif fault_type == "linear_trend":
            plt.axvspan(start, end, color="yellow", alpha=0.2, label="Linear Segment")

    # Plot zero threshold line
    plt.axhline(
        y=zero_threshold,
        color="cyan",
        linestyle=":",
        alpha=0.5,
        label=f"Zero Threshold ({zero_threshold:.2e})",
    )
    plt.axhline(y=-zero_threshold, color="cyan", linestyle=":", alpha=0.5)

    # Add inset plot to show detail around the transition area
    if "last_oscillating_sample" in results and results["last_oscillating_sample"] is not None:
        last_osc = results["last_oscillating_sample"]
        # Create inset for detailed view of transition

        # Define the region to zoom (expand 10% to each side)
        inset_width = min(500, len(waveform) // 4)
        inset_center = last_osc
        inset_left = max(0, inset_center - inset_width // 2)
        inset_right = min(len(waveform), inset_center + inset_width // 2)

        # Only create inset if we have enough data around the transition
        if inset_right - inset_left > 50:
            ax_inset = plt.axes([0.2, 0.2, 0.35, 0.35])  # [left, bottom, width, height]
            ax_inset.plot(x[inset_left:inset_right], waveform[inset_left:inset_right])
            ax_inset.axvline(x=last_osc, color="g", linestyle="--")

            # Add threshold lines to inset
            ax_inset.axhline(y=zero_threshold, color="cyan", linestyle=":", alpha=0.5)
            ax_inset.axhline(y=-zero_threshold, color="cyan", linestyle=":", alpha=0.5)

            # Add vertical lines for any faults that fall in this region
            for fault_type, start, end in fault_segments:
                if inset_left <= start <= inset_right:
                    color = "orange" if fault_type == "trailing_zero" else "purple"
                    ax_inset.axvline(x=start, color=color, linestyle="--")

            ax_inset.set_title("Transition Detail")
            ax_inset.grid(True, alpha=0.3)

    # Add labels and legend
    plt.xlabel("Sample")
    plt.ylabel("Amplitude")
    plt.title(title)
    # Use a smaller, condensed legend and place it at the bottom
    plt.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=3, fontsize=9)
    plt.grid(True, alpha=0.3)

    # Add text with analysis results
    fault_text = "FAULTY" if results["is_faulty"] else "NORMAL"
    fault_color = "red" if results["is_faulty"] else "green"

    fault_reasons = []
    if results["has_trailing_zeros"]:
        trailing_info = f"Trailing Zeros (start: {results['trailing_zero_index']})"
        fault_reasons.append(trailing_info)
    if results["is_linear_trend"]:
        linear_info = f"Linear Trend (start: {results['linear_trend_index']})"
        fault_reasons.append(linear_info)
    if results["has_small_range"]:
        fault_reasons.append("Small Range")

    if fault_reasons:
        fault_info = f"Status: {fault_text} ({', '.join(fault_reasons)})"
    else:
        fault_info = f"Status: {fault_text}"

    # Add summary at the top
    summary = f"Last Oscillating Sample: {results['last_oscillating_sample']}"
    plt.figtext(0.5, 0.95, summary, fontsize=12, ha="center", color="blue", weight="bold")

    # Add fault status at the bottom
    plt.figtext(0.5, 0.01, fault_info, fontsize=12, ha="center", color=fault_color, weight="bold")

    plt.tight_layout(rect=[0, 0.05, 1, 0.95])  # Adjust layout to make room for text
    plt.show()


def find_last_oscillating_sample(waveform, window_size=20, min_crossings=2):
    """
    Find the last oscillating sample in the waveform by detecting where the signal
    stops crossing zero (changing between positive and negative values).

    Parameters:
    waveform (numpy.ndarray): The waveform to analyze
    window_size (int): Size of the sliding window to check for zero crossings
    min_crossings (int): Minimum number of zero crossings to consider a window as oscillating

    Returns:
    int: Index of the last oscillating sample
    """
    n_samples = len(waveform)

    # If too short, return the middle point
    if n_samples <= window_size * 2:
        return n_samples // 2

    # Calculate an adaptive threshold based on the signal amplitude
    signal_max = np.max(np.abs(waveform))
    threshold = max(1e-10, signal_max * 0.001)  # 0.1% of max or 1e-10, whichever is larger

    # Start from the end and move backward
    for i in range(n_samples - window_size, 0, -1):
        window = waveform[i : i + window_size]

        # Check if the window has oscillations (sign changes)
        # First, eliminate values close to zero based on threshold
        window_filtered = np.where(np.abs(window) < threshold, 0, window)
        # Remove zeros to focus on actual sign changes
        window_nonzero = window_filtered[window_filtered != 0]

        if len(window_nonzero) == 0:
            continue  # Skip if all values are essentially zero

        # Count sign changes
        sign_changes = np.sum(np.diff(np.signbit(window_nonzero)) != 0)

        if sign_changes >= min_crossings:
            # Found window with sufficient oscillations
            return i + window_size - 1  # Return the last sample in this window

    # If no clear oscillations found, fall back to variance method
    # Calculate variance in sliding windows
    var_values = []
    for i in range(0, n_samples - window_size):
        window = waveform[i : i + window_size]
        var_values.append(np.var(window))

    # Convert to numpy array
    var_values = np.array(var_values)

    # Find where the variance drops significantly
    threshold = np.max(var_values) * 0.1

    # Start from 1/3 of the way through to avoid false positives at the beginning
    start_idx = len(var_values) // 3
    for i in range(len(var_values) - 1, start_idx, -1):
        if var_values[i] < threshold and var_values[i - 1] > threshold:
            # Found a drop below threshold (going backward)
            return i + window_size // 2

    # Default fallback
    return n_samples // 2


def evaluate_waveform_channel(
    waveform,
    channel_name=None,
    plot=False,
    r_squared_threshold=0.95,
    segment_threshold=0.10,
    min_segment_length=300,
    range_threshold=1e-5,
):
    """
    Evaluate a single waveform channel to identify faults.

    Parameters:
    waveform (array-like): The waveform channel to analyze
    channel_name (str): Optional name for the channel for reporting purposes
    plot (bool): Whether to plot the waveform and analysis results
    r_squared_threshold (float): Threshold for R² to consider a segment as linear
    segment_threshold (float): Minimum fraction of the waveform length that must be linear to flag as faulty
    min_segment_length (int): Minimum length of a segment to be considered for analysis
    range_threshold (float): Threshold for considering a waveform range as small

    Returns:
    dict: Dictionary with evaluation results
    """
    results = {
        "channel": channel_name,
        "has_trailing_zeros": False,
        "trailing_zero_index": None,
        "is_linear_trend": False,
        "linear_trend_index": None,
        "has_small_range": False,
        "linear_params": None,
        "is_faulty": False,
        "last_oscillating_sample": None,
    }

    # Find the last oscillating sample based on zero crossings
    last_osc = find_last_oscillating_sample(waveform)
    results["last_oscillating_sample"] = last_osc

    # Check for trailing zeros
    has_trailing_zeros, trailing_zero_index = check_trailing_zeros(waveform)
    results["has_trailing_zeros"] = has_trailing_zeros
    results["trailing_zero_index"] = trailing_zero_index

    # Check for small range (treat as trailing zero fault)
    results["has_small_range"] = check_small_range(waveform, threshold=range_threshold)

    # Check for linear trend
    is_linear, linear_params = check_linear_trend(
        waveform,
        r_squared_threshold=r_squared_threshold,
        segment_threshold=segment_threshold,
        min_segment_length=min_segment_length,
    )

    results["is_linear_trend"] = is_linear
    results["linear_params"] = linear_params

    # Store linear trend start index if available
    if is_linear and linear_params[3] is not None:  # linear_params[3] is segment_info
        results["linear_trend_index"] = linear_params[3]["start"]

    # Determine if the waveform is faulty
    results["is_faulty"] = (
        results["has_trailing_zeros"] or results["is_linear_trend"] or results["has_small_range"]
    )

    # Compare the last oscillating sample with fault indices
    # If the detected faults occur earlier than our oscillation detection,
    # use the earlier of the two
    fault_indices = []
    if results["trailing_zero_index"] is not None:
        fault_indices.append(results["trailing_zero_index"])
    if results["linear_trend_index"] is not None:
        fault_indices.append(results["linear_trend_index"])

    if fault_indices:
        first_fault_index = min(fault_indices)
        # If the first fault occurs before our detected last oscillation,
        # update the last oscillating sample
        if first_fault_index <= results["last_oscillating_sample"]:
            results["last_oscillating_sample"] = max(0, first_fault_index - 1)

    # Plot the waveform if requested
    if plot:
        plot_title = f"Channel: {channel_name}" if channel_name else "Waveform Analysis"
        plot_waveform_analysis(waveform, results, title=plot_title)

    return results


def evaluate_multi_channel_waveform(
    waveform_data,
    plot=False,
    r_squared_threshold=0.95,
    segment_threshold=0.10,
    min_segment_length=300,
    range_threshold=1e-5,
):
    """
    Evaluate a multi-channel waveform to identify faults in each channel.

    Parameters:
    waveform_data (numpy.ndarray): Array with shape [n_samples, n_channels]
    plot (bool): Whether to plot the analysis results for each channel
    r_squared_threshold (float): Threshold for R² to consider a segment as linear
    segment_threshold (float): Minimum fraction of the waveform length that must be linear to flag as faulty
    min_segment_length (int): Minimum length of a segment to be considered for analysis
    range_threshold (float): Threshold for considering a waveform range as small

    Returns:
    list: List of dictionaries with evaluation results for each channel
    bool: True if any channel is faulty, False otherwise
    """
    if waveform_data.ndim != 2:
        raise ValueError("Expected 2D array with shape [n_samples, n_channels]")

    n_samples, n_channels = waveform_data.shape

    # Process each channel separately
    results = []
    any_faulty = False

    for i in range(n_channels):
        channel_data = waveform_data[:, i]
        channel_name = f"Channel {i+1}"
        channel_results = evaluate_waveform_channel(
            channel_data,
            channel_name=channel_name,
            plot=plot,
            r_squared_threshold=r_squared_threshold,
            segment_threshold=segment_threshold,
            min_segment_length=min_segment_length,
            range_threshold=range_threshold,
        )

        results.append(channel_results)

        if channel_results["is_faulty"]:
            any_faulty = True

    return results, any_faulty


def print_largest_last_oscillating_sample(waveform_data, plot=False):
    """
    Process a waveform with 3 channels, considering non-zero but flat signals as trailing zeros,
    and print the largest index of the last oscillating sample among all channels.

    Parameters:
    waveform_data (numpy.ndarray): Array with shape [n_samples, 3]
    plot (bool): Whether to plot visualization of the analysis results

    Returns:
    int: Largest last oscillating sample index
    """
    if waveform_data.ndim != 2 or waveform_data.shape[1] != 3:
        raise ValueError("Expected 2D array with shape [n_samples, 3]")

    # Process each channel
    channel_results, _ = evaluate_multi_channel_waveform(waveform_data, plot=plot)

    # Adjust last oscillating sample for non-zero but flat channels
    for i, result in enumerate(channel_results):
        channel_data = waveform_data[:, i]

        # Check if the signal is flat but non-zero
        signal_max = np.max(np.abs(channel_data))
        threshold = max(1e-10, signal_max * 0.001)  # 0.1% of max or 1e-10, whichever is larger

        is_flat = result["has_small_range"]
        mean_value = np.mean(channel_data)
        is_nonzero = abs(mean_value) > threshold

        # If it's flat but non-zero, set last oscillating sample to 0
        if is_flat and is_nonzero:
            result["last_oscillating_sample"] = 0

            # Update the plot if enabled
            if plot:
                plot_waveform_analysis(
                    waveform_data[:, i], result, title=f"Channel {i+1} - Adjusted to Trailing Zeros"
                )

    # Find the largest last oscillating sample index among all channels
    largest_index = 0
    for result in channel_results:
        if result["last_oscillating_sample"] is not None:
            largest_index = max(largest_index, result["last_oscillating_sample"])

    # Just print the largest index (no return value needed)
    print(largest_index)

    return largest_index  # Still return it for internal use


def process_batch_largest_oscillating_samples(waveforms):
    """
    Process a batch of waveforms, each with 3 channels, and print the largest last
    oscillating sample index for each waveform.

    Parameters:
    waveforms (numpy.ndarray): Array with shape [num, n_samples, 3]
    """
    if waveforms.ndim != 3 or waveforms.shape[2] != 3:
        raise ValueError("Expected 3D array with shape [num, n_samples, 3]")

    num_waveforms = waveforms.shape[0]

    for i in range(num_waveforms):
        waveform = waveforms[i]
        index = print_largest_last_oscillating_sample(waveform)
        print(index)


# Main function to process waveforms and print only the largest last oscillating sample
def print_largest_last_oscillating_sample(waveform_data, plot=False):
    """
    Process a waveform with 3 channels, considering non-zero but flat signals as trailing zeros,
    and print the largest index of the last oscillating sample among all channels.
    Also return information about the status (faulty/normal) and fault causes.

    Parameters:
    waveform_data (numpy.ndarray): Array with shape [n_samples, 3]
    plot (bool): Whether to plot visualization of the analysis results

    Returns:
    tuple: (largest_index, is_faulty, has_trailing_zeros, has_linear_trend, has_small_range)
    """
    if waveform_data.ndim != 2 or waveform_data.shape[1] != 3:
        raise ValueError("Expected 2D array with shape [n_samples, 3]")

    # Process each channel
    channel_results, any_faulty = evaluate_multi_channel_waveform(waveform_data, plot=plot)

    # Initialize fault flags
    has_trailing_zeros = False
    has_linear_trend = False
    has_small_range = False

    # Adjust last oscillating sample for non-zero but flat channels
    for i, result in enumerate(channel_results):
        channel_data = waveform_data[:, i]

        # Check if the signal is flat but non-zero
        signal_max = np.max(np.abs(channel_data))
        threshold = max(1e-10, signal_max * 0.001)  # 0.1% of max or 1e-10, whichever is larger

        is_flat = result["has_small_range"]
        mean_value = np.mean(channel_data)
        is_nonzero = abs(mean_value) > threshold

        # If it's flat but non-zero, set last oscillating sample to 0
        if is_flat and is_nonzero:
            result["last_oscillating_sample"] = 0

            # Update the plot if enabled
            if plot:
                plot_waveform_analysis(
                    waveform_data[:, i], result, title=f"Channel {i+1} - Adjusted to Trailing Zeros"
                )

        # Collect fault information
        if result["has_trailing_zeros"]:
            has_trailing_zeros = True
        if result["is_linear_trend"]:
            has_linear_trend = True
        if result["has_small_range"]:
            has_small_range = True

    # Find the largest last oscillating sample index among all channels
    largest_index = 0
    for result in channel_results:
        if result["last_oscillating_sample"] is not None:
            largest_index = max(largest_index, result["last_oscillating_sample"])

    # Just print the largest index
    # print(largest_index)

    # Return index and status information
    return largest_index, any_faulty, has_trailing_zeros, has_linear_trend, has_small_range


def process_waveforms(waveforms_data, plot=False, save_to_file=None):
    """
    Process waveforms with shape [num, n_samples, 3] and print only the largest
    last oscillating sample for each waveform. Also saves the mapping between
    waveform number and status information.

    Parameters:
    waveforms_data (numpy.ndarray): Array with shape [num, n_samples, 3]
    plot (bool): Whether to plot visualization of the analysis results
    save_to_file (str): Optional path to save the results to a file

    Returns:
    list: List of dictionaries with results for each waveform
    """
    if waveforms_data.ndim != 3 or waveforms_data.shape[2] != 3:
        raise ValueError("Expected waveform data with shape [num, n_samples, 3]")

    num_waveforms = waveforms_data.shape[0]

    # List to store results for each waveform
    results = []

    for i in range(num_waveforms):
        # Get the current waveform
        waveform = waveforms_data[i]

        # Find and print the largest last oscillating sample and status
        largest_index, is_faulty, has_trailing_zeros, has_linear_trend, has_small_range = (
            print_largest_last_oscillating_sample(waveform, plot=plot)
        )

        # Create result dictionary with all information
        result = {
            "waveform_number": i,
            "largest_last_oscillating_sample": largest_index,
            "is_faulty": is_faulty,
            "has_trailing_zeros": has_trailing_zeros,
            "has_linear_trend": has_linear_trend,
            "has_small_range": has_small_range,
        }

        # Also print status information
        status = "FAULTY" if is_faulty else "NORMAL"
        fault_reasons = []
        if has_trailing_zeros:
            fault_reasons.append("trailing zeros")
        if has_linear_trend:
            fault_reasons.append("linear trend")
        if has_small_range:
            fault_reasons.append("small range (flat non-zero)")

        status_str = f"Status: {status}"
        if fault_reasons:
            status_str += f" caused by {', '.join(fault_reasons)}"

        # print(status_str)

        results.append(result)

    # Save results to file if specified
    if save_to_file:
        try:
            with open(save_to_file, "w") as f:
                # Write header
                f.write(
                    "waveform_number,largest_last_oscillating_sample,is_faulty,has_trailing_zeros,has_linear_trend,has_small_range\n"
                )

                # Write data for each waveform
                for result in results:
                    f.write(
                        f"{result['waveform_number']},{result['largest_last_oscillating_sample']},"
                        f"{str(result['is_faulty']).lower()},{str(result['has_trailing_zeros']).lower()},"
                        f"{str(result['has_linear_trend']).lower()},{str(result['has_small_range']).lower()}\n"
                    )

            print(f"Results saved to {save_to_file}")
        except Exception as e:
            print(f"Error saving results to file: {e}")

    return results


h5_file_path = "raw_waveforms_filtered.h5"

# Dictionary to hold each variable as a NumPy array
data_dict = {}

# Open the HDF5 file in read-only mode
with h5py.File(h5_file_path, "r") as file:
    # List all available keys (datasets)
    keys = list(file.keys())
    print("Available keys:", keys)

    # Loop over the provided keys and save each dataset to a dictionary as a NumPy array
    for key in keys:
        data_dict[key] = np.array(file[key])
        print(f"Loaded '{key}' with shape {data_dict[key].shape}")

azimuth = data_dict["azimuth"]
azimuthal_gap = data_dict["azimuthal_gap"]
back_azimuth = data_dict["back_azimuth"]
event_ID = data_dict["event_ID"]
hypocentral_distance = data_dict["hypocentral_distance"]
hypocentre_depth = data_dict["hypocentre_depth"]
hypocentre_dip = data_dict["hypocentre_dip"]
hypocentre_latitude = data_dict["hypocentre_latitude"]
hypocentre_longitude = data_dict["hypocentre_longitude"]
hypocentre_rake = data_dict["hypocentre_rake"]
hypocentre_strike = data_dict["hypocentre_strike"]
is_onshore = data_dict["is_onshore"]
magnitude = data_dict["magnitude"]
station_latitude = data_dict["station_latitude"]
station_longitude = data_dict["station_longitude"]
station_name = data_dict["station_name"]
station_network = data_dict["station_network"]
trace_sampling_rate = data_dict["trace_sampling_rate"]
waveforms = data_dict["waveforms"]
vs30 = data_dict["vs30"]
z_filename = data_dict["z_filename"]

indices_map_all = process_waveforms(waveforms, plot=False, save_to_file="results.csv")

indices = []
for i in range(len(indices_map_all)):
    indices.append(indices_map_all[i]["largest_last_oscillating_sample"])

indices = np.asarray(indices)

## Save in h5 format

file_path = "raw_waveforms_filtered_indices.h5"
with h5py.File(file_path, "w") as h5f:
    h5f.create_dataset("event_ID", data=event_ID)
    h5f.create_dataset("hypocentral_distance", data=hypocentral_distance * 1e-3)
    h5f.create_dataset("hypocentre_depth", data=hypocentre_depth)
    h5f.create_dataset("hypocentre_latitude", data=hypocentre_latitude)
    h5f.create_dataset("hypocentre_longitude", data=hypocentre_longitude)
    h5f.create_dataset("is_onshore", data=is_onshore)
    h5f.create_dataset("magnitude", data=magnitude)
    h5f.create_dataset("station_latitude", data=station_latitude)
    h5f.create_dataset("station_longitude", data=station_longitude)
    h5f.create_dataset("station_name", data=station_name)
    h5f.create_dataset("station_network", data=station_network)
    h5f.create_dataset("azimuth", data=azimuth)
    h5f.create_dataset("back_azimuth", data=back_azimuth)
    h5f.create_dataset("azimuthal_gap", data=azimuthal_gap)
    h5f.create_dataset("hypocentre_strike", data=hypocentre_strike)
    h5f.create_dataset("hypocentre_dip", data=hypocentre_dip)
    h5f.create_dataset("hypocentre_rake", data=hypocentre_rake)
    h5f.create_dataset("trace_sampling_rate", data=trace_sampling_rate)
    h5f.create_dataset("vs30", data=vs30)
    h5f.create_dataset("z_filename", data=z_filename)
    h5f.create_dataset("indices_valid_waveforms", data=indices)
    h5f.create_dataset("waveforms", data=waveforms)

print(f"\nHDF5 file '{file_path}' created successfully.")
