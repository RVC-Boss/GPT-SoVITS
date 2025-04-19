import numpy as np

"""
Essentially returns the index of the middle of the zero zone + the starting index.
So if the starting index was 0 and we found the zero zone to be from 12789:12800,
then we would be returning 0 + 12795 or 12795 (since the window was of size 11 and midpoint is 6)
This method works by using a sliding window mechanic on each chunk, where we
slide the window from the end going to the start. If all the values in the window
meet the threshold, then we assign this as the zero zone.
TLDR: Returns the zero zone where a region in the audio has enough silence.
"""
def find_zero_zone(chunk, start_index, search_length, search_window_size=11):
    zone = chunk[start_index:start_index + search_length]
    # print(f"Zero-crossing search zone: Start={start_index}, Length={len(zone)}")

    zero_threshold = 1.0e-4
    # Check for y consecutive zeros
    for idx in range(len(zone), -1 + search_window_size, -1):
        index_to_start = idx-search_window_size
        abs_zone = np.abs(zone[index_to_start:idx])
        if np.all(abs_zone < zero_threshold):
            # print(f"Found Abs Zone: {abs_zone}")
            # print(f"Extended Abs Zone: {chunk[idx-21:idx+10]}")
            index_midpoint = index_to_start + int(search_window_size // 2)
            # print(f"Returning {start_index} + {index_midpoint}")
            return (start_index + index_midpoint), None
    
    # print("Falling back to zero crossing due to no zero zone found.  You may hear more prominent pops and clicks in the audio.  Try increasing search length or cumulative tokens.")
    return find_zero_crossing(chunk, start_index, search_length)

def find_zero_crossing(chunk, start_index, search_length):
    # If the model is falling back on the this function, it might be a bad indicator that the search length is too low
    
    zone = chunk[start_index:start_index + search_length]
    sign_changes = np.where(np.diff(np.sign(zone)) != 0)[0]
    
    if len(sign_changes) == 0:
        raise ("No zero-crossings found in this zone. This should not be happening, debugging time.")
    else:
        zc_index = start_index + sign_changes[0] + 1
        # print(f"Zero-crossing found at index {zc_index}")
        # Determine the crossing direction in chunk1
        prev_value = chunk[zc_index - 1]
        curr_value = chunk[zc_index]
        crossing_direction = np.sign(curr_value) - np.sign(prev_value)
        # print(f"Crossing direction in chunk1: {np.sign(prev_value)} to {np.sign(curr_value)}")
        return zc_index, crossing_direction

def find_matching_index(chunk, center_index, max_offset, crossing_direction):
    """
    Finds a zero-crossing in data that matches the specified crossing direction,
    starting from center_index and searching outward.
    """
    if crossing_direction == None:
        return center_index # if zero zone
    
    # fall back for zero_crossing
    data_length = len(chunk)
    # print(f"Center index in chunk2: {center_index}")
    for offset in range(max_offset + 1):
        # Check index bounds
        idx_forward = center_index + offset
        idx_backward = center_index - offset

        # Check forward direction
        if idx_forward < data_length - 1:
            prev_sign = np.sign(chunk[idx_forward])
            curr_sign = np.sign(chunk[idx_forward + 1])
            direction = curr_sign - prev_sign
            if direction == crossing_direction:
                # print(f"Matching zero-crossing found at index {idx_forward + 1} (forward)")
                return idx_forward + 1

        # Check backward direction
        if idx_backward > 0:
            prev_sign = np.sign(chunk[idx_backward - 1])
            curr_sign = np.sign(chunk[idx_backward])
            direction = curr_sign - prev_sign
            if direction == crossing_direction:
                # print(f"Matching zero-crossing found at index {idx_backward} (backward)")
                return idx_backward

    # print("No matching zero-crossings found in this zone.")
    return None