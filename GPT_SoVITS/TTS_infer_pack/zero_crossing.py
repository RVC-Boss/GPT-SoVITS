import numpy as np
import wave
import struct

def read_wav_file(filename):
    """
    Reads a WAV file and returns the sample rate and data as a numpy array.
    """
    with wave.open(filename, 'rb') as wf:
        sample_rate = wf.getframerate()
        n_frames = wf.getnframes()
        sample_width = wf.getsampwidth()
        n_channels = wf.getnchannels()

        audio_data = wf.readframes(n_frames)
        # Determine the format string for struct unpacking
        fmt = "<" + {1:'b', 2:'h', 4:'i'}[sample_width] * n_frames * n_channels
        audio_samples = struct.unpack(fmt, audio_data)
        audio_array = np.array(audio_samples, dtype=int)

        # If stereo, reshape the array
        if n_channels > 1:
            audio_array = audio_array.reshape(-1, n_channels)
        return sample_rate, audio_array, sample_width, n_channels

def write_wav_file(filename, sample_rate, data, sample_width, n_channels):
    """
    Writes numpy array data to a WAV file.
    """
    with wave.open(filename, 'wb') as wf:
        wf.setnchannels(n_channels)
        wf.setsampwidth(sample_width)
        wf.setframerate(sample_rate)
        # Flatten the array if it's multi-dimensional
        if data.ndim > 1:
            data = data.flatten()
        # Pack the data into bytes
        fmt = "<" + {1:'b', 2:'h', 4:'i'}[sample_width] * len(data)
        byte_data = struct.pack(fmt, *data)
        wf.writeframes(byte_data)
        
def find_zero_zone(chunk, start_index, search_length, num_zeroes=11):
    zone = chunk[start_index:start_index + search_length]
    print(f"Zero-crossing search zone: Start={start_index}, Length={len(zone)}")

    zero_threshold = 1.0e-4
    # Check for y consecutive zeros
    for idx in range(len(zone), -1 + num_zeroes, -1):
        index_to_start = idx-num_zeroes
        abs_zone = np.abs(zone[index_to_start:idx])
        if np.all(abs_zone < zero_threshold):
            index_midpoint = index_to_start + int(num_zeroes // 2)
            return (start_index + index_midpoint), None
    
    print("Falling back to zero crossing due to no zero zone found.  You may hear more prominent pops and clicks in the audio.  Try increasing search length or cumulative tokens.")
    return find_zero_crossing(chunk, start_index, search_length)

def find_zero_crossing(chunk, start_index, search_length):
    # If the model is falling back on the this function, it might be a bad indicator that the search length is too low
    
    zone = chunk[start_index:start_index + search_length]
    sign_changes = np.where(np.diff(np.sign(zone)) != 0)[0]
    
    if len(sign_changes) == 0:
        raise ("No zero-crossings found in this zone. This should not be happening, debugging time.")
    else:
        zc_index = start_index + sign_changes[0] + 1
        print(f"Zero-crossing found at index {zc_index}")
        # Determine the crossing direction in chunk1
        prev_value = chunk[zc_index - 1]
        curr_value = chunk[zc_index]
        crossing_direction = np.sign(curr_value) - np.sign(prev_value)
        print(f"Crossing direction in chunk1: {np.sign(prev_value)} to {np.sign(curr_value)}")
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
    print(f"Center index in chunk2: {center_index}")
    for offset in range(max_offset + 1):
        # Check index bounds
        idx_forward = center_index + offset
        idx_backward = center_index - offset
        found = False

        # Check forward direction
        if idx_forward < data_length - 1:
            prev_sign = np.sign(chunk[idx_forward])
            curr_sign = np.sign(chunk[idx_forward + 1])
            direction = curr_sign - prev_sign
            if direction == crossing_direction:
                print(f"Matching zero-crossing found at index {idx_forward + 1} (forward)")
                return idx_forward + 1

        # Check backward direction
        if idx_backward > 0:
            prev_sign = np.sign(chunk[idx_backward - 1])
            curr_sign = np.sign(chunk[idx_backward])
            direction = curr_sign - prev_sign
            if direction == crossing_direction:
                print(f"Matching zero-crossing found at index {idx_backward} (backward)")
                return idx_backward

    print("No matching zero-crossings found in this zone.")
    return None

# legacy, just for history.  delete me sometime
def splice_chunks(chunk1, chunk2, search_length, y):
    """
    Splices two audio chunks at zero-crossing points.
    """
    # Define the zone to search in chunk1
    start_index1 = len(chunk1) - search_length
    if start_index1 < 0:
        start_index1 = 0
        search_length = len(chunk1)
    print(f"Searching for zero-crossing in chunk1 from index {start_index1} to {len(chunk1)}")
    # Find zero-crossing in chunk1
    zc_index1, crossing_direction = find_zero_crossing(chunk1, start_index1, search_length, y)
    if zc_index1 is None:
        print("No zero-crossing found in chunk1 within the specified zone.")
        return None
    
    # Define the zone to search in chunk2 near the same index
    # Since chunk2 overlaps with chunk1, we can assume that index positions correspond
    # Adjusted search in chunk2
    # You can adjust this value if needed
    center_index = zc_index1  # Assuming alignment between chunk1 and chunk2
    max_offset = search_length

    # Ensure center_index is within bounds
    if center_index < 0:
        center_index = 0
    elif center_index >= len(chunk2):
        center_index = len(chunk2) - 1

    print(f"Searching for matching zero-crossing in chunk2 around index {center_index} with max offset {max_offset}")

    zc_index2 = find_matching_zero_crossing(chunk2, center_index, max_offset, crossing_direction)

    if zc_index2 is None:
        print("No matching zero-crossing found in chunk2.")
        return None

    print(f"Zero-crossing in chunk1 at index {zc_index1}, chunk2 at index {zc_index2}")
    # Splice the chunks
    new_chunk = np.concatenate((chunk1[:zc_index1], chunk2[zc_index2:]))
    print(f"Spliced chunk length: {len(new_chunk)}")
    return new_chunk

# legacy, just for history.  delete me sometime
def process_audio_chunks(filenames, sample_rate, x, y, output_filename):
    """
    Processes and splices a list of audio chunks.
    """
    # Read the first chunk
    sr, chunk_data, sample_width, n_channels = read_wav_file(filenames[0])
    if sr != sample_rate:
        print(f"Sample rate mismatch in {filenames[0]}")
        return
    print(f"Processing {filenames[0]}")
    # Initialize the combined audio with the first chunk
    combined_audio = chunk_data
    # Process remaining chunks
    for filename in filenames[1:]:
        sr, next_chunk_data, _, _ = read_wav_file(filename)
        if sr != sample_rate:
            print(f"Sample rate mismatch in {filename}")
            return
        print(f"Processing {filename}")
        # Splice the current combined audio with the next chunk
        new_combined = splice_chunks(combined_audio, next_chunk_data, x, y)
        if new_combined is None:
            print(f"Failed to splice chunks between {filename} and previous chunk.")
            return
        combined_audio = new_combined
    # Write the final combined audio to output file
    write_wav_file(output_filename, sample_rate, combined_audio, sample_width, n_channels)
    print(f"Final audio saved to {output_filename}")

# Main execution
if __name__ == "__main__":
    # User-specified parameters
    sample_rate = 32000  # Sample rate in Hz
    x = 500            # Number of frames to search from the end of the chunk
    y = 10               # Number of consecutive zeros to look for
    output_filename = "combined_output.wav"
    folder_with_chunks = "output_chunks"
    import os
    def absolute_file_paths(directory):
        path = os.path.abspath(directory)
        return [entry.path for entry in os.scandir(path) if entry.is_file()]
    # List of input audio chunk filenames in sequential order
    filenames = absolute_file_paths(folder_with_chunks)
    # Process and splice the audio chunks
    process_audio_chunks(filenames, sample_rate, x, y, output_filename)
