import math
from typing import Tuple

import torch
import torchaudio
from torch import Tensor

__all__ = [
    "get_mel_banks",
    "inverse_mel_scale",
    "inverse_mel_scale_scalar",
    "mel_scale",
    "mel_scale_scalar",
    "spectrogram",
    "fbank",
    "fbank_onnx"
    "mfcc",
    "vtln_warp_freq",
    "vtln_warp_mel_freq",
]

# numeric_limits<float>::epsilon() 1.1920928955078125e-07
EPSILON = torch.tensor(torch.finfo(torch.float).eps)
# 1 milliseconds = 0.001 seconds
MILLISECONDS_TO_SECONDS = 0.001

# window types
HAMMING = "hamming"
HANNING = "hanning"
POVEY = "povey"
RECTANGULAR = "rectangular"
BLACKMAN = "blackman"
WINDOWS = [HAMMING, HANNING, POVEY, RECTANGULAR, BLACKMAN]


def _get_epsilon(device, dtype):
    return EPSILON.to(device=device, dtype=dtype)


def _next_power_of_2(x: int) -> int:
    r"""Returns the smallest power of 2 that is greater than x"""
    return 1 if x == 0 else 2 ** (x - 1).bit_length()


def _get_strided(waveform: Tensor, window_size: int, window_shift: int, snip_edges: bool) -> Tensor:
    r"""Given a waveform (1D tensor of size ``num_samples``), it returns a 2D tensor (m, ``window_size``)
    representing how the window is shifted along the waveform. Each row is a frame.

    Args:
        waveform (Tensor): Tensor of size ``num_samples``
        window_size (int): Frame length
        window_shift (int): Frame shift
        snip_edges (bool): If True, end effects will be handled by outputting only frames that completely fit
            in the file, and the number of frames depends on the frame_length.  If False, the number of frames
            depends only on the frame_shift, and we reflect the data at the ends.

    Returns:
        Tensor: 2D tensor of size (m, ``window_size``) where each row is a frame
    """
    assert waveform.dim() == 1
    num_samples = waveform.size(0)
    strides = (window_shift * waveform.stride(0), waveform.stride(0))

    if snip_edges:
        if num_samples < window_size:
            return torch.empty((0, 0), dtype=waveform.dtype, device=waveform.device)
        else:
            m = 1 + (num_samples - window_size) // window_shift
    else:
        reversed_waveform = torch.flip(waveform, [0])
        m = (num_samples + (window_shift // 2)) // window_shift
        pad = window_size // 2 - window_shift // 2
        pad_right = reversed_waveform
        if pad > 0:
            # torch.nn.functional.pad returns [2,1,0,1,2] for 'reflect'
            # but we want [2, 1, 0, 0, 1, 2]
            pad_left = reversed_waveform[-pad:]
            waveform = torch.cat((pad_left, waveform, pad_right), dim=0)
        else:
            # pad is negative so we want to trim the waveform at the front
            waveform = torch.cat((waveform[-pad:], pad_right), dim=0)

    sizes = (m, window_size)
    return waveform.as_strided(sizes, strides)


def _feature_window_function(
    window_type: str,
    window_size: int,
    blackman_coeff: float,
    device: torch.device,
    dtype: int,
) -> Tensor:
    r"""Returns a window function with the given type and size"""
    if window_type == HANNING:
        return torch.hann_window(window_size, periodic=False, device=device, dtype=dtype)
    elif window_type == HAMMING:
        return torch.hamming_window(window_size, periodic=False, alpha=0.54, beta=0.46, device=device, dtype=dtype)
    elif window_type == POVEY:
        # like hanning but goes to zero at edges
        return torch.hann_window(window_size, periodic=False, device=device, dtype=dtype).pow(0.85)
    elif window_type == RECTANGULAR:
        return torch.ones(window_size, device=device, dtype=dtype)
    elif window_type == BLACKMAN:
        a = 2 * math.pi / (window_size - 1)
        window_function = torch.arange(window_size, device=device, dtype=dtype)
        # can't use torch.blackman_window as they use different coefficients
        return (
            blackman_coeff
            - 0.5 * torch.cos(a * window_function)
            + (0.5 - blackman_coeff) * torch.cos(2 * a * window_function)
        ).to(device=device, dtype=dtype)
    else:
        raise Exception("Invalid window type " + window_type)


def _get_log_energy(strided_input: Tensor, epsilon: Tensor, energy_floor: float) -> Tensor:
    r"""Returns the log energy of size (m) for a strided_input (m,*)"""
    device, dtype = strided_input.device, strided_input.dtype
    log_energy = torch.max(strided_input.pow(2).sum(1), epsilon).log()  # size (m)
    if energy_floor == 0.0:
        return log_energy
    return torch.max(log_energy, torch.tensor(math.log(energy_floor), device=device, dtype=dtype))


def _get_waveform_and_window_properties(
    waveform: Tensor,
    channel: int,
    sample_frequency: float,
    frame_shift: float,
    frame_length: float,
    round_to_power_of_two: bool,
    preemphasis_coefficient: float,
) -> Tuple[Tensor, int, int, int]:
    r"""Gets the waveform and window properties"""
    channel = max(channel, 0)
    assert channel < waveform.size(0), "Invalid channel {} for size {}".format(channel, waveform.size(0))
    waveform = waveform[channel, :]  # size (n)
    window_shift = int(sample_frequency * frame_shift * MILLISECONDS_TO_SECONDS)
    window_size = int(sample_frequency * frame_length * MILLISECONDS_TO_SECONDS)
    padded_window_size = _next_power_of_2(window_size) if round_to_power_of_two else window_size

    assert 2 <= window_size <= len(waveform), "choose a window size {} that is [2, {}]".format(
        window_size, len(waveform)
    )
    assert 0 < window_shift, "`window_shift` must be greater than 0"
    assert padded_window_size % 2 == 0, (
        "the padded `window_size` must be divisible by two. use `round_to_power_of_two` or change `frame_length`"
    )
    assert 0.0 <= preemphasis_coefficient <= 1.0, "`preemphasis_coefficient` must be between [0,1]"
    assert sample_frequency > 0, "`sample_frequency` must be greater than zero"
    return waveform, window_shift, window_size, padded_window_size


def _get_window(
    waveform: Tensor,
    padded_window_size: int,
    window_size: int,
    window_shift: int,
    window_type: str,
    blackman_coeff: float,
    snip_edges: bool,
    raw_energy: bool,
    energy_floor: float,
    dither: float,
    remove_dc_offset: bool,
    preemphasis_coefficient: float,
) -> Tuple[Tensor, Tensor]:
    r"""Gets a window and its log energy

    Returns:
        (Tensor, Tensor): strided_input of size (m, ``padded_window_size``) and signal_log_energy of size (m)
    """
    device, dtype = waveform.device, waveform.dtype
    epsilon = _get_epsilon(device, dtype)

    # size (m, window_size)
    strided_input = _get_strided(waveform, window_size, window_shift, snip_edges)

    if dither != 0.0:
        rand_gauss = torch.randn(strided_input.shape, device=device, dtype=dtype)
        strided_input = strided_input + rand_gauss * dither

    if remove_dc_offset:
        # Subtract each row/frame by its mean
        row_means = torch.mean(strided_input, dim=1).unsqueeze(1)  # size (m, 1)
        strided_input = strided_input - row_means

    if raw_energy:
        # Compute the log energy of each row/frame before applying preemphasis and
        # window function
        signal_log_energy = _get_log_energy(strided_input, epsilon, energy_floor)  # size (m)

    if preemphasis_coefficient != 0.0:
        # strided_input[i,j] -= preemphasis_coefficient * strided_input[i, max(0, j-1)] for all i,j
        offset_strided_input = torch.nn.functional.pad(strided_input.unsqueeze(0), (1, 0), mode="replicate").squeeze(
            0
        )  # size (m, window_size + 1)
        strided_input = strided_input - preemphasis_coefficient * offset_strided_input[:, :-1]

    # Apply window_function to each row/frame
    window_function = _feature_window_function(window_type, window_size, blackman_coeff, device, dtype).unsqueeze(
        0
    )  # size (1, window_size)
    strided_input = strided_input * window_function  # size (m, window_size)

    # Pad columns with zero until we reach size (m, padded_window_size)
    if padded_window_size != window_size:
        padding_right = padded_window_size - window_size
        strided_input = torch.nn.functional.pad(
            strided_input.unsqueeze(0), (0, padding_right), mode="constant", value=0
        ).squeeze(0)

    # Compute energy after window function (not the raw one)
    if not raw_energy:
        signal_log_energy = _get_log_energy(strided_input, epsilon, energy_floor)  # size (m)

    return strided_input, signal_log_energy


def _subtract_column_mean(tensor: Tensor, subtract_mean: bool) -> Tensor:
    # subtracts the column mean of the tensor size (m, n) if subtract_mean=True
    # it returns size (m, n)
    if subtract_mean:
        col_means = torch.mean(tensor, dim=0).unsqueeze(0)
        tensor = tensor - col_means
    return tensor


def spectrogram(
    waveform: Tensor,
    blackman_coeff: float = 0.42,
    channel: int = -1,
    dither: float = 0.0,
    energy_floor: float = 1.0,
    frame_length: float = 25.0,
    frame_shift: float = 10.0,
    min_duration: float = 0.0,
    preemphasis_coefficient: float = 0.97,
    raw_energy: bool = True,
    remove_dc_offset: bool = True,
    round_to_power_of_two: bool = True,
    sample_frequency: float = 16000.0,
    snip_edges: bool = True,
    subtract_mean: bool = False,
    window_type: str = POVEY,
) -> Tensor:
    r"""Create a spectrogram from a raw audio signal. This matches the input/output of Kaldi's
    compute-spectrogram-feats.

    Args:
        waveform (Tensor): Tensor of audio of size (c, n) where c is in the range [0,2)
        blackman_coeff (float, optional): Constant coefficient for generalized Blackman window. (Default: ``0.42``)
        channel (int, optional): Channel to extract (-1 -> expect mono, 0 -> left, 1 -> right) (Default: ``-1``)
        dither (float, optional): Dithering constant (0.0 means no dither). If you turn this off, you should set
            the energy_floor option, e.g. to 1.0 or 0.1 (Default: ``0.0``)
        energy_floor (float, optional): Floor on energy (absolute, not relative) in Spectrogram computation.  Caution:
            this floor is applied to the zeroth component, representing the total signal energy.  The floor on the
            individual spectrogram elements is fixed at std::numeric_limits<float>::epsilon(). (Default: ``1.0``)
        frame_length (float, optional): Frame length in milliseconds (Default: ``25.0``)
        frame_shift (float, optional): Frame shift in milliseconds (Default: ``10.0``)
        min_duration (float, optional): Minimum duration of segments to process (in seconds). (Default: ``0.0``)
        preemphasis_coefficient (float, optional): Coefficient for use in signal preemphasis (Default: ``0.97``)
        raw_energy (bool, optional): If True, compute energy before preemphasis and windowing (Default: ``True``)
        remove_dc_offset (bool, optional): Subtract mean from waveform on each frame (Default: ``True``)
        round_to_power_of_two (bool, optional): If True, round window size to power of two by zero-padding input
            to FFT. (Default: ``True``)
        sample_frequency (float, optional): Waveform data sample frequency (must match the waveform file, if
            specified there) (Default: ``16000.0``)
        snip_edges (bool, optional): If True, end effects will be handled by outputting only frames that completely fit
            in the file, and the number of frames depends on the frame_length.  If False, the number of frames
            depends only on the frame_shift, and we reflect the data at the ends. (Default: ``True``)
        subtract_mean (bool, optional): Subtract mean of each feature file [CMS]; not recommended to do
            it this way.  (Default: ``False``)
        window_type (str, optional): Type of window ('hamming'|'hanning'|'povey'|'rectangular'|'blackman')
         (Default: ``'povey'``)

    Returns:
        Tensor: A spectrogram identical to what Kaldi would output. The shape is
        (m, ``padded_window_size // 2 + 1``) where m is calculated in _get_strided
    """
    device, dtype = waveform.device, waveform.dtype
    epsilon = _get_epsilon(device, dtype)

    waveform, window_shift, window_size, padded_window_size = _get_waveform_and_window_properties(
        waveform, channel, sample_frequency, frame_shift, frame_length, round_to_power_of_two, preemphasis_coefficient
    )

    if len(waveform) < min_duration * sample_frequency:
        # signal is too short
        return torch.empty(0)

    strided_input, signal_log_energy = _get_window(
        waveform,
        padded_window_size,
        window_size,
        window_shift,
        window_type,
        blackman_coeff,
        snip_edges,
        raw_energy,
        energy_floor,
        dither,
        remove_dc_offset,
        preemphasis_coefficient,
    )

    # size (m, padded_window_size // 2 + 1, 2)
    fft = torch.fft.rfft(strided_input)

    # Convert the FFT into a power spectrum
    power_spectrum = torch.max(fft.abs().pow(2.0), epsilon).log()  # size (m, padded_window_size // 2 + 1)
    power_spectrum[:, 0] = signal_log_energy

    power_spectrum = _subtract_column_mean(power_spectrum, subtract_mean)
    return power_spectrum


def inverse_mel_scale_scalar(mel_freq: float) -> float:
    return 700.0 * (math.exp(mel_freq / 1127.0) - 1.0)


def inverse_mel_scale(mel_freq: Tensor) -> Tensor:
    return 700.0 * ((mel_freq / 1127.0).exp() - 1.0)


def mel_scale_scalar(freq: float) -> float:
    return 1127.0 * math.log(1.0 + freq / 700.0)


def mel_scale(freq: Tensor) -> Tensor:
    return 1127.0 * (1.0 + freq / 700.0).log()


def vtln_warp_freq(
    vtln_low_cutoff: float,
    vtln_high_cutoff: float,
    low_freq: float,
    high_freq: float,
    vtln_warp_factor: float,
    freq: Tensor,
) -> Tensor:
    r"""This computes a VTLN warping function that is not the same as HTK's one,
    but has similar inputs (this function has the advantage of never producing
    empty bins).

    This function computes a warp function F(freq), defined between low_freq
    and high_freq inclusive, with the following properties:
        F(low_freq) == low_freq
        F(high_freq) == high_freq
    The function is continuous and piecewise linear with two inflection
        points.
    The lower inflection point (measured in terms of the unwarped
        frequency) is at frequency l, determined as described below.
    The higher inflection point is at a frequency h, determined as
        described below.
    If l <= f <= h, then F(f) = f/vtln_warp_factor.
    If the higher inflection point (measured in terms of the unwarped
        frequency) is at h, then max(h, F(h)) == vtln_high_cutoff.
        Since (by the last point) F(h) == h/vtln_warp_factor, then
        max(h, h/vtln_warp_factor) == vtln_high_cutoff, so
        h = vtln_high_cutoff / max(1, 1/vtln_warp_factor).
          = vtln_high_cutoff * min(1, vtln_warp_factor).
    If the lower inflection point (measured in terms of the unwarped
        frequency) is at l, then min(l, F(l)) == vtln_low_cutoff
        This implies that l = vtln_low_cutoff / min(1, 1/vtln_warp_factor)
                            = vtln_low_cutoff * max(1, vtln_warp_factor)
    Args:
        vtln_low_cutoff (float): Lower frequency cutoffs for VTLN
        vtln_high_cutoff (float): Upper frequency cutoffs for VTLN
        low_freq (float): Lower frequency cutoffs in mel computation
        high_freq (float): Upper frequency cutoffs in mel computation
        vtln_warp_factor (float): Vtln warp factor
        freq (Tensor): given frequency in Hz

    Returns:
        Tensor: Freq after vtln warp
    """
    assert vtln_low_cutoff > low_freq, "be sure to set the vtln_low option higher than low_freq"
    assert vtln_high_cutoff < high_freq, "be sure to set the vtln_high option lower than high_freq [or negative]"
    l = vtln_low_cutoff * max(1.0, vtln_warp_factor)
    h = vtln_high_cutoff * min(1.0, vtln_warp_factor)
    scale = 1.0 / vtln_warp_factor
    Fl = scale * l  # F(l)
    Fh = scale * h  # F(h)
    assert l > low_freq and h < high_freq
    # slope of left part of the 3-piece linear function
    scale_left = (Fl - low_freq) / (l - low_freq)
    # [slope of center part is just "scale"]

    # slope of right part of the 3-piece linear function
    scale_right = (high_freq - Fh) / (high_freq - h)

    res = torch.empty_like(freq)

    outside_low_high_freq = torch.lt(freq, low_freq) | torch.gt(freq, high_freq)  # freq < low_freq || freq > high_freq
    before_l = torch.lt(freq, l)  # freq < l
    before_h = torch.lt(freq, h)  # freq < h
    after_h = torch.ge(freq, h)  # freq >= h

    # order of operations matter here (since there is overlapping frequency regions)
    res[after_h] = high_freq + scale_right * (freq[after_h] - high_freq)
    res[before_h] = scale * freq[before_h]
    res[before_l] = low_freq + scale_left * (freq[before_l] - low_freq)
    res[outside_low_high_freq] = freq[outside_low_high_freq]

    return res


def vtln_warp_mel_freq(
    vtln_low_cutoff: float,
    vtln_high_cutoff: float,
    low_freq,
    high_freq: float,
    vtln_warp_factor: float,
    mel_freq: Tensor,
) -> Tensor:
    r"""
    Args:
        vtln_low_cutoff (float): Lower frequency cutoffs for VTLN
        vtln_high_cutoff (float): Upper frequency cutoffs for VTLN
        low_freq (float): Lower frequency cutoffs in mel computation
        high_freq (float): Upper frequency cutoffs in mel computation
        vtln_warp_factor (float): Vtln warp factor
        mel_freq (Tensor): Given frequency in Mel

    Returns:
        Tensor: ``mel_freq`` after vtln warp
    """
    return mel_scale(
        vtln_warp_freq(
            vtln_low_cutoff, vtln_high_cutoff, low_freq, high_freq, vtln_warp_factor, inverse_mel_scale(mel_freq)
        )
    )


def get_mel_banks(
    num_bins: int,
    window_length_padded: int,
    sample_freq: float,
    low_freq: float,
    high_freq: float,
    vtln_low: float,
    vtln_high: float,
    vtln_warp_factor: float,
    device=None,
    dtype=None,
) -> Tuple[Tensor, Tensor]:
    """
    Returns:
        (Tensor, Tensor): The tuple consists of ``bins`` (which is
        melbank of size (``num_bins``, ``num_fft_bins``)) and ``center_freqs`` (which is
        center frequencies of bins of size (``num_bins``)).
    """
    assert num_bins > 3, "Must have at least 3 mel bins"
    assert window_length_padded % 2 == 0
    num_fft_bins = window_length_padded / 2
    nyquist = 0.5 * sample_freq

    if high_freq <= 0.0:
        high_freq += nyquist

    assert (0.0 <= low_freq < nyquist) and (0.0 < high_freq <= nyquist) and (low_freq < high_freq), (
        "Bad values in options: low-freq {} and high-freq {} vs. nyquist {}".format(low_freq, high_freq, nyquist)
    )

    # fft-bin width [think of it as Nyquist-freq / half-window-length]
    fft_bin_width = sample_freq / window_length_padded
    mel_low_freq = mel_scale_scalar(low_freq)
    mel_high_freq = mel_scale_scalar(high_freq)

    # divide by num_bins+1 in next line because of end-effects where the bins
    # spread out to the sides.
    mel_freq_delta = (mel_high_freq - mel_low_freq) / (num_bins + 1)

    if vtln_high < 0.0:
        vtln_high += nyquist

    assert vtln_warp_factor == 1.0 or (
        (low_freq < vtln_low < high_freq) and (0.0 < vtln_high < high_freq) and (vtln_low < vtln_high)
    ), "Bad values in options: vtln-low {} and vtln-high {}, versus low-freq {} and high-freq {}".format(
        vtln_low, vtln_high, low_freq, high_freq
    )

    bin = torch.arange(num_bins).unsqueeze(1)
    left_mel = mel_low_freq + bin * mel_freq_delta  # size(num_bins, 1)
    center_mel = mel_low_freq + (bin + 1.0) * mel_freq_delta  # size(num_bins, 1)
    right_mel = mel_low_freq + (bin + 2.0) * mel_freq_delta  # size(num_bins, 1)

    if vtln_warp_factor != 1.0:
        left_mel = vtln_warp_mel_freq(vtln_low, vtln_high, low_freq, high_freq, vtln_warp_factor, left_mel)
        center_mel = vtln_warp_mel_freq(vtln_low, vtln_high, low_freq, high_freq, vtln_warp_factor, center_mel)
        right_mel = vtln_warp_mel_freq(vtln_low, vtln_high, low_freq, high_freq, vtln_warp_factor, right_mel)

    # center_freqs = inverse_mel_scale(center_mel)  # size (num_bins)
    # size(1, num_fft_bins)
    mel = mel_scale(fft_bin_width * torch.arange(num_fft_bins)).unsqueeze(0)

    # size (num_bins, num_fft_bins)
    up_slope = (mel - left_mel) / (center_mel - left_mel)
    down_slope = (right_mel - mel) / (right_mel - center_mel)

    if vtln_warp_factor == 1.0:
        # left_mel < center_mel < right_mel so we can min the two slopes and clamp negative values
        bins = torch.max(torch.zeros(1), torch.min(up_slope, down_slope))
    else:
        # warping can move the order of left_mel, center_mel, right_mel anywhere
        bins = torch.zeros_like(up_slope)
        up_idx = torch.gt(mel, left_mel) & torch.le(mel, center_mel)  # left_mel < mel <= center_mel
        down_idx = torch.gt(mel, center_mel) & torch.lt(mel, right_mel)  # center_mel < mel < right_mel
        bins[up_idx] = up_slope[up_idx]
        bins[down_idx] = down_slope[down_idx]

    return bins.to(device=device, dtype=dtype)  # , center_freqs


cache = {}


def fbank(
    waveform: Tensor,
    blackman_coeff: float = 0.42,
    channel: int = -1,
    dither: float = 0.0,
    energy_floor: float = 1.0,
    frame_length: float = 25.0,
    frame_shift: float = 10.0,
    high_freq: float = 0.0,
    htk_compat: bool = False,
    low_freq: float = 20.0,
    min_duration: float = 0.0,
    num_mel_bins: int = 23,
    preemphasis_coefficient: float = 0.97,
    raw_energy: bool = True,
    remove_dc_offset: bool = True,
    round_to_power_of_two: bool = True,
    sample_frequency: float = 16000.0,
    snip_edges: bool = True,
    subtract_mean: bool = False,
    use_energy: bool = False,
    use_log_fbank: bool = True,
    use_power: bool = True,
    vtln_high: float = -500.0,
    vtln_low: float = 100.0,
    vtln_warp: float = 1.0,
    window_type: str = POVEY,
) -> Tensor:
    r"""Create a fbank from a raw audio signal. This matches the input/output of Kaldi's
    compute-fbank-feats.

    Args:
        waveform (Tensor): Tensor of audio of size (c, n) where c is in the range [0,2)
        blackman_coeff (float, optional): Constant coefficient for generalized Blackman window. (Default: ``0.42``)
        channel (int, optional): Channel to extract (-1 -> expect mono, 0 -> left, 1 -> right) (Default: ``-1``)
        dither (float, optional): Dithering constant (0.0 means no dither). If you turn this off, you should set
            the energy_floor option, e.g. to 1.0 or 0.1 (Default: ``0.0``)
        energy_floor (float, optional): Floor on energy (absolute, not relative) in Spectrogram computation.  Caution:
            this floor is applied to the zeroth component, representing the total signal energy.  The floor on the
            individual spectrogram elements is fixed at std::numeric_limits<float>::epsilon(). (Default: ``1.0``)
        frame_length (float, optional): Frame length in milliseconds (Default: ``25.0``)
        frame_shift (float, optional): Frame shift in milliseconds (Default: ``10.0``)
        high_freq (float, optional): High cutoff frequency for mel bins (if <= 0, offset from Nyquist)
         (Default: ``0.0``)
        htk_compat (bool, optional): If true, put energy last.  Warning: not sufficient to get HTK compatible features
         (need to change other parameters). (Default: ``False``)
        low_freq (float, optional): Low cutoff frequency for mel bins (Default: ``20.0``)
        min_duration (float, optional): Minimum duration of segments to process (in seconds). (Default: ``0.0``)
        num_mel_bins (int, optional): Number of triangular mel-frequency bins (Default: ``23``)
        preemphasis_coefficient (float, optional): Coefficient for use in signal preemphasis (Default: ``0.97``)
        raw_energy (bool, optional): If True, compute energy before preemphasis and windowing (Default: ``True``)
        remove_dc_offset (bool, optional): Subtract mean from waveform on each frame (Default: ``True``)
        round_to_power_of_two (bool, optional): If True, round window size to power of two by zero-padding input
            to FFT. (Default: ``True``)
        sample_frequency (float, optional): Waveform data sample frequency (must match the waveform file, if
            specified there) (Default: ``16000.0``)
        snip_edges (bool, optional): If True, end effects will be handled by outputting only frames that completely fit
            in the file, and the number of frames depends on the frame_length.  If False, the number of frames
            depends only on the frame_shift, and we reflect the data at the ends. (Default: ``True``)
        subtract_mean (bool, optional): Subtract mean of each feature file [CMS]; not recommended to do
            it this way.  (Default: ``False``)
        use_energy (bool, optional): Add an extra dimension with energy to the FBANK output. (Default: ``False``)
        use_log_fbank (bool, optional):If true, produce log-filterbank, else produce linear. (Default: ``True``)
        use_power (bool, optional): If true, use power, else use magnitude. (Default: ``True``)
        vtln_high (float, optional): High inflection point in piecewise linear VTLN warping function (if
            negative, offset from high-mel-freq (Default: ``-500.0``)
        vtln_low (float, optional): Low inflection point in piecewise linear VTLN warping function (Default: ``100.0``)
        vtln_warp (float, optional): Vtln warp factor (only applicable if vtln_map not specified) (Default: ``1.0``)
        window_type (str, optional): Type of window ('hamming'|'hanning'|'povey'|'rectangular'|'blackman')
         (Default: ``'povey'``)

    Returns:
        Tensor: A fbank identical to what Kaldi would output. The shape is (m, ``num_mel_bins + use_energy``)
        where m is calculated in _get_strided
    """
    device, dtype = waveform.device, waveform.dtype

    waveform, window_shift, window_size, padded_window_size = _get_waveform_and_window_properties(
        waveform, channel, sample_frequency, frame_shift, frame_length, round_to_power_of_two, preemphasis_coefficient
    )

    if len(waveform) < min_duration * sample_frequency:
        # signal is too short
        return torch.empty(0, device=device, dtype=dtype)

    # strided_input, size (m, padded_window_size) and signal_log_energy, size (m)
    strided_input, signal_log_energy = _get_window(
        waveform,
        padded_window_size,
        window_size,
        window_shift,
        window_type,
        blackman_coeff,
        snip_edges,
        raw_energy,
        energy_floor,
        dither,
        remove_dc_offset,
        preemphasis_coefficient,
    )

    # size (m, padded_window_size // 2 + 1)
    spectrum = torch.fft.rfft(strided_input).abs()
    if use_power:
        spectrum = spectrum.pow(2.0)

    # size (num_mel_bins, padded_window_size // 2)
    # print(num_mel_bins, padded_window_size, sample_frequency, low_freq, high_freq, vtln_low, vtln_high, vtln_warp)

    cache_key = "%s-%s-%s-%s-%s-%s-%s-%s-%s-%s" % (
        num_mel_bins,
        padded_window_size,
        sample_frequency,
        low_freq,
        high_freq,
        vtln_low,
        vtln_high,
        vtln_warp,
        device,
        dtype,
    )
    if cache_key not in cache:
        mel_energies = get_mel_banks(
            num_mel_bins,
            padded_window_size,
            sample_frequency,
            low_freq,
            high_freq,
            vtln_low,
            vtln_high,
            vtln_warp,
            device,
            dtype,
        )
        cache[cache_key] = mel_energies
    else:
        mel_energies = cache[cache_key]

    # pad right column with zeros and add dimension, size (num_mel_bins, padded_window_size // 2 + 1)
    mel_energies = torch.nn.functional.pad(mel_energies, (0, 1), mode="constant", value=0)

    # sum with mel fiterbanks over the power spectrum, size (m, num_mel_bins)
    mel_energies = torch.mm(spectrum, mel_energies.T)
    if use_log_fbank:
        # avoid log of zero (which should be prevented anyway by dithering)
        mel_energies = torch.max(mel_energies, _get_epsilon(device, dtype)).log()

    # if use_energy then add it as the last column for htk_compat == true else first column
    if use_energy:
        signal_log_energy = signal_log_energy.unsqueeze(1)  # size (m, 1)
        # returns size (m, num_mel_bins + 1)
        if htk_compat:
            mel_energies = torch.cat((mel_energies, signal_log_energy), dim=1)
        else:
            mel_energies = torch.cat((signal_log_energy, mel_energies), dim=1)

    mel_energies = _subtract_column_mean(mel_energies, subtract_mean)
    return mel_energies


def _get_dct_matrix(num_ceps: int, num_mel_bins: int) -> Tensor:
    # returns a dct matrix of size (num_mel_bins, num_ceps)
    # size (num_mel_bins, num_mel_bins)
    dct_matrix = torchaudio.functional.create_dct(num_mel_bins, num_mel_bins, "ortho")
    # kaldi expects the first cepstral to be weighted sum of factor sqrt(1/num_mel_bins)
    # this would be the first column in the dct_matrix for torchaudio as it expects a
    # right multiply (which would be the first column of the kaldi's dct_matrix as kaldi
    # expects a left multiply e.g. dct_matrix * vector).
    dct_matrix[:, 0] = math.sqrt(1 / float(num_mel_bins))
    dct_matrix = dct_matrix[:, :num_ceps]
    return dct_matrix


def _get_lifter_coeffs(num_ceps: int, cepstral_lifter: float) -> Tensor:
    # returns size (num_ceps)
    # Compute liftering coefficients (scaling on cepstral coeffs)
    # coeffs are numbered slightly differently from HTK: the zeroth index is C0, which is not affected.
    i = torch.arange(num_ceps)
    return 1.0 + 0.5 * cepstral_lifter * torch.sin(math.pi * i / cepstral_lifter)


def mfcc(
    waveform: Tensor,
    blackman_coeff: float = 0.42,
    cepstral_lifter: float = 22.0,
    channel: int = -1,
    dither: float = 0.0,
    energy_floor: float = 1.0,
    frame_length: float = 25.0,
    frame_shift: float = 10.0,
    high_freq: float = 0.0,
    htk_compat: bool = False,
    low_freq: float = 20.0,
    num_ceps: int = 13,
    min_duration: float = 0.0,
    num_mel_bins: int = 23,
    preemphasis_coefficient: float = 0.97,
    raw_energy: bool = True,
    remove_dc_offset: bool = True,
    round_to_power_of_two: bool = True,
    sample_frequency: float = 16000.0,
    snip_edges: bool = True,
    subtract_mean: bool = False,
    use_energy: bool = False,
    vtln_high: float = -500.0,
    vtln_low: float = 100.0,
    vtln_warp: float = 1.0,
    window_type: str = POVEY,
) -> Tensor:
    r"""Create a mfcc from a raw audio signal. This matches the input/output of Kaldi's
    compute-mfcc-feats.

    Args:
        waveform (Tensor): Tensor of audio of size (c, n) where c is in the range [0,2)
        blackman_coeff (float, optional): Constant coefficient for generalized Blackman window. (Default: ``0.42``)
        cepstral_lifter (float, optional): Constant that controls scaling of MFCCs (Default: ``22.0``)
        channel (int, optional): Channel to extract (-1 -> expect mono, 0 -> left, 1 -> right) (Default: ``-1``)
        dither (float, optional): Dithering constant (0.0 means no dither). If you turn this off, you should set
            the energy_floor option, e.g. to 1.0 or 0.1 (Default: ``0.0``)
        energy_floor (float, optional): Floor on energy (absolute, not relative) in Spectrogram computation.  Caution:
            this floor is applied to the zeroth component, representing the total signal energy.  The floor on the
            individual spectrogram elements is fixed at std::numeric_limits<float>::epsilon(). (Default: ``1.0``)
        frame_length (float, optional): Frame length in milliseconds (Default: ``25.0``)
        frame_shift (float, optional): Frame shift in milliseconds (Default: ``10.0``)
        high_freq (float, optional): High cutoff frequency for mel bins (if <= 0, offset from Nyquist)
         (Default: ``0.0``)
        htk_compat (bool, optional): If true, put energy last.  Warning: not sufficient to get HTK compatible
         features (need to change other parameters). (Default: ``False``)
        low_freq (float, optional): Low cutoff frequency for mel bins (Default: ``20.0``)
        num_ceps (int, optional): Number of cepstra in MFCC computation (including C0) (Default: ``13``)
        min_duration (float, optional): Minimum duration of segments to process (in seconds). (Default: ``0.0``)
        num_mel_bins (int, optional): Number of triangular mel-frequency bins (Default: ``23``)
        preemphasis_coefficient (float, optional): Coefficient for use in signal preemphasis (Default: ``0.97``)
        raw_energy (bool, optional): If True, compute energy before preemphasis and windowing (Default: ``True``)
        remove_dc_offset (bool, optional): Subtract mean from waveform on each frame (Default: ``True``)
        round_to_power_of_two (bool, optional): If True, round window size to power of two by zero-padding input
            to FFT. (Default: ``True``)
        sample_frequency (float, optional): Waveform data sample frequency (must match the waveform file, if
            specified there) (Default: ``16000.0``)
        snip_edges (bool, optional): If True, end effects will be handled by outputting only frames that completely fit
            in the file, and the number of frames depends on the frame_length.  If False, the number of frames
            depends only on the frame_shift, and we reflect the data at the ends. (Default: ``True``)
        subtract_mean (bool, optional): Subtract mean of each feature file [CMS]; not recommended to do
            it this way.  (Default: ``False``)
        use_energy (bool, optional): Add an extra dimension with energy to the FBANK output. (Default: ``False``)
        vtln_high (float, optional): High inflection point in piecewise linear VTLN warping function (if
            negative, offset from high-mel-freq (Default: ``-500.0``)
        vtln_low (float, optional): Low inflection point in piecewise linear VTLN warping function (Default: ``100.0``)
        vtln_warp (float, optional): Vtln warp factor (only applicable if vtln_map not specified) (Default: ``1.0``)
        window_type (str, optional): Type of window ('hamming'|'hanning'|'povey'|'rectangular'|'blackman')
         (Default: ``"povey"``)

    Returns:
        Tensor: A mfcc identical to what Kaldi would output. The shape is (m, ``num_ceps``)
        where m is calculated in _get_strided
    """
    assert num_ceps <= num_mel_bins, "num_ceps cannot be larger than num_mel_bins: %d vs %d" % (num_ceps, num_mel_bins)

    device, dtype = waveform.device, waveform.dtype

    # The mel_energies should not be squared (use_power=True), not have mean subtracted
    # (subtract_mean=False), and use log (use_log_fbank=True).
    # size (m, num_mel_bins + use_energy)
    feature = fbank(
        waveform=waveform,
        blackman_coeff=blackman_coeff,
        channel=channel,
        dither=dither,
        energy_floor=energy_floor,
        frame_length=frame_length,
        frame_shift=frame_shift,
        high_freq=high_freq,
        htk_compat=htk_compat,
        low_freq=low_freq,
        min_duration=min_duration,
        num_mel_bins=num_mel_bins,
        preemphasis_coefficient=preemphasis_coefficient,
        raw_energy=raw_energy,
        remove_dc_offset=remove_dc_offset,
        round_to_power_of_two=round_to_power_of_two,
        sample_frequency=sample_frequency,
        snip_edges=snip_edges,
        subtract_mean=False,
        use_energy=use_energy,
        use_log_fbank=True,
        use_power=True,
        vtln_high=vtln_high,
        vtln_low=vtln_low,
        vtln_warp=vtln_warp,
        window_type=window_type,
    )

    if use_energy:
        # size (m)
        signal_log_energy = feature[:, num_mel_bins if htk_compat else 0]
        # offset is 0 if htk_compat==True else 1
        mel_offset = int(not htk_compat)
        feature = feature[:, mel_offset : (num_mel_bins + mel_offset)]

    # size (num_mel_bins, num_ceps)
    dct_matrix = _get_dct_matrix(num_ceps, num_mel_bins).to(dtype=dtype, device=device)

    # size (m, num_ceps)
    feature = feature.matmul(dct_matrix)

    if cepstral_lifter != 0.0:
        # size (1, num_ceps)
        lifter_coeffs = _get_lifter_coeffs(num_ceps, cepstral_lifter).unsqueeze(0)
        feature *= lifter_coeffs.to(device=device, dtype=dtype)

    # if use_energy then replace the last column for htk_compat == true else first column
    if use_energy:
        feature[:, 0] = signal_log_energy

    if htk_compat:
        energy = feature[:, 0].unsqueeze(1)  # size (m, 1)
        feature = feature[:, 1:]  # size (m, num_ceps - 1)
        if not use_energy:
            # scale on C0 (actually removing a scale we previously added that's
            # part of one common definition of the cosine transform.)
            energy *= math.sqrt(2)

        feature = torch.cat((feature, energy), dim=1)

    feature = _subtract_column_mean(feature, subtract_mean)
    return feature

def _get_log_energy_onnx(strided_input: Tensor, epsilon: Tensor, energy_floor: float = 1.0) -> Tensor:
    r"""Returns the log energy of size (m) for a strided_input (m,*)"""
    device, dtype = strided_input.device, strided_input.dtype
    log_energy = torch.max(strided_input.pow(2).sum(1), epsilon).log()  # size (m)
    return torch.max(log_energy, torch.tensor(0.0, device=device, dtype=dtype))


def _get_waveform_and_window_properties_onnx(
    waveform: Tensor,
) -> Tuple[Tensor, int, int, int]:
    r"""ONNX-compatible version with hardcoded parameters from traced fbank call:
    channel=-1, sample_frequency=16000, frame_shift=10.0, frame_length=25.0, 
    round_to_power_of_two=True, preemphasis_coefficient=0.97"""
    
    # Hardcoded values from traced parameters
    # channel=-1 -> 0 after max(channel, 0)
    channel = 0
    
    # Extract channel 0 from waveform
    if waveform.dim() == 1:
        # Mono waveform, use as-is
        waveform_selected = waveform
    else:
        # Multi-channel, select first channel
        waveform_selected = waveform[channel, :]
    
    # Hardcoded calculations:
    # window_shift = int(16000 * 10.0 * 0.001) = 160
    # window_size = int(16000 * 25.0 * 0.001) = 400
    # padded_window_size = _next_power_of_2(400) = 512
    window_shift = 160
    window_size = 400
    padded_window_size = 512
    
    return waveform_selected, window_shift, window_size, padded_window_size

def _get_window_onnx(
    waveform: Tensor,
) -> Tuple[Tensor, Tensor]:
    r"""ONNX-compatible version with hardcoded parameters from traced fbank call:
    padded_window_size=512, window_size=400, window_shift=160, window_type='povey',
    blackman_coeff=0.42, snip_edges=True, raw_energy=True, energy_floor=1.0,
    dither=0, remove_dc_offset=True, preemphasis_coefficient=0.97
    
    Returns:
        (Tensor, Tensor): strided_input of size (m, 512) and signal_log_energy of size (m)
    """
    device, dtype = waveform.device, waveform.dtype
    epsilon = _get_epsilon(device, dtype)
    
    # Hardcoded values from traced parameters
    window_size = 400
    window_shift = 160
    padded_window_size = 512
    snip_edges = True

    # size (m, window_size)
    strided_input = _get_strided_onnx(waveform, window_size, window_shift, snip_edges)
    
    # dither=0, so skip dithering (lines 209-211 from original)
    
    # remove_dc_offset=True, so execute this branch
    row_means = torch.mean(strided_input, dim=1).unsqueeze(1)  # size (m, 1)
    strided_input = strided_input - row_means
    
    # raw_energy=True, so execute this branch
    signal_log_energy = _get_log_energy_onnx(strided_input, epsilon)  # energy_floor=1.0
    
    # preemphasis_coefficient=0.97 != 0.0, so execute this branch
    offset_strided_input = torch.nn.functional.pad(strided_input.unsqueeze(0), (1, 0), mode="replicate").squeeze(0)
    strided_input = strided_input - 0.97 * offset_strided_input[:, :-1]
    
    # Apply povey window function to each row/frame
    # povey window: torch.hann_window(window_size, periodic=False).pow(0.85)
    window_function = torch.hann_window(window_size, periodic=False, device=device, dtype=dtype).pow(0.85).unsqueeze(0)
    strided_input = strided_input * window_function
    
    # Pad columns from window_size=400 to padded_window_size=512
    padding_right = padded_window_size - window_size  # 112
    strided_input = torch.nn.functional.pad(strided_input.unsqueeze(0), (0, padding_right), mode="constant", value=0).squeeze(0)
    
    # raw_energy=True, so skip the "not raw_energy" branch (lines 244-245)    
    return strided_input, signal_log_energy


def _get_strided_onnx(waveform: Tensor, window_size = 400, window_shift = 160, snip_edges = 512) -> Tensor:
    seq_len = waveform.size(0)
    
    # Calculate number of windows
    num_windows = 1 + (seq_len - window_size) // window_shift
    
    # Create indices for all windows at once
    window_starts = torch.arange(0, num_windows * window_shift, window_shift, device=waveform.device)
    window_indices = window_starts.unsqueeze(1) + torch.arange(window_size, device=waveform.device).unsqueeze(0)
    
    # Extract windows using advanced indexing
    windows = waveform[window_indices]  # [num_windows, window_size]
    
    return windows


def _subtract_column_mean_onnx(tensor: Tensor) -> Tensor:
    """ONNX-compatible version with hardcoded parameters from traced fbank call:
    subtract_mean=False, so this function returns the input tensor unchanged.
    
    Args:
        tensor: Input tensor of size (m, n)
        
    Returns:
        Tensor: Same as input tensor (m, n) since subtract_mean=False
    """
    # subtract_mean=False from traced parameters, so return tensor as-is
    return tensor


def get_mel_banks_onnx(
    device=None,
    dtype=None,
) -> Tensor:
    """ONNX-compatible version with hardcoded parameters from traced fbank call:
    num_bins=80, window_length_padded=512, sample_freq=16000, low_freq=20.0,
    high_freq=0.0, vtln_low=100.0, vtln_high=-500.0, vtln_warp_factor=1.0
    
    Returns:
        Tensor: melbank of size (80, 256) (num_bins, num_fft_bins)
    """
    # Hardcoded values from traced parameters  
    num_bins = 80
    window_length_padded = 512
    sample_freq = 16000.0
    low_freq = 20.0
    high_freq = 0.0  # Will be adjusted to nyquist
    vtln_warp_factor = 1.0
    
    # Calculate dynamic values to ensure accuracy
    num_fft_bins = window_length_padded // 2  # 256 (integer division)
    nyquist = 0.5 * sample_freq  # 8000.0
    
    # high_freq <= 0.0, so high_freq += nyquist
    if high_freq <= 0.0:
        high_freq += nyquist  # 8000.0
    
    # fft-bin width = sample_freq / window_length_padded = 16000 / 512 = 31.25
    fft_bin_width = sample_freq / window_length_padded
    
    # Calculate mel scale values dynamically
    mel_low_freq = mel_scale_scalar(low_freq)
    mel_high_freq = mel_scale_scalar(high_freq)
    
    # mel_freq_delta = (mel_high_freq - mel_low_freq) / (num_bins + 1)
    mel_freq_delta = (mel_high_freq - mel_low_freq) / (num_bins + 1)
    
    # vtln_warp_factor == 1.0, so no VTLN warping needed
    
    # Create mel bin centers
    bin_indices = torch.arange(num_bins, device=device, dtype=dtype).unsqueeze(1)
    left_mel = mel_low_freq + bin_indices * mel_freq_delta
    center_mel = mel_low_freq + (bin_indices + 1.0) * mel_freq_delta
    right_mel = mel_low_freq + (bin_indices + 2.0) * mel_freq_delta
    
    # No VTLN warping since vtln_warp_factor == 1.0
    
    # Create frequency bins for FFT
    fft_freqs = fft_bin_width * torch.arange(num_fft_bins, device=device, dtype=dtype)
    mel = mel_scale(fft_freqs).unsqueeze(0)  # size(1, num_fft_bins)
    
    # Calculate triangular filter banks
    up_slope = (mel - left_mel) / (center_mel - left_mel)
    down_slope = (right_mel - mel) / (right_mel - center_mel)
    
    # Since vtln_warp_factor == 1.0, use the simpler branch
    bins = torch.max(torch.zeros(1, device=device, dtype=dtype), torch.min(up_slope, down_slope))
    
    return bins


def fbank_onnx(
    waveform: Tensor, num_mel_bins=80, sample_frequency=16000, dither=0
) -> Tensor:
    r"""ONNX-compatible fbank function with hardcoded parameters from traced call:
    num_mel_bins=80, sample_frequency=16000, dither=0
    blackman_coeff: float = 0.42,
    channel: int = -1,
    energy_floor: float = 1.0,
    frame_length: float = 25.0,
    frame_shift: float = 10.0,
    high_freq: float = 0.0,
    htk_compat: bool = False,
    low_freq: float = 20.0,
    min_duration: float = 0.0,
    preemphasis_coefficient: float = 0.97,
    raw_energy: bool = True,
    remove_dc_offset: bool = True,
    round_to_power_of_two: bool = True,
    snip_edges: bool = True,
    subtract_mean: bool = False,
    use_energy: bool = False,
    use_log_fbank: bool = True,
    use_power: bool = True,
    vtln_high: float = -500.0,
    vtln_low: float = 100.0,
    vtln_warp: float = 1.0,
    window_type: str = POVEY
    
    Args:
        waveform (Tensor): Tensor of audio of size (c, n) where c is in the range [0,2)
        
    Returns:
        Tensor: A fbank identical to what Kaldi would output. The shape is (m, 80)
        where m is calculated in _get_strided
    """
    device, dtype = waveform.device, waveform.dtype

    # Use ONNX-compatible version of _get_waveform_and_window_properties
    waveform_selected, window_shift, window_size, padded_window_size = _get_waveform_and_window_properties_onnx(waveform)

    # min_duration=0.0, so skip the duration check (signal will never be too short)
    
    # Use ONNX-compatible version of _get_window 
    strided_input, signal_log_energy = _get_window_onnx(waveform_selected)

    # spectrum = torch.fft.rfft(strided_input).abs()
    
    m, frame_size = strided_input.shape

    # Process all frames at once using batch processing
    # Reshape to (m, 1, frame_size) to treat each frame as a separate batch item
    batched_frames = strided_input.unsqueeze(1)  # Shape: (m, 1, 512)

    # Create rectangular window for all frames at once
    rectangular_window = torch.ones(512, device=strided_input.device, dtype=strided_input.dtype)

    # Apply STFT to all frames simultaneously
    # The batch dimension allows us to process all m frames in parallel
    stft_result = torch.stft(
        batched_frames.flatten(0, 1),  # Shape: (m, 512) - flatten batch and channel dims
        n_fft=512,
        hop_length=512,  # Process entire frame at once
        window=rectangular_window,
        center=False,  # Don't add padding
        return_complex=False
    )

    # stft_result shape: (m, 257, 1, 2) where last dim is [real, imag]
    # Calculate magnitude: sqrt(real^2 + imag^2)
    real_part = stft_result[..., 0]  # Shape: (m, 257, 1)
    imag_part = stft_result[..., 1]  # Shape: (m, 257, 1)
    spectrum = torch.sqrt(real_part.pow(2) + imag_part.pow(2)).squeeze(-1)  # Shape: (m, 257)

    # use_power=True, so execute this branch
    spectrum = spectrum.pow(2.0)

    # Get mel filterbanks using ONNX-compatible version
    mel_energies = get_mel_banks_onnx(device, dtype)

    # pad right column with zeros to match FFT output size (80, 256) -> (80, 257)
    mel_energies = torch.nn.functional.pad(mel_energies, (0, 1), mode="constant", value=0)

    # sum with mel filterbanks over the power spectrum, size (m, 80)
    mel_energies = torch.mm(spectrum, mel_energies.T)
    
    # use_log_fbank=True, so execute this branch
    mel_energies = torch.max(mel_energies, _get_epsilon(device, dtype)).log()

    # use_energy=False, so skip the energy addition (lines 828-834)

    # Use ONNX-compatible version of _subtract_column_mean
    mel_energies = _subtract_column_mean_onnx(mel_energies)
    
    return mel_energies

# Test to compare original fbank vs fbank_onnx
if __name__ == "__main__":
    import torch
    
    print("Testing fbank vs fbank_onnx with traced parameters...")
    
    # Create test waveform
    torch.manual_seed(42)
    sample_rate = 16000
    duration = 1.0  # 1 second
    num_samples = int(sample_rate * duration)
    
    # Create a test waveform (sine wave + noise)
    t = torch.linspace(0, duration, num_samples)
    frequency = 440.0  # A4 note
    waveform = torch.sin(2 * torch.pi * frequency * t) + 0.1 * torch.randn(num_samples)
    
    # Test with both mono and stereo inputs
    mono_waveform = waveform.unsqueeze(0)  # Shape: (1, num_samples)
    
    print(f"Test waveform shape: {mono_waveform.shape}")
    
    # Test parameters from trace: num_mel_bins=80, sample_frequency=16000, dither=0
    try:
        print("\n=== DEBUGGING: Step-by-step comparison ===")
        
        # Step 1: Check waveform processing
        orig_waveform, orig_window_shift, orig_window_size, orig_padded_window_size = _get_waveform_and_window_properties(
            mono_waveform, -1, 16000.0, 10.0, 25.0, True, 0.97
        )
        onnx_waveform, onnx_window_shift, onnx_window_size, onnx_padded_window_size = _get_waveform_and_window_properties_onnx(mono_waveform)
        
        print(f"Original waveform shape: {orig_waveform.shape}")
        print(f"ONNX waveform shape: {onnx_waveform.shape}")
        print(f"Waveform difference: {torch.max(torch.abs(orig_waveform - onnx_waveform)).item():.2e}")
        print(f"Window params - orig: shift={orig_window_shift}, size={orig_window_size}, padded={orig_padded_window_size}")
        print(f"Window params - onnx: shift={onnx_window_shift}, size={onnx_window_size}, padded={onnx_padded_window_size}")
        
        # Step 2: Check windowing
        orig_strided, orig_energy = _get_window(
            orig_waveform, orig_padded_window_size, orig_window_size, orig_window_shift,
            'povey', 0.42, True, True, 1.0, 0, True, 0.97
        )
        onnx_strided, onnx_energy = _get_window_onnx(onnx_waveform)
        
        print(f"\nOriginal strided shape: {orig_strided.shape}")
        print(f"ONNX strided shape: {onnx_strided.shape}")
        print(f"Strided difference: {torch.max(torch.abs(orig_strided - onnx_strided)).item():.2e}")
        print(f"Energy difference: {torch.max(torch.abs(orig_energy - onnx_energy)).item():.2e}")
        
        # Step 3: Check mel banks
        orig_mel_banks = get_mel_banks(80, 512, 16000.0, 20.0, 0.0, 100.0, -500.0, 1.0, mono_waveform.device, mono_waveform.dtype)
        onnx_mel_banks = get_mel_banks_onnx(mono_waveform.device, mono_waveform.dtype)
        
        print(f"\nOriginal mel banks shape: {orig_mel_banks.shape}")
        print(f"ONNX mel banks shape: {onnx_mel_banks.shape}")
        print(f"Mel banks difference: {torch.max(torch.abs(orig_mel_banks - onnx_mel_banks)).item():.2e}")
        
        # Step 4: Full comparison
        print("\n=== FULL COMPARISON ===")
        
        # Original fbank
        original_result = fbank(
            mono_waveform,
            num_mel_bins=80,
            sample_frequency=16000,
            dither=0
        )
        
        # ONNX-compatible fbank  
        onnx_result = fbank_onnx(mono_waveform)
        
        print(f"Original fbank output shape: {original_result.shape}")
        print(f"ONNX fbank output shape: {onnx_result.shape}")
        
        # Check if shapes match
        if original_result.shape == onnx_result.shape:
            print("✅ Output shapes match")
        else:
            print("❌ Output shapes don't match")
            print(f"  Original: {original_result.shape}")
            print(f"  ONNX: {onnx_result.shape}")
        
        # Check numerical differences
        if original_result.shape == onnx_result.shape:
            diff = torch.abs(original_result - onnx_result)
            max_diff = torch.max(diff).item()
            mean_diff = torch.mean(diff).item()
            relative_diff = torch.mean(diff / (torch.abs(original_result) + 1e-8)).item()
            
            print(f"Max absolute difference: {max_diff:.2e}")
            print(f"Mean absolute difference: {mean_diff:.2e}")
            print(f"Mean relative difference: {relative_diff:.2e}")
            
            # Find where the max difference occurs
            max_idx = torch.argmax(diff)
            max_coords = torch.unravel_index(max_idx, diff.shape)
            print(f"Max difference at coordinates: {max_coords}")
            print(f"  Original value: {original_result[max_coords].item():.6f}")
            print(f"  ONNX value: {onnx_result[max_coords].item():.6f}")
            
            # Check if results are numerically close
            tolerance = 1e-5
            if max_diff < tolerance:
                print(f"✅ Results are numerically identical (within {tolerance})")
            else:
                print(f"❌ Results {max_diff} differ by more than {tolerance}")

            # Additional statistics
            print(f"Original result range: [{torch.min(original_result).item():.3f}, {torch.max(original_result).item():.3f}]")
            print(f"ONNX result range: [{torch.min(onnx_result).item():.3f}, {torch.max(onnx_result).item():.3f}]")
            
    except Exception as e:
        print(f"❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()
