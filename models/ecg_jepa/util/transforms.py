# Copyright 2024 ST-MEM paper authors. <https://github.com/bakqui/ST-MEM>

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from scipy.signal import butter, resample, sosfiltfilt, square


__all__ = ['Resample',
           'RandomCrop',
           'CenterCrop',
           'MovingWindowCrop',
           'NCrop',
           'SOSFilter',
           'HighpassFilter',
           'LowpassFilter',
           'Standardize',
           'RandomSingleLeadMask',
           'RandomLeadMask',
           'YFlip',
           'RandomMask',
           'Cutout',
           'RandomShift',
           'SineNoise',
           'SquareNoise',
           'WhiteNoise',
           'RandomPartialSineNoise',
           'RandomPartialSquareNoise',
           'RandomPartialWhiteNoise',
           'ClassLabel',
           'ClassOneHot',
           'RandomApply',
           'Compose',
           'ToTensor',
           'RandAugment',
           'get_transforms_from_config',
           'get_rand_augment_from_config',
           ]


"""Preprocessing1
"""
class Resample:
    """Resample the input sequence.
    """
    def __init__(self,
                 target_length: Optional[int] = None,
                 target_fs: Optional[int] = None) -> None:
        self.target_length = target_length
        self.target_fs = target_fs

    def __call__(self, x: np.ndarray, fs: Optional[int] = None) -> np.ndarray:
        if fs and self.target_fs and fs != self.target_fs:
            x = resample(x, int(x.shape[1] * self.target_fs / fs), axis=1)
        elif self.target_length and x.shape[1] != self.target_length:
            x = resample(x, self.target_length, axis=1)
        return x

class RandomCrop:
    """Crop randomly the input sequence.
    """
    def __init__(self, crop_length: int) -> None:
        self.crop_length = crop_length

    def __call__(self, x: np.ndarray) -> np.ndarray:
        if self.crop_length > x.shape[1]:
            raise ValueError(f"crop_length must be smaller than the length of x ({x.shape[1]}).")
        start_idx = np.random.randint(0, x.shape[1] - self.crop_length + 1)
        return x[:, start_idx:start_idx + self.crop_length]

class CenterCrop:
    """Crop the input sequence at the center.
    """
    def __init__(self, crop_length: int) -> None:
        self.crop_length = crop_length

    def __call__(self, x: np.ndarray) -> np.ndarray:
        if self.crop_length > x.shape[1]:
            raise ValueError(f"crop_length must be smaller than the length of x ({x.shape[1]}).")
        start_idx = (x.shape[1] - self.crop_length) // 2
        return x[:, start_idx:start_idx + self.crop_length]

class MovingWindowCrop:
    """Crop the input sequence with a moving window.
    """
    def __init__(self, crop_length: int, crop_stride: int) -> None:
        self.crop_length = crop_length
        self.crop_stride = crop_stride

    def __call__(self, x: np.ndarray) -> np.ndarray:
        if self.crop_length > x.shape[1]:
            raise ValueError(f"crop_length must be smaller than the length of x ({x.shape[1]}).")
        start_idx = np.arange(0, x.shape[1] - self.crop_length + 1, self.crop_stride)
        return np.stack([x[:, i:i + self.crop_length] for i in start_idx], axis=0)

class NCrop:
    """Crop the input sequence to N segments with equally spaced intervals.
    """
    def __init__(self, crop_length: int, num_segments: int) -> None:
        self.crop_length = crop_length
        self.num_segments = num_segments

    def __call__(self, x: np.ndarray) -> np.ndarray:
        if self.crop_length > x.shape[1]:
            raise ValueError(f"crop_length must be smaller than the length of x ({x.shape[1]}).")
        start_idx = np.arange(start=0,
                              stop=x.shape[1] - self.crop_length + 1,
                              step=(x.shape[1] - self.crop_length) // (self.num_segments - 1))
        return np.stack([x[:, i:i + self.crop_length] for i in start_idx], axis=0)

class SOSFilter:
    """Apply SOS filter to the input sequence.
    """
    def __init__(self,
                 fs: int,
                 cutoff: float,
                 order: int = 5,
                 btype: str = 'highpass') -> None:
        self.sos = butter(order, cutoff, btype=btype, fs=fs, output='sos')

    def __call__(self, x):
        return sosfiltfilt(self.sos, x)

class HighpassFilter(SOSFilter):
    """Apply highpass filter to the input sequence.
    """
    def __init__(self, fs: int, cutoff: float, order: int = 5) -> None:
        super(HighpassFilter, self).__init__(fs, cutoff, order, btype='highpass')

class LowpassFilter(SOSFilter):
    """Apply lowpass filter to the input sequence.
    """
    def __init__(self, fs: int, cutoff: float, order: int = 5) -> None:
        super(LowpassFilter, self).__init__(fs, cutoff, order, btype='lowpass')

class Standardize:
    """Standardize the input sequence.
    """
    def __init__(self, axis: Union[int, Tuple[int, ...], List[int]] = (-1, -2)) -> None:
        if isinstance(axis, list):
            axis = tuple(axis)
        self.axis = axis

    def __call__(self, x: np.ndarray) -> np.ndarray:
        loc = np.mean(x, axis=self.axis, keepdims=True)
        scale = np.std(x, axis=self.axis, keepdims=True)
        # Set rst = 0 if std = 0
        return np.divide(x - loc, scale, out=np.zeros_like(x), where=scale != 0)


"""Augmentations
"""
class _BaseAugment:
    """Base class for augmentations.
    """
    def __call__(self, x: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def _set_level(self, level: int, max_level: int = 10, **kwargs) -> None:
        pass


"""Group 1: Lead manipulation
"""
class LeadMask(_BaseAugment):
    """Mask the lead.
    """
    def __init__(self,
                 mask_indices: Optional[List[int]] = None,
                 mode: Optional[str] = None,
                 ) -> None:
        self.mask_indices = mask_indices
        if mask_indices is None:
            if mode == 'limb':
                self.mask_indices = [6, 7, 8, 9, 10, 11]
            elif mode == 'lead1':
                self.mask_indices = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
            elif mode == 'lead2':
                self.mask_indices = [0, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
            else:
                raise ValueError(f"Invalid mode: {mode}")

    def __call__(self, x: np.ndarray) -> np.ndarray:
        rst = x.copy()
        if x.shape[0] > 1:
            mask_indices = [i for i in self.mask_indices if i < x.shape[0]]
            rst[mask_indices] = 0
        return rst

class RandomSingleLeadMask(_BaseAugment):
    """Randomly select a lead and mask it.
    """
    def __call__(self, x: np.ndarray) -> np.ndarray:
        rst = x.copy()
        mask_idx = np.random.randint(x.shape[0])
        rst[mask_idx] = 0
        return rst

class RandomLeadMask(_BaseAugment):
    """Randomly mask the leads and re-scale the signal.
    """
    def __init__(self,
                 mask_ratio: float = 0.3,
                 axis: Union[int, Tuple[int, ...], List[int]] = (-1, -2)) -> None:
        self.mask_ratio = mask_ratio
        if isinstance(axis, list):
            axis = tuple(axis)
        self.axis = axis

    def __call__(self, x: np.ndarray) -> np.ndarray:
        rst = x.copy()
        not_masked = []
        for i in range(x.shape[0]):
            if np.random.rand() < self.mask_ratio:
                rst[i] = 0
            else:
                not_masked.append(i)
        if not_masked:
            rst_not_masked = rst[not_masked]
            new_loc = np.mean(rst_not_masked)
            new_scale = np.std(rst_not_masked)
            rst[not_masked] = np.divide(rst_not_masked - new_loc,
                                        new_scale,
                                        out=np.zeros_like(rst_not_masked),
                                        where=new_scale != 0)
        return rst

class YFlip(_BaseAugment):
    """Flip the signal along the y-axis.
    """
    def __call__(self, x: np.ndarray) -> np.ndarray:
        return -x


"""Group 2: Signal manipulation
"""
class _Mask(_BaseAugment):
    """Base class for signal masking.
    """
    def __init__(self, mask_ratio: float = 0.3) -> None:
        self.mask_ratio = mask_ratio

    def __call__(self, x: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def _set_level(self, level: int, max_level: int = 10) -> None:
        self.mask_ratio = level / max_level * 0.3

class RandomMask(_Mask):
    """Randomly mask the input sequence.
    """
    def __call__(self, x: np.ndarray) -> np.ndarray:
        rst = x.copy()
        count = np.random.randint(0, int(x.shape[-1] * self.mask_ratio))
        indices = np.random.choice(x.shape[-1], (1, count), replace=False)
        rst[:, indices] = 0
        return rst

class Cutout(_Mask):
    """Cutout the input sequence.
    """
    def __call__(self, x: np.ndarray) -> np.ndarray:
        rst = x.copy()
        count = int(np.random.uniform(0, self.mask_ratio) * x.shape[-1])
        start_idx = np.random.randint(0, x.shape[-1] - count)
        rst[:, start_idx:start_idx + count] = 0
        return rst

class RandomShift(_Mask):
    """Randomly shift (left or right) the input sequence and pad zeros.
    """
    def __call__(self, x: np.ndarray) -> np.ndarray:
        rst = x.copy()
        direction = np.random.choice([-1, 1])
        sig_len = x.shape[-1]
        shift = int(np.random.uniform(0, self.mask_ratio) * sig_len)
        if direction == 1:
            rst[:, shift:] = rst[:, :sig_len - shift]
            rst[:, :shift] = 0
        else:
            rst[:, :sig_len - shift] = rst[:, shift:]
            rst[:, sig_len - shift:] = 0
        return rst


"""Group 3: Noise manipulation
"""
class _Noise(_BaseAugment):
    """Base class for noise manipulation.
    """
    def __init__(self, amplitude: float = 0.3, freq: float = 0.5) -> None:
        self.amplitude = amplitude
        self.freq = freq

    def _get_noise(self, x: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def __call__(self, x: np.ndarray) -> np.ndarray:
        noise = self._get_noise(x)
        return x + noise

    def _set_level(self, level: int, max_level: int = 10) -> None:
        level = level / max_level
        self.amplitude = level * 0.3
        self.freq = 0.5 / level

class SineNoise(_Noise):
    """Add sine noise to the input sequence.
    """
    def _get_noise(self, x: np.ndarray) -> np.ndarray:
        t = np.expand_dims(np.arange(x.shape[-1]) / x.shape[-1], axis=0)
        return self.amplitude * np.sin(2 * np.pi * t / self.freq)

class SquareNoise(_Noise):
    """Add square noise to the input sequence.
    """
    def _get_noise(self, x: np.ndarray) -> np.ndarray:
        t = np.expand_dims(np.arange(x.shape[-1]) / x.shape[-1], axis=0)
        return self.amplitude * square(2 * np.pi * t / self.freq)

class WhiteNoise(_Noise):
    """Add white noise to the input sequence.
    """
    def _get_noise(self, x: np.ndarray) -> np.ndarray:
        return self.amplitude * np.random.randn(*x.shape)

class _RandomPartialNoise(_Noise):
    """Base class for adding noise to the random part of the input sequence.
    """
    def __init__(self, amplitude: float = 0.3, freq: float = 0.5, ratio: float = 0.3) -> None:
        super(_RandomPartialNoise, self).__init__(amplitude, freq)
        self.ratio = ratio

    def _get_partial_noise(self, x: np.ndarray) -> np.ndarray:
        noise = self._get_noise(x)
        count = int(np.random.uniform(0, self.ratio) * x.shape[-1])
        start_idx = np.random.randint(0, x.shape[-1] - count)
        partial_noise = np.zeros_like(x)
        partial_noise[:, start_idx:start_idx + count] = noise[:, :count]
        return partial_noise

    def __call__(self, x: np.ndarray) -> np.ndarray:
        noise = self._get_partial_noise(x)
        return x + noise

    def _set_level(self, level: int, max_level: int = 10) -> None:
        super(_RandomPartialNoise, self)._set_level(level, max_level)
        self.ratio = level / max_level * 0.3

class RandomPartialSineNoise(_RandomPartialNoise, SineNoise):
    """Add sine noise to the random part of the input sequence.
    """

class RandomPartialSquareNoise(_RandomPartialNoise, SquareNoise):
    """Add square noise to the random part of the input sequence.
    """

class RandomPartialWhiteNoise(_RandomPartialNoise, WhiteNoise):
    """Add white noise to the random part of the input sequence.
    """


"""Label transformation
"""
class ClassLabel:
    """Transform one-hot label to class label.
    """
    def __call__(self, y: np.ndarray) -> int:
        return np.argmax(y)

class ClassOneHot:
    """Transform class label to one-hot label.
    """
    def __init__(self, num_classes: int) -> None:
        self.num_classes = num_classes

    def __call__(self, y: int) -> np.ndarray:
        return np.eye(self.num_classes)[y]


"""Etc
"""
class RandomApply:
    """Apply randomly the given transform.
    """
    def __init__(self, transform: _BaseAugment, prob: float = 0.5) -> None:
        self.transform = transform
        self.prob = prob

    def __call__(self, x: np.ndarray) -> np.ndarray:
        if np.random.rand() < self.prob:
            x = self.transform(x)
        return x

class Compose:
    """Compose several transforms together.
    """
    def __init__(self, transforms: List[Any]) -> None:
        self.transforms = transforms

    def __call__(self, x: np.ndarray) -> np.ndarray:
        for transform in self.transforms:
            x = transform(x)
        return x

class ToTensor:
    """Convert ndarrays in sample to Tensors.
    """
    _DTYPES = {
        "float": torch.float32,
        "double": torch.float64,
        "int": torch.int32,
        "long": torch.int64,
    }

    def __init__(self, dtype: Union[str, torch.dtype] = torch.float32) -> None:
        if isinstance(dtype, str):
            assert dtype in self._DTYPES, f"Invalid dtype: {dtype}"
            dtype = self._DTYPES[dtype]
        self.dtype = dtype

    def __call__(self, x: Any) -> torch.Tensor:
        return torch.tensor(x, dtype=self.dtype)


"""Random augmentation
"""
class RandAugment:
    """RandAugment: Practical automated data augmentation with a reduced search space.
        ref: https://arxiv.org/abs/1909.13719
    """
    def __init__(self,
                 ops: list,
                 level: int = 10,
                 num_layers: int = 2,
                 prob: float = 0.5,
                 ) -> None:
        self.ops = []
        for op in ops:
            if hasattr(op, '_set_level'):
                op._set_level(level=level)
            self.ops.append(RandomApply(op, prob=prob))
        self.num_layers = num_layers
        self.prob = prob

    def __call__(self, x: np.ndarray) -> np.ndarray:
        ops = np.random.choice(self.ops, self.num_layers, replace=False)
        for op in ops:
            x = op(x)
        return x


PREPROCESSING = {
    'resample': Resample,
    'random_crop': RandomCrop,
    'center_crop': CenterCrop,
    'moving_window_crop': MovingWindowCrop,
    'n_crop': NCrop,
    'highpass_filter': HighpassFilter,
    'lowpass_filter': LowpassFilter,
    'standardize': Standardize,
}

AUGMENTATIONS = {
    'erase': RandomSingleLeadMask,
    'flip': YFlip,
    'drop': RandomMask,
    'cutout': Cutout,
    'shift': RandomShift,
    'sine': SineNoise,
    'square': SquareNoise,
    'white_noise': WhiteNoise,
    'partial_sine': RandomPartialSineNoise,
    'partial_square': RandomPartialSquareNoise,
    'partial_white_noise': RandomPartialWhiteNoise,
    'rlm': RandomLeadMask,
}

def get_transforms_from_config(config: List[Union[str, Dict[str, Any]]]) -> List[_BaseAugment]:
    """Get transforms from config.
    """
    transforms = []
    for transform in config:
        if isinstance(transform, str):
            name = transform
            kwargs = {}
        elif isinstance(transform, dict):
            assert len(transform) == 1, "Each transform must have only one key."
            name, kwargs = list(transform.items())[0]
        else:
            raise ValueError(f"Invalid transform: {transform}, it must be a string or a dictionary.")
        if name in PREPROCESSING:
            transforms.append(PREPROCESSING[name](**kwargs))
        elif name in AUGMENTATIONS:
            transforms.append(AUGMENTATIONS[name](**kwargs))
        else:
            raise ValueError(f"Invalid name: {name}")
    return transforms

def get_rand_augment_from_config(config: Dict[str, Any]) -> RandAugment:
    """Get RandAugment from config.
    """
    op_names = config.get('op_names', [])
    assert op_names, "op_names must be provided."
    level = config.get('level', 10)
    num_layers = config.get('num_layers', 2)
    prob = config.get('prob', 0.5)
    aug_config = {op_name: {} for op_name in op_names}
    return RandAugment(ops=get_transforms_from_config(aug_config),
                       level=level,
                       num_layers=num_layers,
                       prob=prob)
