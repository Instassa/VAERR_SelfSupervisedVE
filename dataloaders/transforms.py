import cv2

cv2.setNumThreads(0)
import random
import numpy as np
import torch
from PIL import Image
import torchvision.transforms


__all__ = [
    "Compose",
    "Normalize",
    "CenterCrop",
    "RgbToGray",
    "RandomCrop",
    "HorizontalFlip",
    "AddNoise",
    "NormalizeUtterance",
    "Resize",
    "Solarization",
    "TimeOut",
    "Cutout",
]


class Compose(object):
    """Compose several preprocess together.
    Args:
        preprocess (list of ``Preprocess`` objects): list of preprocess to compose.
    """

    def __init__(self, preprocess):
        self.preprocess = preprocess

    def __call__(self, sample):
        for t in self.preprocess:
            sample = t(sample)
        return sample

    def __repr__(self):
        format_string = self.__class__.__name__ + "("
        for t in self.preprocess:
            format_string += "\n"
            format_string += "    {0}".format(t)
        format_string += "\n)"
        return format_string


class RgbToGray(object):
    """Convert image to grayscale.
    Converts a numpy.ndarray (H x W x C) in the range
    [0, 255] to a numpy.ndarray of shape (H x W x C) in the range [0.0, 1.0].
    """

    def __call__(self, frames):
        """
        Args:
            img (numpy.ndarray): Image to be converted to gray.
        Returns:
            numpy.ndarray: grey image
        """
        frames = np.stack([cv2.cvtColor(_, cv2.COLOR_RGB2GRAY) for _ in frames], axis=0)
        return frames

    def __repr__(self):
        return self.__class__.__name__ + "()"


class Normalize(object):
    """Normalize a ndarray image with mean and standard deviation."""

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, frames):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized Tensor image.
        """
        frames = (frames - self.mean) / self.std
        return frames

    def __repr__(self):
        return self.__class__.__name__ + "(mean={0}, std={1})".format(
            self.mean, self.std
        )


class CenterCrop(object):
    """Crop the given image at the center"""

    def __init__(self, size):
        self.size = size

    def __call__(self, frames):
        """
        Args:
            img (numpy.ndarray): Images to be cropped.
        Returns:
            numpy.ndarray: Cropped image.
        """
        t, h, w = frames.shape
        th, tw = self.size
        delta_w = int(round((w - tw)) / 2.0)
        delta_h = int(round((h - th)) / 2.0)
        frames = frames[:, delta_h : delta_h + th, delta_w : delta_w + tw]
        return frames


class RandomCrop(object):
    """Crop the given image at the center"""

    def __init__(self, size):
        self.size = size

    def __call__(self, frames):
        """
        Args:
            img (numpy.ndarray): Images to be cropped.
        Returns:
            numpy.ndarray: Cropped image.
        """
        t, h, w = frames.shape
        th, tw = self.size
        delta_w = random.randint(0, w - tw)
        delta_h = random.randint(0, h - th)
        frames = frames[:, delta_h : delta_h + th, delta_w : delta_w + tw]
        return frames

    def __repr__(self):
        return self.__class__.__name__ + "(size={0})".format(self.size)


class Resize(object):
    """Resize the given image"""

    def __init__(self, size):
        self.size = size

    def __call__(self, frames):
        return np.array([cv2.resize(_, (self.size[1], self.size[0])) for _ in frames])


class Solarization(object):
    """Flip image horizontally."""

    def __init__(self, solar_ratio, solar_thresh):
        self.solar_ratio = solar_ratio
        self.solar_thresh = solar_thresh

    def __call__(self, frames):
        """
        Args:
            img (numpy.ndarray): Images to be flipped with a probability flip_ratio
        Returns:
            numpy.ndarray: Cropped image.
        """
        t, h, w = frames.shape
        if random.random() < self.solar_ratio:
            for index in range(t):
                orig_img = (Image.fromarray(frames[index])).convert('L')
                # frames[index] = np.asarray(torchvision.transforms.functional.solarize(orig_img, threshold=0.5))
                
                solarizer = torchvision.transforms.RandomSolarize(threshold=int(self.solar_thresh*np.max(orig_img)))
                frames[index] = solarizer(orig_img)
        return frames

class SaltAndPepper(object):
    """Flip image horizontally."""

    def __init__(self, amount):
        self.amount = amount

    def __call__(self, frames):
        """
        Args:
            img (numpy.ndarray): Images to be flipped with a probability flip_ratio
        Returns:
            numpy.ndarray: Cropped image.
        """

        t,row,col = frames.shape
        s_vs_p = 0.5
        # amount = 0.01
        out = np.copy(frames)
        # Salt mode
        num_salt = np.ceil(self.amount * frames.size * s_vs_p)
        coords = [np.random.randint(0, i - 1, int(num_salt))
                for i in frames.shape]
        out[coords] = 1

        # Pepper mode
        num_pepper = np.ceil(self.amount* frames.size * (1. - s_vs_p))
        coords = [np.random.randint(0, i - 1, int(num_pepper))
                for i in frames.shape]
        out[coords] = 0
        return out

        # t, h, w = frames.shape
        # if random.random() < self.solar_ratio:
        #     for index in range(t):
        #         orig_img = (Image.fromarray(frames[index])).convert('L')
        #         # print('shape frame:', (frames[index]).shape)
        #         # print('shape solar:', (np.asarray(torchvision.transforms.functional.solarize(orig_img, threshold=0.5))).shape)
        #         frames[index] = np.asarray(torchvision.transforms.functional.solarize(orig_img, threshold=0.5))
        #         # print('GOT HERE')
        # return frames


class HorizontalFlip(object):
    """Flip image horizontally."""

    def __init__(self, flip_ratio):
        self.flip_ratio = flip_ratio

    def __call__(self, frames):
        """
        Args:
            img (numpy.ndarray): Images to be flipped with a probability flip_ratio
        Returns:
            numpy.ndarray: Cropped image.
        """
        t, h, w = frames.shape
        if random.random() < self.flip_ratio:
            for index in range(t):
                frames[index] = cv2.flip(frames[index], 1)
        return frames

class TimeOut(object):
    """Flip image horizontally."""

    def __init__(self, missing_ratio):
        self.missing_ratio = missing_ratio

    def __call__(self, frames):
        """
        Args:
            img (numpy.ndarray): Images to be flipped with a probability flip_ratio
        Returns:
            numpy.ndarray: Cropped image.
        """
        t, h, w = frames.shape
        if random.random() < self.missing_ratio:
            for index in range(t):
                frames[index] = np.zeros(np.shape(frames[index]))
        return frames


class Cutout(object):
    """Randomly mask out one or more patches from an image.
    Args:
        n_holes (int): Number of patches to cut out of each image.
        length (int): The length (in pixels) of each square patch.
    """

    def __init__(self, n_holes, length):
        self.n_holes = n_holes
        self.length = length

    def __call__(self, frames):

        """
        Args:
            img (numpy.ndarray): Images.
        Returns:
            numpy.ndarray: images with cutout augment.
        """
        T, h, w = frames.shape
        mask = np.ones((T, h, w), np.uint8)
        for n in range(self.n_holes):
            y = random.randrange(0, h)
            x = random.randrange(0, w)

            y1 = np.clip(y - self.length // 2, 0, h)
            y2 = np.clip(y + self.length // 2, 0, h)
            x1 = np.clip(x - self.length // 2, 0, w)
            x2 = np.clip(x + self.length // 2, 0, w)

            mask[:, y1: y2, x1: x2] = 0

        return frames * mask

class NormalizeUtterance:
    """Normalize per raw audio by removing the mean and divided by the standard deviation"""

    def __call__(self, signal):
        signal_std = 0.0 if np.std(signal) == 0.0 else np.std(signal)
        signal_mean = np.mean(signal)
        return (signal - signal_mean) / signal_std


class AddNoise(object):
    """Add SNR noise [-1, 1]"""

    def __init__(self, noise, snr_levels=[-5, 0, 5, 10, 15, 20, 9999]):
        assert noise.dtype in [
            np.float32,
            np.float64,
        ], "noise only supports float data type"

        self.noise = noise
        self.snr_levels = snr_levels

    def get_power(self, clip):
        clip2 = clip.copy()
        clip2 = clip2 ** 2
        return np.sum(clip2) / (len(clip2) * 1.0)

    def __call__(self, signal):
        assert signal.dtype in [
            np.float32,
            np.float64,
        ], "signal only supports float32 data type"
        snr_target = random.choice(self.snr_levels)
        if snr_target == 9999:
            return signal
        else:
            # -- get noise
            start_idx = random.randint(0, len(self.noise) - len(signal))
            noise_clip = self.noise[start_idx : start_idx + len(signal)]

            sig_power = self.get_power(signal)
            noise_clip_power = self.get_power(noise_clip)
            factor = (sig_power / noise_clip_power) / (10 ** (snr_target / 10.0))
            desired_signal = (signal + noise_clip * np.sqrt(factor)).astype(np.float32)
            return desired_signal


class Scale(object):
    """Scale audio tensor from a 16-bit integer (represented as a FloatTensor)
    to a floating point number between -1.0 and 1.0.  Note the 16-bit number is
    called the "bit depth" or "precision", not to be confused with "bit rate".
    Args:
        factor (int): maximum value of input tensor. default: 16-bit depth
    """

    def __init__(self, factor=2 ** 31):
        self.factor = factor

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor of audio of size (Samples x Channels)
        Returns:
            Tensor: Scaled by the scale factor. (default between -1.0 and 1.0)
        """
        if not tensor.dtype.is_floating_point:
            tensor = tensor.to(torch.float32)

        return tensor / self.factor

    def __repr__(self):
        return self.__class__.__name__ + "()"
