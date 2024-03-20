from kedro.io import AbstractDataSet
import numpy as np
from typing import Any, Dict
from kedro.io.core import get_filepath_str, get_protocol_and_path
from pathlib import PurePosixPath
import fsspec
from pydub import AudioSegment
import pydub


class VideoDataSet(AbstractDataSet[np.ndarray, np.ndarray]):
    """``ImageDataSet`` loads / save image data from a given filepath as `numpy` array using Pillow.

        Example:
        ::

        ImageDataSet(filepath='/img/file/path.png')
        """

    def __init__(self, filepath: str):
        """Creates a new instance of ImageDataSet to load / save image data at the given filepath.

        Args:
            filepath: The location of the image file to load / save data.
        """
        protocol, path = get_protocol_and_path(filepath)
        self._protocol = protocol
        self._filepath = PurePosixPath(path)
        self._fs = fsspec.filesystem(self._protocol)

    def _load(self) -> pydub.audio_segment.AudioSegment:
        """Loads data from the image file.

        Returns:
            Data from the image file as a numpy array.
        """
        load_path = get_filepath_str(self._filepath, self._protocol)
        with self._fs.open(load_path) as f:
            track = AudioSegment.from_file(f, format='m4a')
            return track

    def _save(self, data: pydub.audio_segment.AudioSegment) -> None:
        """Saves image data to the specified filepath"""

        save_path = get_filepath_str(self._filepath, self._protocol)
        with self._fs.open(save_path, "wb") as f:
            data.export(f, format='wav')

    def _describe(self) -> Dict[str, Any]:
        """Returns a dict that describes the attributes of the dataset"""
        pass
