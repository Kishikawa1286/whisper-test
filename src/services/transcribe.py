from typing import Dict, Any
import whisper
import soundfile as sf
import numpy as np


from utils.resample import resample as resample_audio


def transcribe(model: whisper.Whisper, audiofile: str) -> Dict[str, Any]:
    # Read the audio file into a numpy array
    audio_data_np, sample_rate = sf.read(audiofile)

    # Convert to monaural if the audio has multiple channels
    if len(audio_data_np.shape) == 2 and audio_data_np.shape[1] >= 2:
        audio_data_np = np.mean(audio_data_np, axis=1)

    # Resample the audio data if necessary
    audio_data_tensor = resample_audio(audio_data_np, sample_rate)

    result = model.transcribe(audio_data_tensor)

    return result.to_dict()  # type: ignore
