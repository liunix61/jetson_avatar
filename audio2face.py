# speech to Audio2Face module utilizing the gRPC protocal from audio2face_streaming_utils

import riva.client
import io
import numpy as np
import threading
import requests
from audio2face_streaming_utils import push_audio_track


class Audio2FaceClass(threading.Thread):
    def __init__(self, sample_rate=44100, callback=None, **kwargs):
        super(Audio2FaceClass, self).__init__()
        self.callback = callback
        self.a2f_url = 'IP_ADDRESS:50051' # The audio2face url by default
        self.a2f_avatar_instance = '/World/audio2face/PlayerStreaming' # The instance name of the avatar in a2f
        self.sample_rate = 44100
        self.server = 'http://IP_ADDRESS:8011'
        self.usd_scene = 'C:/Users/admin/AppData/Local/ov/pkg/audio2face-2022.2.1/exts/omni.audio2face.wizard/assets/demo_fullface_streaming.usda'
        self.a2f_instance = self.A2F()   
         
    def A2F(self):
        payload = {
            "file_name": self.usd_scene
            }
        usd = requests.post(f'{self.server}/A2F/USD/Load', json=payload)
        print(f"USD scene: {usd.json()['message']}")
        self.a2f_instance = requests.get(f'{self.server}/A2F/GetInstances').json()
        self.a2f_instance = self.a2f_instance['result']['fullface_instances'][0]
        print(f'A2F Instance: {self.a2f_instance}')
        return self.a2f_instance

    def A2E(self):
        payload = {
            "a2f_instance": self.a2f_instance,
            "emotions": {
                "neutral": 0.5,
                "joy": 1
                } 
            }
        a2e = requests.post(f'{self.server}/A2F/A2E/SetEmotionByName', json=payload)
        print(f'A2E parameters: {a2e.json()["message"]}')

    def wav_to_numpy_float32(self, wav_byte) -> float:
        """
        :param wav_byte: wav byte
        :return: float32
        """
        return wav_byte.astype(np.float32, order='C') / 32768.0


    def make_avatar_speaks(self, audio) -> None:
        """
        :param audio: tts audio
        :return: None
        """
        if self.callback:
            self.callback(audio)
        push_audio_track(self.a2f_url, self.wav_to_numpy_float32(audio), self.sample_rate, self.a2f_avatar_instance)
