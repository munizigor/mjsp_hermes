import base64
import pickle
from scipy.io import wavfile
import sys
import requests
import sys
import numpy as np

if __name__ == "__main__":
    input_path = sys.argv[1]
    id_emergencia = sys.argv[2]
    server_addr = sys.argv[3]

    samplerate, data = wavfile.read(input_path)
    print(f"number of channels = {data.shape[1]}")
    length = data.shape[0] / samplerate
    print(f"length = {length}s")

    '''blob_data = pickle.dumps(data)
    print(type(blob_data))
    blob_b64 = base64.b64encode(blob_data).decode('utf-8')
    print(type(blob_b64))
    print(type(samplerate))
    print(type(id_emergencia))'''

    post_content = {'sp_rate': samplerate, 'id_emergencia': id_emergencia}
    files = {'audio_data': open(input_path, 'rb')}
    response = requests.post(f'http://{server_addr}/?id_emergencia={id_emergencia}&sp_rate={samplerate}', files=files, verify=False)
    print(response.text)