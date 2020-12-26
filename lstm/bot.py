#!/usr/bin/python3 

import os
import time
import multiprocessing as mp

import cv2
import mss
import numpy as np

import pykitml as pk

# Values shared between processess
A_val = mp.Value('d', 0)
left_val = mp.Value('d', 0) 

def on_frame(server, frame, A_val, left_val): 
    # Toggle start button to start rounds
    if(frame%10 < 5): start = True
    else: start = False

    # Set joypad
    server.set_joypad(A=A_val.value==1, left=left_val.value==1, start=start)

    # Continue emulation
    server.frame_advance()

# Initialize and start server
def start_server(A_val, left_val):
    server = pk.FCEUXServer(lambda server, frame: on_frame(server, frame, A_val, left_val))
    print(server.info)
    server.start()

if __name__ == '__main__':
    p = mp.Process(target=start_server, args=(A_val, left_val))
    p.start()

    # Load models
    from sklearn.decomposition import PCA
    from joblib import load
    pca = load('pca.joblib')
    import torch
    import torch.nn as nn
    model = nn.LSTM(64, 3, 2).float()
    model.load_state_dict(torch.load('lstm.pt'))
    last_render = time.time()
    dps = 0

    with mss.mss() as sct:
        monitor = {"top": 224, "left": 256-116, "width":256, "height":256}

        running = True
        while running:    
            last_time = time.time()

            # Get raw pixels from the screen, save it to a Numpy array
            img = np.array(sct.grab(monitor))
            # Convert to gray scale
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
            # Resize image
            img = cv2.resize(src=img, dsize=(64, 64))
            # Reshape
            img = img.reshape(4096)
            # Normalize
            img = img/255

            # PCA
            img = pca.transform([img])
            
            outputs, _ = model(torch.from_numpy(np.array([img])).float())
            a, left, _ = torch.where(outputs.flatten() == outputs.flatten().max(), 1, 0)
            A_val.value = a.item()
            left_val.value = left.item()
            if A_val.value:
                print("Punch!")
            if left_val.value:
                print("Dodge!")
