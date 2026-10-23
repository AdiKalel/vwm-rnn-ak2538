from __future__ import annotations

import sys
import json
import pickle
import time
from datetime import datetime
from pathlib import Path

import numpy as np

import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_experiment import RESULTS_DIR, _weights, run_trial, train

# Running this file directly performs one visible 1-item demonstration.
# In your own code, use `print(run_trial.eloss(...))`: the loss methods
# return numbers and do not print automatically.
print("changed neurons to 256")
weights = _weights()
report = run_trial(weights=weights, set_size=1, store_states=True)
print(f"total_loss      = {report['total_loss']:.6g}")
print(f"error_loss      = {report['error_loss']:.6g}")
print(f"activation_loss = {report['activation_loss']:.6g}")
print(f"target_present  = {report['theta_present_slots']}")
print(f"decoded_present = {report['decoded_present_slots']}")
tloss = np.zeros(10)
eloss = np.zeros(10)
aloss = np.zeros(10)
noise = np.zeros(10)
for i in range(10):
    # Keep states because activation_loss is mean(abs(state)); setting
    # store_states=False is only valid when plotting error/total loss.
    total_eloss = 0
    total_aloss = 0
    for j in range(10):
        report = run_trial(weights=weights, set_size= 2, seed=i*100 + j, store_states=True, noise_factor=0.1 * i)
        total_eloss += report["error_loss"]
        print(f"set_size={2}, seed={i*100 + j}, error_loss={report['error_loss']:.6g}, activation_loss={report['activation_loss']:.6g}")
        total_aloss += report["activation_loss"]
    eloss[i] = total_eloss / 10
    tloss[i] = (total_eloss + total_aloss) / 10
    aloss[i] = total_aloss / 10
    noise[i] = 0.1 * i

plt.plot(noise, eloss, label="error loss")
plt.xlabel("Noise")
plt.ylabel("Error loss")
plt.title("Error loss vs noise")
plt.legend()
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
plt.savefig(RESULTS_DIR / "eloss_vs_noise.png", dpi=160, bbox_inches="tight")
plt.show()
plt.plot(noise, aloss, label="activation loss")
plt.xlabel("Noise")
plt.ylabel("Activation loss")
plt.title("Activation loss vs noise")
plt.legend()
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
plt.savefig(RESULTS_DIR / "aloss_vs_noise.png", dpi=160, bbox_inches="tight")
plt.show()
#print(f"saved graph to {RESULTS_DIR / 'loss_vs_set_size.png'}")
print("check: total_loss - error_loss - activation_loss =", tloss - eloss - aloss)
print('this is changed')