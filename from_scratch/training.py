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

print("Starting training..")
#train(steps=100, show_plot=True, num_trials=300, item_numbers=(1,), logging_period=10, neurons = 256)
weights, history = train(
    steps=2000,
    stage=1,
    verification_period=100,
    neurons = 256,
    show_plot=True,
    weights=_weights(256, "initialize"),
    )
import json
import matplotlib.pyplot as plt

h = json.load(open(r"C:\Users\arkal\vwm_rnn\from_scratch\results\server_history.json"))
plt.plot(h["total_losses"], label="training loss (every step)")
plt.plot(h["iterations"], h["overall_errors"], label="eval angular error (logged steps)")
plt.xlabel("Step")
plt.legend()
plt.show()
#weights, history = train(
 #   steps=2000,
  #  stage=2,
   # resume_from="from_scratch/results/training/latest_checkpoint.pkl",
    #verification_period=100,
#)   