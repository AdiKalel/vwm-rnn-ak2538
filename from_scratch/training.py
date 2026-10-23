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

train(steps=100, show_plot=True, num_trials=300, item_numbers=(1,), logging_period=10, neurons = 256)
weights, history = train(
    steps=2000,
    stage=1,
    verification_period=100,
    neurons = 256,
    show_plot=True,
    weights_source="initialise"
    )
#weights, history = train(
 #   steps=2000,
  #  stage=2,
   # resume_from="from_scratch/results/training/latest_checkpoint.pkl",
    #verification_period=100,
#)   