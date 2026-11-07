"""PCA of neural activity in selected time windows and stimulus slots."""

from __future__ import annotations

import argparse
import re
import secrets
from pathlib import Path

import numpy as np

import run_experiment as experiment
from vwm_scratch.trial import run_trial as jax_run_trial


def decode_pca(weights=None, num_trials=1000, samples_per_trial=5, seed=None,
			   output_dir=None, noise_type=None, noise_factor=None, slots=(1,),
			   time_range_ms=None, weight_name=None):
	"""Fit three PCs to evenly spaced states; slots are 1-based, time range is [start, end) in ms."""
	if num_trials < 1 or samples_per_trial < 1:
		raise ValueError("num_trials and samples_per_trial must be positive")
	if num_trials * samples_per_trial < 3:
		raise ValueError("PCA requires at least three sampled states")
	slots = tuple(slots)
	if not slots or any(not isinstance(slot, (int, np.integer)) or isinstance(slot, bool)
						or not 1 <= slot <= experiment.MAX_ITEMS for slot in slots):
		raise ValueError(f"slots must contain integers between 1 and {experiment.MAX_ITEMS}")
	if len(set(slots)) != len(slots):
		raise ValueError("slots must not contain duplicates")
	slots = tuple(sorted(slots))
	decode_start = int((experiment.INIT_MS + experiment.STIMULUS_MS + experiment.DELAY_MS) / experiment.DT_MS)
	if time_range_ms is None:
		time_range_ms = (decode_start * experiment.DT_MS, experiment._steps() * experiment.DT_MS)
	if len(time_range_ms) != 2:
		raise ValueError("time_range_ms must be a (start, end) pair")
	start_ms, end_ms = (float(value) for value in time_range_ms)
	if not np.isfinite(start_ms) or not np.isfinite(end_ms) or not 0 <= start_ms < end_ms <= experiment._steps() * experiment.DT_MS:
		raise ValueError("time_range_ms must lie within the trial and have start < end")
	times = np.arange(experiment._steps()) * experiment.DT_MS
	available_indices = np.flatnonzero((times >= start_ms) & (times < end_ms))
	if samples_per_trial > len(available_indices):
		raise ValueError("samples_per_trial exceeds the number of timesteps in the requested time range")
	sample_indices = available_indices[np.linspace(0, len(available_indices) - 1, samples_per_trial, dtype=int)]
	trajectory_indices = np.flatnonzero(times >= experiment.INIT_MS)

	jax, jnp, _ = experiment._jax()
	if weight_name is None:
		weight_name = (Path(experiment.WEIGHTS_PATH).stem if weights is None and experiment.WEIGHTS_SOURCE == "file"
					   else "initialized" if weights is None else "custom")
	weights = experiment._weights() if weights is None else weights
	weight_label = re.sub(r"[^A-Za-z0-9._-]+", "_", Path(str(weight_name)).stem).strip("._") or "custom"
	neuron_count = weights["B"].shape[0]
	if neuron_count < 3:
		raise ValueError("PCA requires at least three neurons")
	noise_type = experiment.NOISE_TYPE if noise_type is None else noise_type
	noise_factor = experiment.NOISE_FACTOR if noise_factor is None else noise_factor
	if seed is None:
		seed = secrets.randbits(32)
	if not 0 <= seed < 2**32 or num_trials > 2**32:
		raise ValueError("seed must fit in 32 bits and num_trials must not exceed 2**32")

	def rollout(inputs, key):
		result = jax_run_trial(
			weights, inputs, jnp.zeros(neuron_count), experiment.DT_MS,
			experiment.SATURATION_RATE_HZ, noise_type, noise_factor, key, True,
			experiment._steps(),
		)
		return result["states"][sample_indices], result["states"][trajectory_indices]

	compiled_rollout = jax.jit(rollout)
	states = np.empty((num_trials, samples_per_trial, neuron_count), dtype=np.float32)
	trajectory_states = np.empty((num_trials, len(trajectory_indices), neuron_count), dtype=np.float32)
	angles = np.empty((num_trials, len(slots)), dtype=np.float32)
	channels_per_slot = 3 if experiment.POSITIVE_INPUT else 2
	slot_mask = np.zeros(experiment.MAX_ITEMS, dtype=np.float32)
	slot_mask[np.asarray(slots) - 1] = 1.0
	input_mask = jnp.asarray(np.repeat(slot_mask, channels_per_slot))
	for trial_index in range(num_trials):
		trial_seed = (seed + trial_index) % 2**32
		key, theta, _, inputs, _ = experiment._trial_data(jax, jnp, max(slots), trial_seed)
		sampled, trajectory = compiled_rollout(inputs * input_mask, key)
		states[trial_index] = np.asarray(sampled)
		trajectory_states[trial_index] = np.asarray(trajectory)
		angles[trial_index] = np.asarray(theta)[np.asarray(slots) - 1]

	samples = states.reshape(-1, neuron_count).astype(np.float64)
	mean = samples.mean(axis=0)
	centered = samples - mean
	_, singular_values, vectors = np.linalg.svd(centered, full_matrices=False)
	components = vectors[:3]
	projections = (centered @ components.T).reshape(num_trials, samples_per_trial, 3)
	trajectory_projections = np.empty((num_trials, len(trajectory_indices), 3), dtype=np.float32)
	for trial_index in range(num_trials):
		trajectory_projections[trial_index] = (trajectory_states[trial_index] - mean) @ components.T
	variance = singular_values**2 / (len(samples) - 1)
	variance_ratio = variance[:3] / variance.sum() if variance.sum() > 0 else np.zeros(3)

	output_dir = Path(output_dir) if output_dir is not None else experiment.ROOT / "PCAs"
	output_dir.mkdir(parents=True, exist_ok=True)
	slot_label = "-".join(str(slot) for slot in slots)
	filename = f"pca_{weight_label}_slots{slot_label}_{start_ms:g}-{end_ms:g}ms_seed{seed}"
	data_path = output_dir / f"{filename}.npz"
	plot_path = output_dir / f"{filename}.png"
	np.savez_compressed(
		data_path, components=components, mean=mean, explained_variance=variance[:3],
		explained_variance_ratio=variance_ratio, projections=projections,
		trajectory_projections=trajectory_projections, trajectory_times_ms=trajectory_indices * experiment.DT_MS,
		sample_indices=sample_indices, sample_times_ms=sample_indices * experiment.DT_MS,
		target_angles=angles, active_slots=np.asarray(slots), time_range_ms=np.asarray([start_ms, end_ms]),
		weight_name=weight_label, seed=seed, noise_type=noise_type, noise_factor=noise_factor,
		num_trials=num_trials, samples_per_trial=samples_per_trial,
	)

	from matplotlib.figure import Figure
	from matplotlib.backends.backend_agg import FigureCanvasAgg

	figure = Figure(figsize=(8, 7))
	FigureCanvasAgg(figure)
	axis = figure.add_subplot(111, projection="3d")
	flat = projections.reshape(-1, 3)
	scatter = axis.scatter(flat[:, 0], flat[:, 1], flat[:, 2],
						   c=np.repeat(angles[:, 0], samples_per_trial), cmap="hsv",
						   vmin=-np.pi, vmax=np.pi, s=5, alpha=0.5)
	axis.set_xlabel(f"PC 1 ({variance_ratio[0]:.1%})")
	axis.set_ylabel(f"PC 2 ({variance_ratio[1]:.1%})")
	axis.set_zlabel(f"PC 3 ({variance_ratio[2]:.1%})")
	axis.set_title(f"Slots {slot_label} PCA: {start_ms:g}-{end_ms:g} ms")
	figure.colorbar(scatter, ax=axis, shrink=0.7, label=f"Slot {slots[0]} target angle (rad)")
	figure.savefig(plot_path, dpi=160, bbox_inches="tight")
	print(f"Saved PCA data to {data_path} and plot to {plot_path} (seed={seed})")
	return data_path, plot_path


def plot_pca_trajectories(pca_path, slots=None, output_dir=None, weights=None,
						  num_trials=1000, sample_stride=1, seed=None,
						  noise_type=None, noise_factor=None):
	"""Simulate trials and project stimulus-to-end trajectories onto a saved PCA basis."""
	from matplotlib.backends.backend_agg import FigureCanvasAgg
	from matplotlib.cm import ScalarMappable
	from matplotlib import colormaps
	from matplotlib.colors import Normalize
	from matplotlib.figure import Figure
	from mpl_toolkits.mplot3d.art3d import Line3DCollection

	pca_path = Path(pca_path)
	with np.load(pca_path, allow_pickle=False) as data:
		components = np.asarray(data["components"])
		mean = np.asarray(data["mean"])
		if "active_slots" in data:
			active_slots = np.asarray(data["active_slots"])
		elif "target_angles" in data and data["target_angles"].ndim == 1:
			active_slots = np.array([1])
		else:
			raise ValueError("PCA file is missing active_slots metadata")
		variance_ratio = np.asarray(data["explained_variance_ratio"])
		weight_name = str(data["weight_name"].item()) if "weight_name" in data else None
		stored_noise_type = str(data["noise_type"].item()) if "noise_type" in data else None
		stored_noise_factor = float(data["noise_factor"].item()) if "noise_factor" in data else None

	if components.ndim != 2 or components.shape[0] != 3 or mean.shape != (components.shape[1],):
		raise ValueError("PCA components and mean must have shapes (3, neurons) and (neurons,)")
	if not np.all(np.isfinite(components)) or not np.all(np.isfinite(mean)):
		raise ValueError("PCA components and mean must be finite")
	if (active_slots.ndim != 1 or not len(active_slots) or
			not np.issubdtype(active_slots.dtype, np.integer) or
			len(np.unique(active_slots)) != len(active_slots) or
			np.any(active_slots < 1) or np.any(active_slots > experiment.MAX_ITEMS)):
		raise ValueError("PCA active_slots must contain distinct slot identifiers")
	if variance_ratio.shape != (3,) or not np.all(np.isfinite(variance_ratio)):
		raise ValueError("PCA must contain three finite explained variance ratios")

	selected_slots = list(active_slots) if slots is None else list(slots)
	if (not selected_slots or any(isinstance(slot, bool) or not isinstance(slot, (int, np.integer))
			or not 1 <= slot <= experiment.MAX_ITEMS for slot in selected_slots)):
		raise ValueError(f"slots must contain integers between 1 and {experiment.MAX_ITEMS}")
	if len(set(selected_slots)) != len(selected_slots):
		raise ValueError("slots must not contain duplicates")
	trajectory_slots = np.asarray(selected_slots, dtype=int)
	if not isinstance(num_trials, int) or isinstance(num_trials, bool) or not 1 <= num_trials <= 2**32:
		raise ValueError("num_trials must be an integer between 1 and 2**32")
	if not isinstance(sample_stride, int) or isinstance(sample_stride, bool) or sample_stride < 1:
		raise ValueError("sample_stride must be a positive integer")
	if seed is None:
		seed = secrets.randbits(32)
	if not isinstance(seed, int) or isinstance(seed, bool) or not 0 <= seed < 2**32:
		raise ValueError("seed must be an integer between 0 and 2**32 - 1")
	if weights is None:
		if not weight_name or weight_name in ("custom", "initialized"):
			raise ValueError("PCA file does not identify a weights file; pass weights explicitly")
		weight_path = experiment.WEIGHTS_DIR / f"{weight_name}.npz"
		if not weight_path.is_file():
			raise FileNotFoundError(f"PCA weights not found: {weight_path}; pass weights explicitly")
		with np.load(weight_path, allow_pickle=False) as data:
			weights = {name: data[name] for name in ("B", "W", "F", "tau", "dale_sign")}
	if weights["B"].shape[0] != mean.shape[0]:
		raise ValueError("The weights neuron count does not match the PCA basis")
	noise_type = noise_type if noise_type is not None else (stored_noise_type if stored_noise_type is not None else experiment.NOISE_TYPE)
	noise_factor = noise_factor if noise_factor is not None else (stored_noise_factor if stored_noise_factor is not None else experiment.NOISE_FACTOR)
	jax, jnp, _ = experiment._jax()
	trajectory_indices = np.flatnonzero(np.arange(experiment._steps()) * experiment.DT_MS >= experiment.INIT_MS)[::sample_stride]
	if len(trajectory_indices) == 0 or trajectory_indices[-1] != experiment._steps() - 1:
		trajectory_indices = np.append(trajectory_indices, experiment._steps() - 1)
	if len(trajectory_indices) < 2:
		raise ValueError("The stimulus-to-end window must contain at least two timesteps")
	channels_per_slot = 3 if experiment.POSITIVE_INPUT else 2
	slot_mask = np.zeros(experiment.MAX_ITEMS, dtype=np.float32)
	slot_mask[trajectory_slots - 1] = 1.0
	input_mask = jnp.asarray(np.repeat(slot_mask, channels_per_slot))

	def rollout(inputs, key):
		result = jax_run_trial(weights, inputs, jnp.zeros(mean.shape[0]), experiment.DT_MS,
							   experiment.SATURATION_RATE_HZ, noise_type, noise_factor, key,
							   True, experiment._steps())
		return (result["states"][trajectory_indices] - jnp.asarray(mean)) @ jnp.asarray(components).T

	compiled_rollout = jax.jit(rollout)
	projections = np.empty((num_trials, len(trajectory_indices), 3), dtype=np.float32)
	angles = np.empty((num_trials, len(trajectory_slots)), dtype=np.float32)
	for trial_index in range(num_trials):
		trial_seed = (seed + trial_index) % 2**32
		key, theta, _, inputs, _ = experiment._trial_data(jax, jnp, int(np.max(trajectory_slots)), trial_seed)
		projections[trial_index] = np.asarray(compiled_rollout(inputs * input_mask, key))
		angles[trial_index] = np.asarray(theta)[trajectory_slots - 1]
	if not np.all(np.isfinite(projections)) or not np.all(np.isfinite(angles)):
		raise ValueError("Trial trajectories and target angles must be finite")
	output_dir = Path(output_dir) if output_dir is not None else pca_path.parent
	output_dir.mkdir(parents=True, exist_ok=True)
	norm = Normalize(vmin=-np.pi, vmax=np.pi)
	paths = []
	for slot in selected_slots:
		column = int(np.flatnonzero(trajectory_slots == slot)[0])
		figure = Figure(figsize=(8, 7))
		FigureCanvasAgg(figure)
		axis = figure.add_subplot(111, projection="3d")
		segments = np.stack((projections[:, :-1], projections[:, 1:]), axis=2)
		end_colors = colormaps["hsv"](norm(angles[:, column]))
		progress = np.linspace(0.0, 1.0, segments.shape[1])
		colors = np.broadcast_to(end_colors[:, None, :], (*segments.shape[:2], 4)).copy()
		colors[..., :3] *= progress[None, :, None]
		colors[..., 3] = 0.65
		lines = Line3DCollection(segments.reshape(-1, 2, 3), colors=colors.reshape(-1, 4), linewidths=0.8)
		axis.add_collection3d(lines)
		flat = projections.reshape(-1, 3)
		axis.auto_scale_xyz(flat[:, 0], flat[:, 1], flat[:, 2])
		for index, setter in enumerate((axis.set_xlabel, axis.set_ylabel, axis.set_zlabel)):
			setter(f"PC {index + 1} ({variance_ratio[index]:.1%})")
		axis.set_title(f"Stimulus-to-end trajectories: black to slot {slot} angle colour")
		figure.colorbar(ScalarMappable(norm=norm, cmap="hsv"), ax=axis, shrink=0.7,
						label=f"Slot {slot} target angle (rad)")
		slot_label = "-".join(str(item) for item in sorted(trajectory_slots))
		path = output_dir / f"{pca_path.stem}_trajectories_slots{slot_label}_trials{num_trials}_stride{sample_stride}_seed{seed}_color_slot{slot}.png"
		figure.savefig(path, dpi=160, bbox_inches="tight")
		paths.append(path)
	print(f"Trajectory run seed: {seed}; noise={noise_type}, factor={noise_factor}; "
		  f"sample stride={sample_stride} ({sample_stride * experiment.DT_MS:g} ms)")
	return paths


def main():
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument("--trials", type=int, default=1000)
	parser.add_argument("--samples", type=int, default=5)
	parser.add_argument("--seed", type=int)
	parser.add_argument("--output-dir", type=Path)
	parser.add_argument("--slots", type=int, nargs="+", default=[1], help="1-based active stimulus slots")
	parser.add_argument("--time-range-ms", type=float, nargs=2, metavar=("START", "END"),
						help="sample window in ms, start inclusive and end exclusive")
	parser.add_argument("--weights", type=Path, help="weights NPZ (defaults to run_experiment settings)")
	parser.add_argument("--plot-trajectories", type=Path, help="plot trajectories from an existing PCA NPZ")
	parser.add_argument("--colour-slots", type=int, nargs="+", help="slots to activate and colour trajectory plots by")
	parser.add_argument("--trajectory-stride", type=int, default=1, help="sample every N model timesteps when plotting trajectories")
	parser.add_argument("--noise-type", choices=("none", "gamma", "gaussian", "puregauss", "csnr"))
	parser.add_argument("--noise-factor", type=float)
	args = parser.parse_args()
	weights = None
	if args.weights is not None:
		with np.load(args.weights, allow_pickle=False) as data:
			weights = {name: data[name] for name in ("B", "W", "F", "tau", "dale_sign")}
	if args.plot_trajectories is not None:
		for path in plot_pca_trajectories(
				args.plot_trajectories, slots=args.colour_slots, output_dir=args.output_dir,
				weights=weights, num_trials=args.trials, sample_stride=args.trajectory_stride,
				seed=args.seed, noise_type=args.noise_type, noise_factor=args.noise_factor):
			print(f"Saved trajectory plot to {path}")
		return
	if args.colour_slots is not None:
		parser.error("--colour-slots requires --plot-trajectories")
	#decode_pca(num_trials=args.trials, samples_per_trial=args.samples,
			   #seed=args.seed, output_dir=args.output_dir, slots=args.slots,
			   #time_range_ms=args.time_range_ms, weights=weights, weight_name=args.weights,
			   #noise_type=args.noise_type, noise_factor=args.noise_factor)
	with np.load("from_scratch/weights/weights_final_tau200_retrained.npz") as data:
		weights = {name: data[name] for name in ("B", "W", "F", "tau", "dale_sign")}
	paths = plot_pca_trajectories(
        "from_scratch/PCAs/decode_pca_adi_1.npz",
        slots=[2],
        weights=weights,
        num_trials=200,
        sample_stride=1,
    )


if __name__ == "__main__":
	main()
