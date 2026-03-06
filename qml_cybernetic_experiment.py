# Updated version of qml_cybernetic_experiment.py with output persistence restored
# Minimal diff: adds --out_prefix, output dirs, per-seed JSON, and summary CSV

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import time
import os
import math
from datetime import datetime
from typing import Any, Optional, Tuple
import numpy as np
from numpy.random import default_rng
import random
from tqdm import trange
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import csv
import sys
import platform
import subprocess
import logging

from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, depolarizing_error

logger = logging.getLogger("qml_cybernetic_experiment")
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s")

# --------------------
# Output helpers
# --------------------

def save_seed_results(history, acc, actions, args, mode, seed, out_root):
    data_dir = os.path.join(out_root, "data")
    os.makedirs(data_dir, exist_ok=True)

    seed_json = {
        "history": history,
        "acc": float(acc),
        "actions": actions,
        "config": vars(args),
        "meta": {
            "mode": mode,
            "seed": int(seed),
            "saved_at": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        }
    }

    seedfile = os.path.join(data_dir, f"{mode}_seed{seed}.json")
    with open(seedfile, "w", encoding="utf-8") as f:
        json.dump(seed_json, f, indent=2)


def write_summary_csv(rows, csvpath):
    keys = ["mode", "seed", "final_acc", "final_loss"]
    with open(csvpath, "w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=keys)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

# --------------------
# Noise model
# --------------------

def make_noise_model(kind="light"):
    if kind in (None, "none"):
        return None
    nm = NoiseModel()
    if kind == "light":
        p1, p2 = 0.005, 0.01
    elif kind == "heavy":
        p1, p2 = 0.02, 0.05
    else:
        return None
    nm.add_all_qubit_quantum_error(depolarizing_error(p1, 1), ["ry", "rz", "rx"])
    nm.add_all_qubit_quantum_error(depolarizing_error(p2, 2), ["cx"])
    return nm

# --------------------
# Circuits
# --------------------

def angle_encoding_circuit(x, qubits):
    qc = QuantumCircuit(qubits)
    for i, xi in enumerate(x):
        qc.ry(xi, i)
    return qc


def hardware_efficient_ansatz(n_qubits, n_layers, params):
    qc = QuantumCircuit(n_qubits)
    idx = 0
    for _ in range(n_layers):
        for q in range(n_qubits):
            qc.ry(params[idx], q)
            idx += 1
        for q in range(n_qubits - 1):
            qc.cx(q, q + 1)
    return qc


def full_model_circuit(x, n_qubits, n_layers, params):
    qc = QuantumCircuit(n_qubits)
    qc.compose(angle_encoding_circuit(x, n_qubits), inplace=True)
    qc.compose(hardware_efficient_ansatz(n_qubits, n_layers, params), inplace=True)
    return qc


def build_measurement_circuit(qc):
    circ = QuantumCircuit(qc.num_qubits, qc.num_qubits)
    circ.compose(qc.copy(), inplace=True)
    circ.measure(range(qc.num_qubits), range(qc.num_qubits))
    return circ

# --------------------
# Execution
# --------------------

def run_circuits_and_get_exps(circuits, backend, shots, seed=None):
    job = backend.run(circuits, shots=shots, seed_simulator=seed)
    result = job.result()
    exps = []
    for i in range(len(circuits)):
        counts = result.get_counts(i)
        total = sum(counts.values())
        exp = 0.0
        for bstr, c in counts.items():
            bit = int(bstr[-1])
            z = 1.0 if bit == 0 else -1.0
            exp += z * (c / total)
        exps.append(exp)
    return exps

# --------------------
# Loss / evaluation
# --------------------

def loss_from_expectation(exp, label):
    prob1 = (1 - exp) / 2.0
    return (prob1 - label) ** 2


def evaluate_loss_over_batch(params, X, y, n_qubits, n_layers, backend, shots):
    circuits = []
    for x in X:
        qc = full_model_circuit(x, n_qubits, n_layers, params)
        circuits.append(build_measurement_circuit(qc))
    circuits = transpile(circuits, backend, optimization_level=0)
    exps = run_circuits_and_get_exps(circuits, backend, shots)
    losses = [loss_from_expectation(e, t) for e, t in zip(exps, y)]
    return float(np.mean(losses)), exps

# --------------------
# SPSA
# --------------------

def spsa_gradient_step(params, X, y, n_qubits, n_layers, backend, shots, rng):
    delta = rng.choice([1.0, -1.0], size=len(params))
    c = 0.08
    th_p = params + c * delta
    th_m = params - c * delta
    Lp, _ = evaluate_loss_over_batch(th_p, X, y, n_qubits, n_layers, backend, shots)
    Lm, _ = evaluate_loss_over_batch(th_m, X, y, n_qubits, n_layers, backend, shots)
    grad = ((Lp - Lm) / (2 * c)) * (1 / delta)
    return grad, 0.5 * (Lp + Lm)

# --------------------
# Controller
# --------------------

def outer_loop_controller(params, grad_var, loss_mean, lr, shots, epoch,
                          thresholds, state, rng):
    actions = []
    n_params = len(params)
    n_last = min(thresholds["n_last"], n_params)

    best_loss = state["best_loss"]
    since = state["epochs_since_improve"]

    if best_loss is None or loss_mean < best_loss - thresholds["improve_tol"]:
        best_loss = loss_mean
        since = 0
    else:
        since += 1

    state["best_loss"] = best_loss
    state["epochs_since_improve"] = since

    plateau = (since >= thresholds["patience"] and
               loss_mean > thresholds["loss_target"])

    if not plateau:
        return params, lr, shots, actions, state

    if grad_var > thresholds["grad_noise_floor"]:
        new_shots = min(thresholds["max_shots"], shots * 2)
        if new_shots != shots:
            shots = new_shots
            actions.append("shots_doubled")

    elif thresholds["grad_flat_threshold"] < grad_var <= thresholds["grad_noise_floor"]:
        if thresholds["inject_param_noise"]:
            sigma = thresholds["inject_sigma"]
            params += rng.normal(0, sigma, size=params.shape)
            actions.append(f"params_noised_sigma_{sigma}")

    elif grad_var <= thresholds["grad_flat_threshold"] and n_last > 0:
        sigma = thresholds["reinit_sigma"]
        params[-n_last:] = rng.normal(0, sigma, size=n_last)
        actions.append(f"reinit_last_sigma_{sigma}")

    return params, lr, shots, actions, state

# --------------------
# Training
# --------------------

def run_experiment(seed, mode, n_qubits, n_layers, epochs, shots, lr,
                   n_samples, data_noise, test_size,
                   noise_kind, thresholds):

    rng = default_rng(seed)
    np.random.seed(seed)
    random.seed(seed)

    backend = AerSimulator(noise_model=make_noise_model(noise_kind))

    X, y = make_moons(n_samples=n_samples, noise=data_noise, random_state=seed)
    X = MinMaxScaler((0, 2 * math.pi)).fit_transform(X)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=test_size,
                                          random_state=seed)

    params = rng.normal(0, 0.1, size=n_layers * n_qubits)
    history = {"train_loss": [], "grad_var": [], "shots": []}
    state = {"best_loss": None, "epochs_since_improve": 0}
    actions_log = []

    for ep in range(epochs):
        grad, loss = spsa_gradient_step(params, Xtr, ytr,
                                        n_qubits, n_layers,
                                        backend, shots, rng)
        params = params - lr * grad
        grad_var = float(np.var(grad))

        actions = []
        if mode.startswith("OF"):
            params, lr, shots, actions, state = outer_loop_controller(
                params, grad_var, loss, lr, shots, ep,
                thresholds, state, rng
            )

        history["train_loss"].append(loss)
        history["grad_var"].append(grad_var)
        history["shots"].append(shots)

        if actions:
            actions_log.append({"epoch": ep+1, "actions": actions})

        logger.info("[%s] seed %d ep %d loss %.4f gv %.2e shots %d %s",
                    mode, seed, ep+1, loss, grad_var, shots, actions)

    correct = 0
    for x, t in zip(Xte, yte):
        qc = full_model_circuit(x, n_qubits, n_layers, params)
        exp = run_circuits_and_get_exps([build_measurement_circuit(qc)],
                                        backend, shots)[0]
        pred = 1 if (1-exp)/2 > 0.5 else 0
        if pred == t:
            correct += 1

    acc = correct / len(yte)
    return history, acc, actions_log

# --------------------
# Comparative + save
# --------------------

def comparative_run_and_save(args):
    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    out_root = f"{args.out_prefix}_{ts}"
    os.makedirs(out_root, exist_ok=True)
    os.makedirs(os.path.join(out_root, "data"), exist_ok=True)

    thresholds = {
        "patience": 5,
        "loss_target": 0.28,
        "improve_tol": 1e-4,
        "grad_noise_floor": 5e-5,
        "grad_flat_threshold": 1e-6,
        "max_shots": 1024,
        "n_last": 4,
        "inject_param_noise": False,
        "inject_sigma": 0.01,
        "reinit_sigma": 0.02,
    }

    modes = [
        ("baseline_clean", None),
        ("baseline_noisy", args.noise_kind),
        ("OF_noisy_shots", args.noise_kind),
        ("OF_noisy_inject", args.noise_kind),
    ]

    summary_rows = []

    for mode, nk in modes:
        thresholds["inject_param_noise"] = ("inject" in mode)

        for seed in args.seeds:
            hist, acc, actions = run_experiment(
                seed, mode,
                args.n_qubits, args.n_layers,
                args.epochs, args.shots, args.lr,
                args.n_samples, args.data_noise,
                args.test_size, nk, thresholds
            )

            save_seed_results(hist, acc, actions, args, mode, seed, out_root)

            summary_rows.append({
                "mode": mode,
                "seed": seed,
                "final_acc": acc,
                "final_loss": hist["train_loss"][-1]
            })

    write_summary_csv(summary_rows, os.path.join(out_root, "summary_results.csv"))
    print("Resultados guardados en:", out_root)

# --------------------
# CLI
# --------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_qubits", type=int, default=4)
    parser.add_argument("--n_layers", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--shots", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0,1,2,3,4])
    parser.add_argument("--n_samples", type=int, default=256)
    parser.add_argument("--data_noise", type=float, default=0.05)
    parser.add_argument("--test_size", type=float, default=0.25)
    parser.add_argument("--noise_kind",
                        type=str,
                        default="light",
                        choices=["light","heavy","none"])
    parser.add_argument("--out_prefix", type=str, default="run")
    args = parser.parse_args()

    comparative_run_and_save(args)

if __name__ == "__main__":
    main()
