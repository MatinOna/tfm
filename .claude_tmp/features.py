from pathlib import Path
import numpy as np
import pandas as pd
import pretty_midi

PIECE_BY_PITCH = {
    36: "kick",
    38: "snare", 40: "snare", 37: "snare",
    42: "hihat_closed", 22: "hihat_closed", 44: "hihat_closed",
    46: "hihat_open", 26: "hihat_open",
    48: "tom_high", 50: "tom_high",
    45: "tom_mid", 47: "tom_mid",
    43: "tom_floor", 58: "tom_floor",
    49: "crash", 55: "crash", 57: "crash", 52: "crash",
    51: "ride", 59: "ride", 53: "ride",
}

PIECES = ["kick", "snare", "hihat_closed", "hihat_open",
          "tom_high", "tom_mid", "tom_floor", "crash", "ride"]


def load_notes(midi_path: Path) -> pd.DataFrame:
    pm = pretty_midi.PrettyMIDI(str(midi_path))
    notes = pm.instruments[0].notes
    df = pd.DataFrame({
        "onset_s": [n.start for n in notes],
        "end_s": [n.end for n in notes],
        "pitch": [n.pitch for n in notes],
        "velocity": [n.velocity for n in notes],
    })
    df["piece"] = df["pitch"].map(PIECE_BY_PITCH)
    return df.sort_values("onset_s").reset_index(drop=True)

BEATS_PER_BAR = 4


def n_bars(notes: pd.DataFrame, bpm: float) -> float:
    return notes["end_s"].max() * bpm / 60 / BEATS_PER_BAR


def activity_features(notes: pd.DataFrame, bpm: float) -> dict:
    bars = n_bars(notes, bpm)
    counts = notes["piece"].value_counts().reindex(PIECES, fill_value=0)
    total = counts.sum()

    features = {}
    for piece in PIECES:
        features[f"density_{piece}"] = float(counts[piece] / bars)
    for piece in PIECES:
        features[f"share_{piece}"] = float(counts[piece] / total)
    features["density_total"] = float(total / bars)
    return features

STEPS_PER_BEAT = 4
STEPS_PER_BAR = STEPS_PER_BEAT * BEATS_PER_BAR


def grid_positions(notes: pd.DataFrame, bpm: float) -> pd.DataFrame:
    out = notes.copy()
    out["beat"] = out["onset_s"] * bpm / 60
    steps = out["beat"] * STEPS_PER_BEAT
    nearest = np.round(steps)
    out["step"] = nearest.astype(int)
    out["pos_in_bar"] = out["step"] % STEPS_PER_BAR
    out["offset"] = steps - nearest
    return out


def microtiming_features(grid: pd.DataFrame) -> dict:
    off = grid["offset"]
    return {
        "micro_mean": float(off.mean()),
        "micro_abs_mean": float(off.abs().mean()),
        "micro_std": float(off.std(ddof=0)),
    }


POSITION_GROUPS = {
    "pos_strong": [0, 8],
    "pos_backbeat": [4, 12],
    "pos_8th_offbeat": [2, 6, 10, 14],
    "pos_16th_offbeat": [1, 3, 5, 7, 9, 11, 13, 15],
}


def position_features(grid: pd.DataFrame) -> dict:

    pos = grid["pos_in_bar"]
    return {name: float(pos.isin(steps).mean()) for name, steps in POSITION_GROUPS.items()}


SIMULTANEITY_BEATS = 1 / 16
IOI_EDGES = [0, 0.1875, 0.29, 0.42, 0.75, np.inf]
IOI_NAMES = ["ioi_short", "ioi_16th", "ioi_triplet", "ioi_8th", "ioi_quarter_plus"]


def onset_times_beats(grid: pd.DataFrame) -> np.ndarray:
    beats = np.sort(grid["beat"].to_numpy())
    keep = np.concatenate([[True], np.diff(beats) >= SIMULTANEITY_BEATS])
    return beats[keep]


def ioi_features(grid: pd.DataFrame) -> dict:
    iois = np.diff(onset_times_beats(grid))
    if len(iois) == 0:
        return {name: np.nan for name in IOI_NAMES}
    counts, _ = np.histogram(iois, bins=IOI_EDGES)
    return {name: float(c / len(iois)) for name, c in zip(IOI_NAMES, counts)}


LHL_WEIGHTS = np.array([0, -4, -3, -4, -2, -4, -3, -4, -1, -4, -3, -4, -2, -4, -3, -4])

# Metrical weights of the 16 positions of a 4/4 bar (Longuet-Higgins & Lee, 1984,
# as given by Fitch & Rosenfeld, 2007): 1 e y a 2 e y a 3 e y a 4 e y a
def syncopation_features(grid: pd.DataFrame, bpm: float) -> dict:
    """Syncopation per bar, based on Longuet-Higgins & Lee (1984).
    """
    occupied = np.unique(grid["step"].to_numpy())
    total = 0.0
    for s, s_next in zip(occupied[:-1], occupied[1:]):
        if s_next - s > 1:
            rest_weight = LHL_WEIGHTS[np.arange(s + 1, s_next) % STEPS_PER_BAR].max()
            note_weight = LHL_WEIGHTS[s % STEPS_PER_BAR]
            if rest_weight > note_weight:
                total += rest_weight - note_weight
    return {"syncopation": float(total / n_bars(grid, bpm))}


DYNAMICS_PIECES = ["kick", "snare", "hihat_closed"]

def velocity_stats(velocities, prefix) -> dict:
    if len(velocities) == 0:
        return {f"{prefix}_mean": 0.0, f"{prefix}_std": 0.0, f"{prefix}_range": 0.0}
    return {f"{prefix}_mean": float(velocities.mean()),
            f"{prefix}_std": float(velocities.std(ddof=0)),
            f"{prefix}_range": float(velocities.max() - velocities.min())}

def dynamics_features(notes) -> dict:
    features = velocity_stats(notes["velocity"], "vel")
    for piece in DYNAMICS_PIECES:
        piece_velocities = notes.loc[notes["piece"] == piece, "velocity"]
        features.update(velocity_stats(piece_velocities, f"vel_{piece}"))
    return features