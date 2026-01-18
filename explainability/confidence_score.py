from typing import Iterable
import numpy as np


def calculate_confidence(distances: Iterable[float]) -> float:
	"""Convert distances (L2) to a 0-1 confidence score.

	Uses a simple inverse transform: similarity = 1 / (1 + distance).
	The returned confidence is the mean similarity across distances.
	"""
	d = np.asarray(list(distances), dtype=float)
	if d.size == 0:
		return 0.0

	sim = 1.0 / (1.0 + d)
	# clamp to [0,1]
	score = float(sim.mean())
	if score < 0:
		score = 0.0
	if score > 1:
		score = 1.0
	return score
