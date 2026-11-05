"""Save the policy and observation normalization needed for evaluation."""
import json

import jax
import jax_dataclasses as jdc
import numpy as np


def save_policy(path, agent, metadata):
    leaves = jax.tree.leaves((agent.params, agent.obs_stats))
    arrays = {f"array_{i}": np.asarray(x) for i, x in enumerate(leaves)}
    tmp = path.with_suffix(".tmp")
    with tmp.open("wb") as handle:
        np.savez(handle, metadata=json.dumps(metadata), **arrays)
    tmp.replace(path)


def restore_policy(data, agent):
    leaves, structure = jax.tree.flatten((agent.params, agent.obs_stats))
    if len(data.files) != len(leaves) + 1:
        raise ValueError("Unexpected number of checkpoint arrays")
    arrays = []
    for i, template in enumerate(leaves):
        array = data[f"array_{i}"]
        if array.shape != template.shape or array.dtype != template.dtype:
            raise ValueError(f"Incompatible checkpoint array {i}")
        arrays.append(jax.device_put(array))
    params, obs_stats = jax.tree.unflatten(structure, arrays)
    return jdc.replace(agent, params=params, obs_stats=obs_stats)
