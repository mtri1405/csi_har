import numpy as np


# The 192 complex samples are treated as an ordered HT40 band from -96 to +95.
# Six guard-band samples are removed at each edge; index 96 is the DC carrier.
EDGE_SUBCARRIERS = 6
DC_SUBCARRIER_INDEX = 96


def remove_unusable_subcarriers(data, remove_zero_columns=True):
    """Remove fixed guard/DC carriers and columns that are zero for every frame."""
    if data.ndim != 2 or data.shape[1] != 192:
        raise ValueError(
            f"Expected CSI amplitude data with shape (frames, 192), got {data.shape}."
        )

    remove_indices = set(range(EDGE_SUBCARRIERS))
    remove_indices.update(range(192 - EDGE_SUBCARRIERS, 192))
    remove_indices.add(DC_SUBCARRIER_INDEX)

    if remove_zero_columns:
        zero_columns = np.all(data == 0, axis=0)
        remove_indices.update(np.flatnonzero(zero_columns).tolist())

    keep_indices = [
        index for index in range(data.shape[1]) if index not in remove_indices
    ]
    return data[:, keep_indices], sorted(remove_indices), keep_indices
