import numpy as np


class SparseArray:
    """
    Custom sparse array implementation
    """
    def __init__(self):
        self.reset()

    def reset(self) -> None:
        self._shape = None
        self._dtype = None
        self._store_raw = False
        self._arr = None
        self._inds = None
        self._vals = None
        self._nonzero_count = 0
        self._nbytes = 0

    def set(self, arr: np.ndarray) -> None:
        """
        Convert and store input array in sparse format.
        :param arr: Numpy array
        """
        self.reset()
        self._shape = arr.shape
        self._dtype = arr.dtype

        # Get non-zero indices
        indices = np.nonzero(arr)

        # Get non-zero count
        self._nonzero_count = len(indices[0]) if len(indices) else 0

        # Get storage type of indices based on maximum dimension
        max_dim = max(self._shape)
        dtypes = [np.uint8, np.uint16, np.uint32, np.uint64]
        for dtype in dtypes:
            if max_dim < np.iinfo(dtype).max:
                self._ind_dtype = dtype
                break

        # Check if we should store raw array (i.e. if storing
        # each non-zero element will take more memory than
        # storing raw array, store raw array)
        self._nbytes = arr.nbytes
        indices_nbytes = len(indices) * len(indices[0]) * indices[0].itemsize if len(indices) else 0
        sparse_nbytes = indices_nbytes + self._nonzero_count * arr.itemsize
        if sparse_nbytes < self._nbytes:
            self._inds = tuple([ind.astype(self._ind_dtype) for ind in indices])
            self._vals = arr[self._inds]
            self._nbytes = sparse_nbytes
        else:
            self._store_raw = True
            self._arr = arr

    @property
    def nonzero_count(self) -> int:
        """
        Gets number of non-zero elements in stored array.
        :returns: non-zero element count
        """
        return self._nonzero_count

    @property
    def nbytes(self) -> int:
        """
        Returns number of bytes used to store array internally.
        :returns: bytes
        """
        return self._nbytes

    def to_numpy(self) -> np.ndarray:
        """
        Returns stored sparse array in dense numpy array format.
        :returns: numpy array
        """
        if self._store_raw:
            return self._arr
        arr = np.zeros(self._shape, dtype=self._dtype)
        arr[self._inds] = self._vals
        return arr


class ObservationCompressor:
    def __init__(self, buffer_size=128, n_envs=8, n_stack=4, image_shape=(84, 84)):
        self.buffer_size = buffer_size
        self.n_envs = n_envs
        self.n_stack = n_stack
        self.image_shape = image_shape
        self.dtype = np.uint8

        self.obs_size = (self.buffer_size * self.n_envs) // self.n_stack
        self.obs = np.zeros((self.obs_size,) + self.image_shape, dtype=self.dtype)
        self.sparse_obs = [
            [SparseArray() for _ in range(self.n_stack - 1)]
            for _ in range(self.obs_size)
        ]
        self.obs_inds = -1 * np.ones(
            (self.buffer_size * self.n_envs, self.n_stack), dtype=np.int32
        )

    def get_nonzero_count(self):
        total = 0
        for stack in self.sparse_obs:
            for sa in stack:
                total += sa.nonzero_count()
        return total

    def _add_obs(self, obs, pos):
        assert obs.shape == (self.n_stack,) + self.image_shape
        assert 0 <= pos < self.buffer_size * self.n_envs

        # Get positions in self.obs array and self.sparse_obs
        obs_pos = pos // self.n_stack
        sparse_obs_pos = pos % self.n_stack

        # Get start and end position of the current env section
        #  NOTE: The self.obs_inds indexes as if we did not compress
        #        by a factor of self.n_stack. Therefore, each env
        #        section is assumed to be of size self.buffer_size
        env_begin_pos = (pos // self.buffer_size) * self.buffer_size
        env_end_pos = ((pos // self.buffer_size) + 1) * self.buffer_size

        # Zero out the next self.n_stack - 1 positions to ensure
        #   future references to the current position are erased
        self.obs_inds[pos : min(pos + self.n_stack, env_end_pos)] = -1

        # Create reference obs_inds entry for input obs, ensuring that any
        #  images of all 0's is mapped to idx=(-1)
        for i in range(self.n_stack):
            if np.any(obs[i]):
                new_pos = pos - (self.n_stack - 1 - i)
                # ensure obs_inds doesn't go back to previous env slice
                self.obs_inds[pos, i] = new_pos if new_pos >= env_begin_pos else -1

        # Store the observation
        observation = np.array(obs[-1]).copy()

        # If sparse_obs_pos == 0, then add full ob to self.obs
        if sparse_obs_pos == 0:
            self.obs[obs_pos] = observation
        else:
            # Diff obs with self.obs[obs_pos]
            obs_diff = observation.astype(np.int16) - self.obs[obs_pos].astype(np.int16)
            self.sparse_obs[obs_pos][sparse_obs_pos - 1].set(obs_diff)

    def add(self, obs, pos):
        assert (
            obs.shape
            == (
                self.n_envs,
                self.n_stack,
            )
            + self.image_shape
        )

        # Add each obs from each env separately, ensuring to set global pos
        for i in range(self.n_envs):
            self._add_obs(obs[i], (self.buffer_size * i) + pos)

    def _get_obs(self, idx):
        # Get observation at index idx from self.obs and/or self.sparse_obs
        if idx == -1:
            return np.zeros(self.image_shape)

        assert 0 <= idx < self.buffer_size * self.n_envs

        # Get positions in self.obs array and self.sparse_obs
        obs_idx = idx // self.n_stack
        sparse_obs_idx = idx % self.n_stack

        obs_base = self.obs[obs_idx]

        if sparse_obs_idx == 0:
            return obs_base

        # Diff obs with self.obs[obs_pos]
        return obs_base + self.sparse_obs[obs_idx][sparse_obs_idx - 1].to_numpy()

    def get(self, batch_inds):
        # Get observation at batch_inds relative to self.obs_inds
        assert all([0 <= idx < self.buffer_size * self.n_envs for idx in batch_inds])

        batch_size = (
            batch_inds.shape[0]
            if isinstance(batch_inds, np.ndarray)
            else len(batch_inds)
        )
        obs = np.zeros(
            (
                batch_size,
                self.n_stack,
            )
            + self.image_shape,
            dtype=self.dtype,
        )

        for i in range(batch_size):
            inds = self.obs_inds[batch_inds[i]]
            for j in range(self.n_stack):
                obs[i, j] = self._get_obs(inds[j]).astype(self.dtype)
        return obs
