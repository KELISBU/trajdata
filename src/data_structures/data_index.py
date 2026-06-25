from typing import List, Tuple, Union

import numpy as np
from tqdm import tqdm


class DataIndex:
    """The data index is effectively a big list of tuples taking the form:

    [(scene_path, total_index_len, valid_scene_ts)] for scene-centric data, or
    [(scene_path, total_index_len, [(agent_name, valid_agent_ts)])] for agent-centric data
    """

    def __init__(
        self,
        data_index: Union[
            List[Tuple[str, int, np.ndarray]],
            List[Tuple[str, int, List[Tuple[str, np.ndarray]]]],
        ],
        verbose: bool = False,
    ) -> None:
        scene_paths, full_index_len, _ = zip(*data_index)

        self._cumulative_lengths: np.ndarray = np.concatenate(
            ([0], np.cumsum(full_index_len))
        )
        self._len: int = self._cumulative_lengths[-1].item()

        self._scene_paths: np.ndarray = np.array(scene_paths).astype(np.string_)

    def __len__(self) -> int:
        return self._len

    def __getitem__(self, index: int) -> Tuple[str, int, int]:
        scene_idx: int = (
            np.searchsorted(self._cumulative_lengths, index, side="right").item() - 1
        )

        scene_path: str = str(self._scene_paths[scene_idx], encoding="utf-8")
        scene_elem_index: int = index - self._cumulative_lengths[scene_idx].item()
        return (scene_path, scene_idx, scene_elem_index)


class AgentDataIndex(DataIndex):
    def __init__(
        self,
        data_index: List[Tuple[str, int, List[Tuple[str, np.ndarray]]]],
        verbose: bool = False,
    ) -> None:
        super().__init__(data_index)

        agent_timesteps: List[List[Tuple[str, np.ndarray]]] = [
            agent_ts_index for _, _, agent_ts_index in data_index
        ]

        self._agent_ids: List[np.ndarray] = list()
        self._agent_times: List[np.ndarray] = list()
        self._cumulative_scene_lengths: List[np.ndarray] = list()
        for scene_data_index in tqdm(
            agent_timesteps, desc="Structuring Agent Data Index", disable=not verbose
        ):
            agent_ids, agent_times = zip(*scene_data_index)

            self._agent_ids.append(np.array(agent_ids).astype(np.string_))

            agent_ts: np.ndarray = np.stack(agent_times)
            self._agent_times.append(agent_ts)
            self._cumulative_scene_lengths.append(
                np.concatenate(([0], np.cumsum(agent_ts[:, 1] - agent_ts[:, 0] + 1)))
            )

    def __getitem__(self, index: int) -> Tuple[str, str, int]:
        scene_path, scene_idx, scene_elem_index = super().__getitem__(index)

        agent_idx: int = (
            np.searchsorted(
                self._cumulative_scene_lengths[scene_idx],
                scene_elem_index,
                side="right",
            ).item()
            - 1
        )

        agent_id: str = str(self._agent_ids[scene_idx][agent_idx], encoding="utf-8")

        agent_timestep: int = (
            scene_elem_index
            - self._cumulative_scene_lengths[scene_idx][agent_idx].item()
            + self._agent_times[scene_idx][agent_idx, 0]
        ).item()

        assert (
            self._agent_times[scene_idx][agent_idx, 0]
            <= agent_timestep
            <= self._agent_times[scene_idx][agent_idx, 1]
        )

        return scene_path, agent_id, agent_timestep
# import os
# import lmdb
# import pickle
# import numpy as np
# from typing import List, Tuple
# from tqdm import tqdm

# Assuming DataIndex is your superclass
# class AgentDataIndex(DataIndex):
#     def __init__(self, data_index: List[Tuple[str, int, List[Tuple[str, np.ndarray]]]], verbose: bool = False,name:str = "", lazy: bool = False):
#         super().__init__(data_index)
        
#         self.lazy = lazy
#         self.verbose = verbose
#         self.lmdb_path = f"/net/ca-home1/home/mai/wjchang/lmdb/{name}_agent_data_index.lmdb"
        
#         if self.lazy:
#             # Check if LMDB exists
#             if not os.path.exists(self.lmdb_path):
#                 # Initialize LMDB and populate it
#                 print(f"Write to LMDB {self.lmdb_path}")
#                 self.env = lmdb.open(self.lmdb_path, map_size=int((1099511627776*6/32)))
#                 self.lazy=False
#                 self.write_to_lmdb(data_index)
#                 self.lazy=True
                
#             else:
#                 print(f"Opening existing LMDB environment from {self.lmdb_path}.")
#                 self.env = lmdb.open(self.lmdb_path, readonly=True, lock = False)
#         else:
#             self.preprocess_data_index(data_index)
    
#     def write_to_lmdb(self, data_index):
#         self.preprocess_data_index(data_index)
#         with self.env.begin(write=True) as txn:
#             for idx,_ in enumerate(tqdm(range(self._len), desc="Writing to LMDB", disable=not self.verbose)):
#                 scene_path, scene_idx, scene_elem_index = super().__getitem__(idx)
#                 agent_idx: int = (
#                 np.searchsorted(
#                     self._cumulative_scene_lengths[scene_idx],
#                     scene_elem_index,
#                     side="right",
#                 ).item()
#                 - 1
#                 )

#                 agent_id: str = str(self._agent_ids[scene_idx][agent_idx], encoding="utf-8")

#                 agent_timestep: int = (
#                     scene_elem_index
#                     - self._cumulative_scene_lengths[scene_idx][agent_idx].item()
#                     + self._agent_times[scene_idx][agent_idx, 0]
#                 ).item()

#                 assert (
#                     self._agent_times[scene_idx][agent_idx, 0]
#                     <= agent_timestep
#                     <= self._agent_times[scene_idx][agent_idx, 1]
#                 )
#                 txn.put(str(idx).encode("utf-8"), pickle.dumps((scene_path, agent_id, agent_timestep)))
#         del self._agent_ids,self._agent_times,self._cumulative_scene_lengths
                

#     def preprocess_data_index(self,data_index):
#         agent_timesteps: List[List[Tuple[str, np.ndarray]]] = [
#             agent_ts_index for _, _, agent_ts_index in data_index
#         ]

#         self._agent_ids: List[np.ndarray] = list()
#         self._agent_times: List[np.ndarray] = list()
#         self._cumulative_scene_lengths: List[np.ndarray] = list()
#         for scene_data_index in tqdm(
#             agent_timesteps, desc="Structuring Agent Data Index", disable=not self.verbose
#         ):
#             agent_ids, agent_times = zip(*scene_data_index)

#             self._agent_ids.append(np.array(agent_ids).astype(np.string_))

#             agent_ts: np.ndarray = np.stack(agent_times)
#             self._agent_times.append(agent_ts)
#             self._cumulative_scene_lengths.append(
#                 np.concatenate(([0], np.cumsum(agent_ts[:, 1] - agent_ts[:, 0] + 1)))
#             )

#     def __getitem__(self, index):
#         scene_path, scene_idx, scene_elem_index = super().__getitem__(index)

#         if self.lazy:
#             with self.env.begin() as txn:
#                 # Retrieve the tuple (scene_path, agent_id, agent_timestep) from LMDB
#                 scene_path, agent_id, agent_timestep = pickle.loads(txn.get(str(index).encode("utf-8")))
#                 return scene_path, agent_id, agent_timestep
#         else:
#             agent_idx: int = (
#                 np.searchsorted(
#                     self._cumulative_scene_lengths[scene_idx],
#                     scene_elem_index,
#                     side="right",
#                 ).item()
#                 - 1
#             )

#             agent_id: str = str(self._agent_ids[scene_idx][agent_idx], encoding="utf-8")

#             agent_timestep: int = (
#                 scene_elem_index
#                 - self._cumulative_scene_lengths[scene_idx][agent_idx].item()
#                 + self._agent_times[scene_idx][agent_idx, 0]
#             ).item()

#             assert (
#                 self._agent_times[scene_idx][agent_idx, 0]
#                 <= agent_timestep
#                 <= self._agent_times[scene_idx][agent_idx, 1]
#             )

#             return scene_path, agent_id, agent_timestep

class SceneDataIndex(DataIndex):
    def __init__(
        self, data_index: List[Tuple[str, int, np.ndarray]], verbose: bool = False
    ) -> None:
        super().__init__(data_index)

        self.scene_ts: List[np.ndarray] = [
            valid_ts
            for _, _, valid_ts in tqdm(
                data_index, desc="Structuring Scene Data Index", disable=not verbose
            )
        ]

    def __getitem__(self, index: int) -> Tuple[str, int]:
        scene_path, scene_idx, scene_elem_index = super().__getitem__(index)

        scene_ts: int = self.scene_ts[scene_idx][scene_elem_index].item()

        return scene_path, scene_ts
