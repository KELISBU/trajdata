from copy import deepcopy
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from trajdata import filtering
from trajdata.augmentation import BatchAugmentation
from trajdata.caching.df_cache import DataFrameCache
from trajdata.data_structures.agent import AgentMetadata, FixedExtent, VariableExtent
from trajdata.data_structures.batch import AgentBatch
from trajdata.data_structures.batch_element import AgentBatchElement
from trajdata.data_structures.collation import agent_collate_fn
from trajdata.data_structures.scene import SceneTimeAgent
from trajdata.data_structures.scene_metadata import Scene
from trajdata.data_structures.state import StateArray
from trajdata.dataset import UnifiedDataset
from trajdata.simulation.sim_cache import SimulationCache
from trajdata.simulation.sim_df_cache import SimulationDataFrameCache
from trajdata.simulation.sim_metrics import SimMetric
from trajdata.simulation.sim_stats import SimStatistic

import warnings
import json

class SimulationScene:
    def __init__(
        self,
        env_name: str,
        scene_name: str,
        scene: Scene,
        dataset: UnifiedDataset,
        init_timestep: int = 0,
        freeze_agents: bool = True,
        return_dict: bool = False,
    ) -> None:
        if not freeze_agents:
            raise NotImplementedError(
                (
                    "Agents that change over time (i.e., following the original dataset) "
                    "are not handled yet internally. Please set freeze_agents=True."
                )
            )

        self.env_name: str = env_name
        self.scene_name: str = scene_name
        self.scene: Scene = deepcopy(scene)
        self.dataset: UnifiedDataset = dataset
        self.init_scene_ts: int = init_timestep
        self.freeze_agents: bool = freeze_agents
        self.return_dict: bool = return_dict
        self.scene_ts: int = self.init_scene_ts

        agents_present: List[AgentMetadata] = self.scene.agent_presence[self.scene_ts]
        self.agents: List[AgentMetadata] = filtering.agent_types(
            agents_present, self.dataset.no_types, self.dataset.only_types
        )
        self.agent_ref_dict = {agent.name: {"current": "", "left": "", "right": "","all":""} for agent in self.agents}
        

        if len(self.agents) == 0:
            raise ValueError(
                (
                    f"Initial timestep {self.scene_ts} contains no agents after filtering. "
                    "Please choose another initial timestep."
                )
            )

        if self.freeze_agents:
            self.scene.agent_presence = self.scene.agent_presence[
                : self.init_scene_ts + 1
            ]
            self.scene.agents = self.agents

        # Note this order of operations is important, we first instantiate
        # the cache with the copied scene_info + modified agents list.
        # Then, we change the env_name and etc later during finalization
        # (if we did it earlier then the cache would go looking inside
        # the sim folder for scene data rather than the original scene
        # data location).
        if self.dataset.cache_class == DataFrameCache:
            self.cache: SimulationCache = SimulationDataFrameCache(
                dataset.cache_path,
                self.scene,
                init_timestep,
                dataset.augmentations,
            )

        self.batch_augments: Optional[List[BatchAugmentation]] = None
        if dataset.augmentations:
            self.batch_augments = [
                batch_aug
                for batch_aug in dataset.augmentations
                if isinstance(batch_aug, BatchAugmentation)
            ]
    def initialize_ref_dict(self):
        for agent in self.agents:
            self.agent_ref_dict[agent.name] = {"current": "","all":"", "left": "", "right": ""}
   
    def load_ref_dict(self, ref_polyline_ids=None, all_ref_polyline_ids=None):
        """
        Load reference polyline IDs into agent dictionary.
        Can load either ref_polyline_ids or all_ref_polyline_ids or both if provided.
        
        Args:
            ref_polyline_ids: Optional list of reference polyline IDs
            all_ref_polyline_ids: Optional list of JSON strings containing lists of polyline ID lists
        """
        num_agents = len(self.agents)
        
        # Initialize the dictionary for all agents if it doesn't exist
        for agent in self.agents:
            if agent.name not in self.agent_ref_dict:
                self.agent_ref_dict[agent.name] = {
                    "current": "",
                    "all": ""
                }
        # Load ref_polyline_ids if provided
        if ref_polyline_ids is not None:
            if len(ref_polyline_ids) != num_agents:
                warnings.warn(f"The number of agents ({num_agents}) and ref_polyline_ids ({len(ref_polyline_ids)}) do not match! Won't load ref_polyline_ids")
            else:
                for i, agent in enumerate(self.agents):
                    self.agent_ref_dict[agent.name]["current"] = ref_polyline_ids[i]
        
        # Load all_ref_polyline_ids if provided
        if all_ref_polyline_ids is not None:
            if len(all_ref_polyline_ids) != num_agents:
                warnings.warn(f"The number of agents ({num_agents}) and all_ref_polyline_ids ({len(all_ref_polyline_ids)}) do not match! Won't load all_ref_polyline_ids")
            else:
                for i, agent in enumerate(self.agents):
                    self.agent_ref_dict[agent.name]["all"] = all_ref_polyline_ids[i]
    def reset(self) -> Union[AgentBatch, Dict[str, Any]]:
        self.scene_ts: int = self.init_scene_ts
        return self.get_obs()
    
    def update_agent_ref(self, agent_name, change_lane_state):
        # Update the agent reference based on the change_lane_state (0, 1, -1)
        if "ego" in agent_name:
            return
        if change_lane_state == 1:  # Change to right
            self.agent_ref_dict[agent_name]["left"] = self.agent_ref_dict[agent_name]["current"]
            self.agent_ref_dict[agent_name]["current"] = self.agent_ref_dict[agent_name]["right"]
            self.agent_ref_dict[agent_name]["right"] = ""
        elif change_lane_state == -1:  # Change to left
            self.agent_ref_dict[agent_name]["right"] = self.agent_ref_dict[agent_name]["current"]
            self.agent_ref_dict[agent_name]["current"] = self.agent_ref_dict[agent_name]["left"]
            self.agent_ref_dict[agent_name]["left"] = ""

    def step(
        self,
        new_xyzh_dict: Dict[str, StateArray],
        return_obs=True,
    ) -> Union[AgentBatch, Dict[str, Any]]:
        self.scene_ts += 1

        self.cache.append_state(new_xyzh_dict)

        if not self.freeze_agents:
            agents_present: List[AgentMetadata] = self.scene.agent_presence[
                self.scene_ts
            ]
            self.agents: List[AgentMetadata] = filtering.agent_types(
                agents_present, self.dataset.no_types, self.dataset.only_types
            )

            self.scene.agent_presence[self.scene_ts] = self.agents
        else:
            self.scene.agent_presence.append(self.agents)

        if return_obs:  
            return self.get_obs()

    def get_obs(
        self, collate: bool = True, get_map: bool = True, get_full_fut_traj: bool = False
    ) -> Union[AgentBatch, Dict[str, Any]]:
        agent_data_list: List[AgentBatchElement] = list()
        self.cache.set_obs_format(self.dataset.obs_format)

        for agent in self.agents:
            scene_time_agent = SceneTimeAgent(
                self.scene, self.scene_ts, self.agents, agent, self.cache
            )
            batch_element: AgentBatchElement = AgentBatchElement(
                self.cache,
                -1,  # Not used
                scene_time_agent,
                history_sec=self.dataset.history_sec,
                future_sec=self.dataset.future_sec,
                agent_interaction_distances=self.dataset.agent_interaction_distances,
                incl_robot_future=False,
                incl_raster_map=get_map and self.dataset.incl_raster_map,
                raster_map_params=self.dataset.raster_map_params,
                map_api=self.dataset._map_api,
                vector_map_params=self.dataset.vector_map_params,
                state_format=self.dataset.state_format,
                standardize_data=self.dataset.standardize_data,
                standardize_derivatives=self.dataset.standardize_derivatives,
                max_neighbor_num=self.dataset.max_neighbor_num,
            )
            agent_data_list.append(batch_element)
            # Modify here to fix the same reference of same agent
            for key, extra_fn in self.dataset.extras.items():
                if key == "closest_lane_point":
                    batch_element.extras[key] = extra_fn(
                        batch_element,
                        ref_polyline_ids = self.agent_ref_dict[agent.name]["current"],
                        all_ref_polyline_ids = self.agent_ref_dict[agent.name]["all"]
                    )
                    #Should determine None and "" 
                    if self.agent_ref_dict[agent.name]["current"] == "":
                        self.agent_ref_dict[agent.name]["current"] = batch_element.extras[key]["ref_polyline_ids"]
                        self.agent_ref_dict[agent.name]["all"] = batch_element.extras[key]["all_ref_polyline_ids"]
                elif "get_full_fut_traj" in key:
                    if get_full_fut_traj:
                        batch_element.extras[key] = extra_fn(batch_element)
                else:
                    batch_element.extras[key] = extra_fn(batch_element)
            for transform_fn in self.dataset.transforms:
                batch_element = transform_fn(batch_element)

            if not self.dataset.vector_map_params.get("collate", False):
                batch_element.vec_map = None

            # Need to reset transformations for each agent since each
            # AgentBatchElement transforms (standardizes) the cache.
            self.cache.reset_obs_frame()

        if collate:
            return agent_collate_fn(
                agent_data_list,
                return_dict=self.return_dict,
                pad_format="outside",
                batch_augments=self.batch_augments,
            )
        else:
            return agent_data_list

    def get_metrics(self, metrics: List[SimMetric]) -> Dict[str, Dict[str, float]]:
        return self.cache.calculate_metrics(
            metrics, ts_range=(self.init_scene_ts + 1, self.scene_ts)
        )

    def get_stats(
        self, stats: List[SimStatistic]
    ) -> Dict[str, Dict[str, Tuple[np.ndarray, np.ndarray]]]:
        return self.cache.calculate_stats(
            stats, ts_range=(self.init_scene_ts + 1, self.scene_ts)
        )

    def finalize(self) -> None:
        # We only change the agent's last timestep here because we use it
        # earlier to check if the agent has any future data from the original
        # dataset.
        for agent in self.agents:
            agent.last_timestep = self.scene_ts

        self.scene.length_timesteps = self.scene_ts + 1

        self.scene.agent_presence = self.scene.agent_presence[: self.scene_ts + 1]

        self.scene.env_metadata.name = self.env_name
        self.scene.env_name = self.env_name
        self.scene.name = self.scene_name

    def save(self) -> None:
        self.dataset.env_cache.save_scene(self.scene)
        self.cache.save_sim_scene(self.scene)

    def add_new_agents(self, agent_data: List[Tuple]):
        existing_agent_names = [agent.name for agent in self.agents]
        agent_data = [
            agent for agent in agent_data if agent[0] not in existing_agent_names
        ]
        if len(agent_data) > 0:
            self.cache.add_agents(agent_data)
            for data in agent_data:
                name, state, ts0, agent_type, extent = data
                metadata = AgentMetadata(
                    name=name,
                    agent_type=agent_type,
                    first_timestep=ts0,
                    last_timestep=ts0 + state.shape[0] - 1,
                    extent=FixedExtent(
                        length=extent[0], width=extent[1], height=extent[2]
                    ),
                )
                self.agents.append(metadata)
def decode_ref_polylines(all_ref_polyline_ids):
    decoded_list = []
    for encoded_item in all_ref_polyline_ids:
        first_decode = json.loads(encoded_item)
        inner_lists = [json.loads(inner_str) for inner_str in first_decode]
        decoded_list.append(inner_lists)
    return decoded_list