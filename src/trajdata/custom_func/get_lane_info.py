import random
from typing import (
    Any, Dict, Iterable, List, Mapping, Optional, Sequence,
    Tuple, Union, Callable
)
import json

import matplotlib.pyplot as plt
import numpy as np
import torch

from trajdata.data_structures.batch_element import AgentBatchElement
from trajdata.maps import VectorMap
from trajdata.maps.vec_map_elements import RoadLane

from trajdata.maps.ref_utils import ref_path
from trajdata.maps.centerline_utils import *
from trajdata.utils.arr_utils import transform_angles_np, transform_coords_np
from trajdata.utils.state_utils import transform_state_np_2d
# Add module-level constants


DEFAULT_VEC_MAP_PARAMS = {
    "NUM_FUTURE_LANES": 6,
    "FIND_ALL_POS_REFS": True,
    "CENTERLINE_LENGTH": 150,
    "CENTERLINE_LOOKAHEAD": 150, 
    "CENTERLINE_LOOKBACK": 50,
    
    "FIND_CLOSEST_DIST_THRESHOLD": 2.5,
    "FIND_CLOSEST_MAX_THRESHOLD": 5.0,
    "FIND_ALL_DIST_THRESHOLD": 10.0,
    "FIND_ALL_MAX_THRESHOLD": 20.0,
    
    "DEFAULT_MAX_HEADING_ERROR": np.pi / 4.0,
    "DFS_THRESHOLD_FUTURE": 150,
    "DFS_THRESHOLD_PAST": 50,
    "DFS_DEFAULT_THRESHOLD": 30,
    
    "EXTEND_DISTANCE": 100.0,
    # "N_INTERPOLATED_POINTS": 5,
    # "EXTEND_DISTANCE_THRESHOLD": 5.0,
    # "EXTEND_HEADING_THRESHOLD": np.pi / 8,
    # "MIN_DISTANCE_LANE": 3.5,
}
def get_lane_info(element: AgentBatchElement,
                  ref_polyline_ids: str = "",
                  all_ref_polyline_ids: str = "",
                  VEC_MAP_PARAMS: Dict = None
                  ) -> Dict:
    """
    Main entry point to retrieve lane info for a predicted agent.
    """
    if VEC_MAP_PARAMS is None:
        VEC_MAP_PARAMS = DEFAULT_VEC_MAP_PARAMS

    fut_len = int(round(element.future_sec[0] / element.dt))

    if ref_polyline_ids == "None":
        return create_empty_return_dict(VEC_MAP_PARAMS)

    # Transform agent history/future to world frame
    world_from_agent = np.linalg.inv(element.agent_from_world_tf)
    agent_past_xyzh_world = transform_state_np_2d(element.agent_history_np, world_from_agent).as_format("x,y,z,h")

    if len(element.agent_future_np) < 1:
        agent_future_xyzh_world = np.full((fut_len, 4), np.nan)
        agent_traj_world = agent_past_xyzh_world
    else:
        agent_future_xyzh_world = transform_state_np_2d(element.agent_future_np, world_from_agent).as_format("x,y,z,h")
        agent_traj_world = np.vstack([agent_past_xyzh_world, agent_future_xyzh_world])

    if ref_polyline_ids:
        closest_ref, exceeded, additional_info = process_existing_lane(
            element, agent_traj_world, ref_polyline_ids, all_ref_polyline_ids, VEC_MAP_PARAMS
        )
        if exceeded:
            return create_empty_return_dict(VEC_MAP_PARAMS, exceed_lane=True)
        if closest_ref is None:
            return create_empty_return_dict(VEC_MAP_PARAMS)
    else:
        closest_ref, additional_info = find_reference_lane(element,
                                                           agent_traj_world,
                                                           all_ref_polyline_ids,
                                                           VEC_MAP_PARAMS)
        exceeded = False
        if closest_ref is None:
            return create_empty_return_dict(VEC_MAP_PARAMS)

    # Extract centerline info
    centerline_world_xy = closest_ref.centerline_xy
    centerline_heading = closest_ref.centerline_h
    
    #need to chop and fix the length to 150
    chopped_center_line_world_xy, chopped_centerline_heading = chop_lane_for_trajectory(
        centerline_world_xy, 
        agent_traj_world[:element.agent_history_len,:2],
        meters_before=VEC_MAP_PARAMS["CENTERLINE_LOOKBACK"],
        meters_after=VEC_MAP_PARAMS["CENTERLINE_LOOKAHEAD"],
        num_pts=VEC_MAP_PARAMS["CENTERLINE_LENGTH"]
    )
    centerline_xy = transform_coords_np(chopped_center_line_world_xy, element.agent_from_world_tf)
    centerline_heading = chopped_centerline_heading


    # Build return dictionary
    return {
        'centerline_world_xy': chopped_center_line_world_xy,
        'centerline_xy': centerline_xy,
        'centerline_heading': centerline_heading,
        'init_centerline_heading': centerline_heading[VEC_MAP_PARAMS["CENTERLINE_LOOKBACK"]:VEC_MAP_PARAMS["CENTERLINE_LOOKBACK"]+5],
        'ref_polyline_ids': json.dumps(closest_ref.polyline_ids),
        'has_lane': True,
        'all_poss_refs': additional_info['all_poss_refs'],
        'all_poss_refs_global': additional_info['all_poss_refs_global'],
        'num_poss_refs': additional_info['num_poss_refs'],
        'all_ref_polyline_ids': additional_info['all_ref_polyline_ids'],
        'exceed_lane': exceeded
    }

def find_closest_lanes(vector_map: VectorMap,
                       agent_cur_xyzh: np.ndarray,
                       dist_threshold: float = 5.0,
                       max_threshold: float = 10.0,
                       max_heading_error: float = np.pi / 4.0
                       ) -> List[RoadLane]:
    """
    Returns a list of lanes close to agent_cur_xyzh, expanding search if none found.
    """
    possible = vector_map.get_current_lane(agent_cur_xyzh,
                                           max_dist=dist_threshold,
                                           max_heading_error=max_heading_error)
    while len(possible) < 1 and dist_threshold < max_threshold:
        dist_threshold *= 2.0
        possible = vector_map.get_current_lane(agent_cur_xyzh,
                                               max_dist=dist_threshold,
                                               max_heading_error=max_heading_error)
    return possible


def process_existing_lane(element: AgentBatchElement,
                          agent_traj_world: np.ndarray,
                          ref_polyline_ids: str,
                          all_ref_polyline_ids: str = "",
                          vec_map_params: Dict = None
                          ) -> Tuple[Optional[ref_path], bool, Dict]:
    """
    Processes an existing reference lane and optionally extends it. 
    Returns (closest_ref, exceeded_lane, additional_info).
    """
    vector_map: VectorMap = element.vec_map

    # Prepare arrays to store additional possible references
    all_poss_refs = np.full((vec_map_params["NUM_FUTURE_LANES"], 
                            vec_map_params["CENTERLINE_LENGTH"], 2), np.nan)
    all_poss_refs_global = np.full((vec_map_params["NUM_FUTURE_LANES"], 
                                   vec_map_params["CENTERLINE_LENGTH"], 2), np.nan)
    all_polyline_ids_list = None

    polyline_ids = json.loads(ref_polyline_ids)
    pred_lanes = [vector_map.get_road_lane(lid) for lid in polyline_ids]
    closest_ref: ref_path = ref_path(pred_lanes)

    # Check if the agent has exceeded this lane's end
    exceeded = closest_ref.exceed_lane(agent_traj_world[[element.agent_history_len - 1], :2])
    if exceeded:
        return None, True, {}

    # Extend the lane if needed
    if closest_ref.check_extend_future_lane(agent_traj_world[[element.agent_history_len - 1], :2], vec_map_params["EXTEND_DISTANCE"]):
        if len(list(closest_ref.last_lane.next_lanes)) > 0:
            next_lane_id = list(closest_ref.last_lane.next_lanes)[0]
            next_lane = vector_map.get_road_lane(next_lane_id)
            next_lane_list = dfs(next_lane, vector_map, 
                                dist=0, 
                                threshold=vec_map_params["DFS_THRESHOLD_FUTURE"])[0]
            closest_ref.extend_future_lane(next_lane_list)

    # Process all possible reference paths if provided
    if all_ref_polyline_ids not in ["None", ""]:
        all_polyline_ids_list = json.loads(all_ref_polyline_ids)
        assert isinstance(all_polyline_ids_list[0], list), "all_ref_polyline_ids must be a list of lists."
        assert len(all_polyline_ids_list) <= vec_map_params["NUM_FUTURE_LANES"], (
            f"Number of reference paths ({len(all_polyline_ids_list)}) "
            f"exceeds allocated space ({vec_map_params['NUM_FUTURE_LANES']})"
        )

        for idx, ref_ids in enumerate(all_polyline_ids_list):
            pred_lanes = [vector_map.get_road_lane(lid) for lid in ref_ids]
            ref_candidate = ref_path(pred_lanes)
            chopped, _ = chop_lane_for_trajectory(
                ref_candidate.centerline_xy,
                agent_traj_world[:element.agent_history_len, :2],
                meters_before=vec_map_params["CENTERLINE_LOOKBACK"],
                meters_after=vec_map_params["CENTERLINE_LOOKAHEAD"],
                num_pts=vec_map_params["CENTERLINE_LENGTH"]
            )
            agent_centric = transform_coords_np(chopped, element.agent_from_world_tf)
            all_poss_refs[idx] = agent_centric
            all_poss_refs_global[idx] = chopped

    additional_info = {
        "ref_polyline_ids": ref_polyline_ids,
        'all_poss_refs': all_poss_refs,
        'all_poss_refs_global': all_poss_refs_global,
        'num_poss_refs': len(all_polyline_ids_list) if all_polyline_ids_list else 1,
        'all_ref_polyline_ids': all_ref_polyline_ids,
        'all_polyline_ids_list': all_polyline_ids_list
    }
    return closest_ref, False, additional_info


def find_reference_lane(element: AgentBatchElement,
                        agent_traj_world: np.ndarray,                      
                        all_ref_polyline_ids: str = "",
                        vec_map_params: Dict = None
                        ) -> Tuple[Optional[ref_path], Dict]:
    """
    Finds a new reference lane if no existing reference is provided. 
    Returns (closest_ref, additional_info).
    """
    vector_map: VectorMap = element.vec_map
    idx_current = element.agent_history_len - 1
    agent_cur_xyzh = agent_traj_world[idx_current]

    # Prepare arrays for possible references
    all_poss_refs = np.full((vec_map_params["NUM_FUTURE_LANES"], 
                            vec_map_params["CENTERLINE_LENGTH"], 2), np.nan)
    all_poss_refs_global = np.full((vec_map_params["NUM_FUTURE_LANES"], 
                                   vec_map_params["CENTERLINE_LENGTH"], 2), np.nan)
    all_polyline_ids_list = None

    # If all_ref_polyline_ids is provided, load them
    if vec_map_params["FIND_ALL_POS_REFS"] and all_ref_polyline_ids not in ["None", ""]:
        all_polyline_ids_list = json.loads(all_ref_polyline_ids)
        assert isinstance(all_polyline_ids_list[0], list)
        assert len(all_polyline_ids_list) <= vec_map_params["NUM_FUTURE_LANES"]

        for idx, ref_ids in enumerate(all_polyline_ids_list):
            pred_lanes = [vector_map.get_road_lane(lid) for lid in ref_ids]
            ref_candidate = ref_path(pred_lanes)
            chopped, _ = chop_lane_for_trajectory(
                ref_candidate.centerline_xy,
                agent_traj_world[:element.agent_history_len, :2],
                meters_before=vec_map_params["CENTERLINE_LOOKBACK"],
                meters_after=vec_map_params["CENTERLINE_LOOKAHEAD"],
                num_pts=vec_map_params["CENTERLINE_LENGTH"]
            )
            agent_centric = transform_coords_np(chopped, element.agent_from_world_tf)
            all_poss_refs[idx] = agent_centric
            all_poss_refs_global[idx] = chopped

        return None, {
            'all_poss_refs': all_poss_refs,
            'all_poss_refs_global': all_poss_refs_global,
            'num_poss_refs': len(all_polyline_ids_list),
            'all_ref_polyline_ids': all_ref_polyline_ids,
            'all_polyline_ids_list': all_polyline_ids_list
        }

    if "nuplan" in element.map_name:
        agent_cur_xyzh[2] = 0.0

    # Find lanes near the agent
    base_lanes = find_closest_lanes(vector_map, agent_cur_xyzh,
                                    dist_threshold=vec_map_params["FIND_CLOSEST_DIST_THRESHOLD"],
                                    max_threshold=vec_map_params["FIND_CLOSEST_MAX_THRESHOLD"])
    if vec_map_params["FIND_ALL_POS_REFS"]:
        extra_lanes = find_closest_lanes(vector_map, agent_cur_xyzh,
                                         dist_threshold=vec_map_params["FIND_ALL_DIST_THRESHOLD"],
                                         max_threshold=vec_map_params["FIND_ALL_MAX_THRESHOLD"],
                                         max_heading_error=vec_map_params["DEFAULT_MAX_HEADING_ERROR"])
        base_lanes = base_lanes + sorted(extra_lanes, key=lambda x: x.id)

    if not base_lanes:
        return None, {
            'all_poss_refs': all_poss_refs,
            'all_poss_refs_global': all_poss_refs_global,
            'num_poss_refs': 0,
            'all_ref_polyline_ids': "None",
            'all_polyline_ids_list': None
        }

    obs_pred_lanes = []
    for lane in base_lanes:
        candidates_future = dfs(lane, vector_map, dist=0, threshold=150)
        candidates_past = dfs(lane, vector_map, dist=0, threshold=50, extend_along_predecessor=True)
        for past_lane_seq in candidates_past:
            for future_lane_seq in candidates_future:
                if past_lane_seq[-1].id == future_lane_seq[0].id:
                    obs_pred_lanes.append(past_lane_seq + future_lane_seq[1:])

    if not obs_pred_lanes:
        return None, {
            'all_poss_refs': all_poss_refs,
            'all_poss_refs_global': all_poss_refs_global,
            'num_poss_refs': 0,
            'all_ref_polyline_ids': "None",
            'all_polyline_ids_list': None
        }

    obs_pred_lanes = remove_overlapping_lane_seq_idx(obs_pred_lanes)
    ref_list = [ref_path(seq) for seq in obs_pred_lanes]

    if len(ref_list) > 1 and len(agent_traj_world) > 1:
        aligned_refs = get_traj_closest_refs_with_similarity(
            agent_traj_world[:, :2],
            agent_traj_world[:, -1:],
            ref_list,
            element.agent_meta_dict["is_stationary"]
        )
    else:
        aligned_refs = ref_list
    assert aligned_refs, "No valid reference lanes found."
    closest_ref = aligned_refs[0]

    # Fill out all possible references
    all_polyline_ids_list = []
    for i, ref_candidate in enumerate(ref_list[:min(vec_map_params["NUM_FUTURE_LANES"], len(ref_list))]):
        chopped, _ = chop_lane_for_trajectory(
            ref_candidate.centerline_xy,
            agent_traj_world[:element.agent_history_len, :2],
            meters_before=vec_map_params["CENTERLINE_LOOKBACK"],
            meters_after=vec_map_params["CENTERLINE_LOOKAHEAD"],
            num_pts=vec_map_params["CENTERLINE_LENGTH"]
        )
        agent_centric = transform_coords_np(chopped, element.agent_from_world_tf)
        all_poss_refs[i] = agent_centric
        all_poss_refs_global[i] = chopped
        all_polyline_ids_list.append(ref_candidate.polyline_ids)

    return closest_ref, {
        'all_poss_refs': all_poss_refs,
        'all_poss_refs_global': all_poss_refs_global,
        'num_poss_refs': len(all_polyline_ids_list),
        'all_ref_polyline_ids': json.dumps(all_polyline_ids_list),
        'all_polyline_ids_list': all_polyline_ids_list
    }


def create_np_arr_pad(np_arr: np.ndarray, padding_shape: Tuple[int, int] = None) -> np.ndarray:
    """Pad np_arr to the specified shape with zeros."""
    if padding_shape is None:
        padding_shape = VEC_MAP_PARAMS["NP_ARR_PAD_SHAPE"]
    np_arr_pad = np.full(padding_shape, 0.0)
    np_arr_pad[:len(np_arr)] = np_arr
    return np_arr_pad


def assert_centerline_ahead(agent_traj_world: np.ndarray,
                            centerline_xy: np.ndarray) -> None:
    """
    Asserts that at the current time, the centerline is ahead of the agent.
    """
    agent_current_pos = agent_traj_world[0, :2]
    distances = np.linalg.norm(centerline_xy - agent_current_pos, axis=1)
    closest_idx = np.argmin(distances)
    assert closest_idx > 0, "The centerline is not ahead of the agent."


def dfs(
    lane: Union[str, RoadLane],
    vector_map: VectorMap,
    dist: float = 0,
    threshold: float = 30,
    extend_along_predecessor: bool = False
) -> List[List[str]]:
    """
    Depth-first search over lane connections, with a distance threshold.
    """
    cache = {}

    def _dfs(curr_lane, curr_dist):
        if curr_dist > threshold:
            return [[curr_lane]]
        if curr_lane in cache:
            return cache[curr_lane]

        traversed = []
        try:
            child_lanes = (curr_lane.next_lanes
                           if not extend_along_predecessor
                           else curr_lane.prev_lanes)
        except:
            child_lanes = []

        for child in child_lanes:
            if isinstance(child, str):
                child = vector_map.get_road_lane(child)
            cl_length = child.center.total_distance
            sub_seqs = _dfs(child, curr_dist + cl_length)
            for seq in sub_seqs:
                if extend_along_predecessor:
                    traversed.append(seq + [curr_lane])
                else:
                    traversed.append([curr_lane] + seq)

        cache[curr_lane] = traversed if traversed else [[curr_lane]]
        return cache[curr_lane]

    if isinstance(lane, str):
        lane = vector_map.get_road_lane(lane)
    return _dfs(lane, dist)




def create_empty_return_dict(vec_map_params: Dict = None,exceed_lane: bool = False) -> Dict[str, Any]:
    """
    Return a default dictionary for cases when lane info cannot be computed.
    """
    return {
        'all_poss_refs': np.full((vec_map_params["NUM_FUTURE_LANES"], vec_map_params["CENTERLINE_LENGTH"], 2), 0),
        'all_poss_refs_global': np.full((vec_map_params["NUM_FUTURE_LANES"], vec_map_params["CENTERLINE_LENGTH"], 2), 0),
        "num_poss_refs": 0,
        'centerline_world_xy': np.full((vec_map_params["CENTERLINE_LENGTH"], 2), 0),
        'centerline_xy': np.full((vec_map_params["CENTERLINE_LENGTH"], 2), 0),
        'centerline_heading': np.full((vec_map_params["CENTERLINE_LENGTH"],), 0),
        'init_centerline_heading': np.full((5,), 0),
        'ref_polyline_ids': "None",
        'all_ref_polyline_ids': "None",
        'has_lane': False,
        "exceed_lane": exceed_lane
    }