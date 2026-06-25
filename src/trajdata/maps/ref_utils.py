from dataclasses import dataclass, field
from trajdata.maps.vec_map_elements import RoadLane
from typing import List

import numpy as np
import matplotlib.pyplot as plt
@dataclass
class ref_path:
    ''' reference path of frenet-serret coordinates
        The ref_path consists: 
            1) sampled points 
            2) analytic information 
            3) List of RoadLane IDs that constitues the reference
            4) List of Possible future reference 
        ------------------------
        attributes:
            id: List of RoadLaneids
            ---------sample points--------
            centerline_xy: xy np.narray of the reference path
            s: corresponding s information 
    '''
    road_lane_list: List[RoadLane] 
    polyline_ids: List[str] = field(default_factory=list)
    
    def __post_init__(self) -> None:
        # This function will be called multiple times of the same id
        self.polyline_ids = [lane.id for lane in self.road_lane_list]
        
        centerline_xyzh = []
        ref_s_boundaries = [0]
        s0 = 0
     
        for lane in self.road_lane_list:
            s0 += lane.center.total_distance
            centerline_xyzh.append(lane.center.xyzh)
            ref_s_boundaries.append(s0)
       

        self.ref_s_boundaries = ref_s_boundaries
        self.centerline_xyzh_list = centerline_xyzh
        centerline_xyzh    = np.vstack(centerline_xyzh)
   
        self.centerline_xy = centerline_xyzh[:,:2]
        self.centerline_h  = centerline_xyzh[:,-1:]
        self.s             = self.calculate_longitudinal_distance(self.centerline_xy)
        self.ref_s_boundaries = ref_s_boundaries
        self.last_lane = lane
       
   
    @staticmethod
    def calculate_longitudinal_distance(xy_coordinates):
        distances = np.linalg.norm(np.diff(xy_coordinates, axis=0), axis=1)
        s = np.insert(np.cumsum(distances), 0, 0.0)
        return s
    
    def get_current_lane(self, xy_point: np.ndarray)->RoadLane:
        """
        Given an XY point, this function finds the current lane in the road_lane_list by:
        1) Calculating its traveled length on the lane
        2) Using s.boundaries to determine the current lane
        
        Args:
            xy_point (np.ndarray): A point in XY coordinates.
        
        Returns:
            current_lane (RoadLane): The current lane containing the xy_point.
        """

        # Calculate the distance along each lane
        from trajdata.maps.centerline_utils import get_nt_distance
        
        nt_arr = get_nt_distance(xy_point[-1:,:],self.centerline_xy)

        # Find the index of the lane containing the xy_point
        current_lane_idx = np.digitize(nt_arr[-1:,1], self.ref_s_boundaries).item() - 1

        if current_lane_idx >= len(self.ref_s_boundaries) - 1:
            current_lane_idx = len(self.ref_s_boundaries) - 2
            print("Warning: current_lane_idx is greater than len(). It has been set to len-1.")

        # Check if current_lane_idx is out of bounds on the low end
        if current_lane_idx < 0:
            current_lane_idx = 0
            print("Warning: current_lane_idx is less than 0. It has been set to 0.")
        # Return the current lane
        current_lane = self.road_lane_list[current_lane_idx]

        return current_lane 
    
    def check_extend_future_lane(self, point, threshold=100) -> bool:
        '''Extend the future reference based on current point (for simulation)
        '''
        last_lane_point = self.last_lane.center.xy[-1]

        return np.linalg.norm(point-last_lane_point)<threshold
    
    def exceed_lane(self, point, threshold=10) -> bool:
        '''Check if the future reference should be immediately extended based on the proximity of the current point to the end of the lane
        '''
        last_lane_point = self.last_lane.center.xy[-1]
        
        return np.linalg.norm(point-last_lane_point) < threshold

    def extend_future_lane(self, next_lane_list):
        '''Extend the future reference based on current point (for simulation)
        '''
        self.polyline_ids.extend([lane.id for lane in next_lane_list])
        self.road_lane_list.extend(next_lane_list)
        s0 = self.ref_s_boundaries[-1]
        for lane  in next_lane_list:
            s0 += lane.center.total_distance
            self.centerline_xyzh_list.append(lane.center.xyzh)
            self.ref_s_boundaries.append(s0)
        
        centerline_xyzh    = np.vstack(self.centerline_xyzh_list)

        self.centerline_xy = centerline_xyzh[:,:2]
        self.centerline_h  = centerline_xyzh[:,-1:]
        self.s             = self.calculate_longitudinal_distance(self.centerline_xy)
        self.last_lane = lane

