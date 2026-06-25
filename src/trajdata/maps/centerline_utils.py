import numpy as np
from shapely.geometry import LinearRing, LineString, Point
from typing import List, Sequence, Set, Tuple
from trajdata.maps.ref_utils import ref_path


def _safe_linestring(centerline: np.ndarray) -> LineString:
    """Build a shapely LineString robust to float32 precision loss.

    On large-magnitude world coords (e.g. NuPlan UTM ~1e6), float32 precision
    is ~0.25 m which can collapse consecutive polyline points into identical
    values, yielding zero-length segments. Subsequent ``.project()`` /
    ``.distance()`` calls then return NaN and GEOS raises
    ``RuntimeWarning: invalid value encountered``. We upcast to float64 and
    drop consecutive duplicates. For NuScenes (world coords ~1e3) this is a
    no-op.
    """
    pts = np.asarray(centerline, dtype=np.float64)
    if pts.ndim == 2 and pts.shape[0] >= 2:
        keep = np.concatenate(([True], np.any(np.diff(pts, axis=0) != 0, axis=1)))
        pts = pts[keep]
    if pts.shape[0] < 2:
        pts = np.vstack([pts, pts[-1:] + 1e-6])
    return LineString(pts)

def is_overlapping_lane_seq(lane_seq1: Sequence[str], lane_seq2: Sequence[str]) -> bool:
    """
    Check if the 2 lane sequences are overlapping.
    """
    if lane_seq2[0] in lane_seq1[1:] and lane_seq1[-1] in lane_seq2[:-1]:
        return True
    elif set(lane_seq2) <= set(lane_seq1):
        return True
    return False

def find_redundant_indices(lane_seqs: Sequence[Sequence[str]]) -> Set[int]:
    """
    Find redundant lane sequence indices based on overlapping criteria.
    """
    redundant_lane_idx = set()
    lane_seq_sets = [set(seq) for seq in lane_seqs]
    
    for i, lane_set_i in enumerate(lane_seq_sets):
        for j, lane_set_j in enumerate(lane_seq_sets):
            if i >= j or j in redundant_lane_idx:
                continue
            
            if lane_set_i.intersection(lane_set_j):
                redundant_lane_idx.add(j)
                
    return redundant_lane_idx

def remove_overlapping_lane_seq_idx(lane_seqs: Sequence[Sequence[str]]) -> List[Sequence[str]]:
    """
    Remove lane sequences which are overlapping to some extent.
    """
    redundant_lane_idx = find_redundant_indices(lane_seqs)
    unique_lane_seqs = [seq for i, seq in enumerate(lane_seqs) if i not in redundant_lane_idx]
    return unique_lane_seqs

def get_normal_and_tangential_distance_point(
    x, y, centerline, start_idx, window_size, delta=0.01, last=False
):
    """
    Calculate normal and tangential distance for a single point.
    """
    point = Point(x, y)
    centerline_ls = _safe_linestring(centerline)

    end_idx = min(start_idx + window_size, len(centerline) - 1)
    if end_idx - start_idx > 0:
        limited_centerline = _safe_linestring(centerline[start_idx:end_idx+1])
    else:
        if end_idx == len(centerline) - 1:
            limited_centerline = _safe_linestring(centerline[-2:])
    
    tang_dist_limited = limited_centerline.project(point)
    norm_dist_limited = point.distance(limited_centerline)
    point_on_limited_cl = limited_centerline.interpolate(tang_dist_limited)
    
    distance_to_start_idx = centerline_ls.project(Point(centerline[start_idx]))
    tang_dist = distance_to_start_idx + tang_dist_limited
    
    new_idx = np.argmin(np.linalg.norm(centerline - np.array(point_on_limited_cl.coords[0]), axis=1))
    
    point_on_cl = centerline_ls.interpolate(tang_dist)
    if not last:
        pt1 = point_on_cl.coords[0]
        pt2 = centerline_ls.interpolate(tang_dist + delta).coords[0]
        pt3 = point.coords[0]
    else:
        pt1 = centerline_ls.interpolate(tang_dist - delta).coords[0]
        pt2 = point_on_cl.coords[0]
        pt3 = point.coords[0]
   
    lr_coords = [pt1, pt2, pt3]
    lr = LinearRing(lr_coords)
    
    if lr.is_ccw:
        return (tang_dist, norm_dist_limited, new_idx)
    return (tang_dist, -norm_dist_limited, new_idx)

def get_nt_distance(xy: np.ndarray, centerline: np.ndarray, window_size=50) -> np.ndarray:
    """
    Get normal and tangential distances for the given xy trajectory.
    """
    traj_len = xy.shape[0]
    nt_distance = np.zeros((traj_len, 2))
    max_dist: float = -1
    
    start_idx = np.argmin(np.linalg.norm(centerline - xy[0], axis=1))

    for i in range(traj_len):
        tang_dist, norm_dist, new_idx = get_normal_and_tangential_distance_point(
            xy[i][0], xy[i][1], centerline, start_idx, window_size, last=False)

        if tang_dist > max_dist:
            max_dist = tang_dist
            last_x = xy[i][0]
            last_y = xy[i][1]
            last_idx = i
        nt_distance[i, 0] = norm_dist
        nt_distance[i, 1] = tang_dist
        start_idx = new_idx

    tang_dist, norm_dist, _ = get_normal_and_tangential_distance_point(
        last_x, last_y, centerline, start_idx, window_size, last=True)
    nt_distance[last_idx, 0] = norm_dist

    return nt_distance

def chop_lane_for_trajectory(centerline: np.ndarray, xy: np.ndarray, meters_before: float, meters_after: float, num_pts: int = 150) -> Tuple[np.ndarray, np.ndarray]:
    """
    Chop lanes before and after the starting point of trajectory.
    """
    centerline_ls = _safe_linestring(centerline)
    point = Point(float(xy[-1, 0]), float(xy[-1, 1]))
    tang_dist = centerline_ls.project(point)

    start_pt = max(tang_dist - meters_before, 0)
    end_pt = min(tang_dist + meters_after, centerline_ls.length)

    dist_to_interp = np.linspace(start_pt, end_pt, num=num_pts).tolist()
    new_centerline = []
    for dist in dist_to_interp:
        interPts = centerline_ls.interpolate(dist)
        new_centerline.append(list(interPts.coords)[0])
    new_centerline = np.array(new_centerline)

    delta_x = np.diff(new_centerline[:, 0])
    delta_y = np.diff(new_centerline[:, 1])
    heading = np.arctan2(delta_y, delta_x)
    heading = np.append(heading, heading[-1])
    
    return new_centerline, heading

def calculate_heading_alignment(traj_xy: np.ndarray, traj_heading: np.ndarray, ref: ref_path) -> Tuple[bool, float]:
    """
    Calculate heading alignment between trajectory and reference path.
    """
    from scipy.spatial.distance import cdist

    dist_matrix = cdist(traj_xy, ref.centerline_xy, metric='euclidean')
    matched_indices = np.argmin(dist_matrix, axis=1)
    matched_ref_heading = ref.centerline_h[matched_indices]

    heading_diff = traj_heading - matched_ref_heading
    heading_alignment = np.abs(heading_diff)
    greater = np.any(heading_alignment > np.pi*60/180)

    return greater, heading_alignment[-1]

def calculate_similarity_score_hierarchical(traj_xy: np.ndarray, traj_h: np.ndarray, ref_path: ref_path) -> Tuple[float, float, bool]:
    """
    Hierarchically calculate similarity score between trajectory and reference path.
    """
    nt_distance = get_nt_distance(traj_xy, ref_path.centerline_xy)
    similarity_score = -np.sum(np.abs(nt_distance[:, 0]))
    
    bad_heading_align, heading_alignment = calculate_heading_alignment(traj_xy, traj_h, ref_path)
    
    return similarity_score, heading_alignment, bad_heading_align

def get_traj_closest_refs_with_similarity(traj_xy: np.ndarray, traj_h: np.ndarray, ref_path_list: List[ref_path], stationary: bool) -> List[ref_path]:
    """
    Get reference paths most aligned with the trajectory.
    """
    assert len(ref_path_list) > 0
    best_match = {
        'similarity_score': -float("inf"),
        'heading_alignment': float("inf"),
        'ref_path': None
    }

    for ref_path in ref_path_list:
        similarity_score, heading_alignment, bad_heading_align = calculate_similarity_score_hierarchical(traj_xy, traj_h, ref_path)
        
        if bad_heading_align and not stationary:
            similarity_score = -float("inf")
        
        if similarity_score > -5:
            if heading_alignment < best_match['heading_alignment']:
                best_match['similarity_score'] = similarity_score
                best_match['heading_alignment'] = heading_alignment
                best_match['ref_path'] = ref_path
        else:
            if similarity_score > best_match['similarity_score']:
                best_match['similarity_score'] = similarity_score
                best_match['heading_alignment'] = heading_alignment
                best_match['ref_path'] = ref_path

    if best_match['ref_path'] is None or best_match['similarity_score'] == -float("inf"):
        return [ref_path_list[0]]
    return [best_match['ref_path']]