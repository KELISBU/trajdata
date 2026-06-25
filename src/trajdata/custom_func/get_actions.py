from trajdata.data_structures.batch_element import AgentBatchElement
from trajdata.data_structures import  StateTensor,StateArray
import numpy as np

def unwrap_angle(angle_diff):
    return (angle_diff + np.pi) % (2 * np.pi) - np.pi
def smooth_large_jumps(data, threshold=np.pi/2.0):
    """
    Smooths out sudden jumps in data by setting them to a nearby value or the mean of the neighbors.
    
    Parameters:
    - data (np.ndarray): 1D array containing the data points.
    - threshold (float): The threshold for a jump to be considered "large".

    Returns:
    - np.ndarray: Smoothed data.
    """
    # Calculate the first-order difference
    diffs = np.diff(data)
    
    # Identify large jumps
    large_jump_indices = np.where(np.abs(diffs) > threshold)[0]
    
    # If the start or end has a large jump, set it to its only neighbor
    if 0 in large_jump_indices:
        data[0] = data[1]
    if len(data) - 2 in large_jump_indices:
        data[-1] = data[-2]
    
    # For other large jumps, use the mean of the neighbors
    large_jump_indices = large_jump_indices[(large_jump_indices > 0) & (large_jump_indices < len(data) - 2)]
    data[large_jump_indices + 1] = (data[large_jump_indices] + data[large_jump_indices + 2]) / 2.0
    
    return data

def get_actions_inverse_dynamics(element: AgentBatchElement, maxturn_rate=np.pi):
    ''' get target actions using inverse dynamics
    '''
    pos_hist, yaw_hist, speed_hist = trajdata2posyawspeed(element.agent_history_np)
    pos, yaw, speed = trajdata2posyawspeed(element.agent_future_np)

    pos = np.vstack([pos_hist[-1:],pos])
    yaw = np.vstack([yaw_hist[-1:], yaw])
    # yaw = np.unwrap(yaw)  # Unwrap yaw values
    # Unwrap yaw values for smooth transitions
    speed = np.vstack([speed_hist[-1:],speed])
    turning_rate,yaw = adjust_yaw_and_calculate_turning_rate(speed,yaw)
    target_states = np.hstack([pos,speed,yaw])# x,y,v,theta

    # target_states  = torch.cat([batch["target_positions"],
    #                             batch["target_speeds"][...,None],batch["target_yaws"]],dim=-1)
    # target_states  = torch.cat([curr_states[:,None],target_states],dim=1) #(B,horizon+1,4)
#   
    target_actions = inverse_dyn(target_states[...,:-1,:],target_states[...,1:,:],element.dt) # a,w
    # Check if any turn rate exceeds the maximum, and if so, visualize it.
    # assert (np.abs(target_actions[:, 1]) <= maxturn_rate).all(), \
    #     f"Turning rate should be within [-{maxturn_rate}, {maxturn_rate}]. Found out-of-bounds values: { target_actions[:, 1][np.abs(target_actions[:, 1]) > maxturn_rate]}"
    target_actions[:,1] = turning_rate / element.dt
    # if not (np.abs(target_actions[:, 1]) <= maxturn_rate).all():
    #     print("Turning rate exceeded maximum allowable value. Visualizing...")
    #     visualize_exceeding_turnrate(target_actions, target_states, maxturn_rate)
    return target_actions

def adjust_yaw_and_calculate_turning_rate(speed_array, yaw_array, speed_threshold=0.3):
    """
    Adjusts yaw and calculates turning rate based on speed and yaw.

    Parameters:
    - speed_array (np.ndarray): 1D array containing the speed data points.
    - yaw_array (np.ndarray): 1D array containing the yaw data points.
    - speed_threshold (float): The speed below which yaw is set to zero.

    Returns:
    - np.ndarray: Turning rate after adjustments.
    - np.ndarray: Calibrated yaw after adjustments.
    """

    # Set yaw to zero when speed is low
    mask_low_speed = (speed_array < speed_threshold)
    yaw_array[mask_low_speed] = 0.0

    # Calculate yaw based on cos and sin
    cos_yaw = np.cos(yaw_array)
    sin_yaw = np.sin(yaw_array)
    
    # Calculate calibrated yaw
    calibrated_yaw = np.arctan2(sin_yaw, cos_yaw)

    # Calculate derivatives of cos and sin yaw
    dcos_dt = np.diff(cos_yaw, axis=0)
    dsin_dt = np.diff(sin_yaw,  axis=0)

    # Calculate turning rate
    turning_rate = calculate_turning_rate_from_cos_sin_derivatives(dcos_dt, dsin_dt, cos_yaw, sin_yaw)

    return turning_rate.squeeze(), calibrated_yaw
def trajdata2posyawspeed(state, nan_to_zero=True):
    """Converts trajdata's state format to pos, yaw, and speed. Set Nans to 0s"""
    # if isinstance(state,StateTensor):
    #     pos = state.position.as_tensor()
    #     yaw = state.heading
    #     speed = state.as_format("v_lon")[...,0].as_tensor()
    # elif isinstance(state,StateArray):
    pos = state.position
    yaw = state.heading
    speed = state.as_format("v_lon")
   
    return pos, yaw, speed
def inverse_dyn(x,xp,dt):
    return (xp[...,2:]-x[...,2:])/dt

def calculate_turning_rate_from_cos_sin_derivatives(dcos_dt, dsin_dt, cos_yaw, sin_yaw):
    """
    Calculates turning rate from derivatives of cos and sin of yaw.

    Parameters:
    - dcos_dt (np.ndarray): Derivative of cos of yaw.
    - dsin_dt (np.ndarray): Derivative of sin of yaw.
    - cos_yaw (np.ndarray): Cos of yaw.
    - sin_yaw (np.ndarray): Sin of yaw.

    Returns:
    - np.ndarray: Turning rate.
    """
    # Align the shapes
    cos_yaw = cos_yaw[:-1]
    sin_yaw = sin_yaw[:-1]

    # Calculate turning rate
    numerator = -dcos_dt * sin_yaw + dsin_dt * cos_yaw
    denominator = cos_yaw ** 2 + sin_yaw ** 2
    turning_rate = numerator / denominator
    
    return turning_rate

import matplotlib.pyplot as plt
import numpy as np

def visualize_exceeding_turnrate(target_actions, target_states, maxturn_rate):
    # Identify where the turning rate exceeds the maximum
    exceed_indices = np.where(np.abs(target_actions[:, 1]) > maxturn_rate)[0]
    
    # Set up plots
    fig, axs = plt.subplots(5, 1, figsize=(12, 16))

    # Plot Positions
    axs[0].plot(target_states[:, 0], target_states[:, 1], label='Position (x, y)')
    axs[0].scatter(target_states[exceed_indices, 0], target_states[exceed_indices, 1], color='red')
    axs[0].set_title('Position')
    axs[0].set_xlabel('x')
    axs[0].set_ylabel('y')

    # Plot Yaw
    axs[1].plot(target_states[:, 3], label='Yaw')
    axs[1].scatter(exceed_indices, target_states[exceed_indices, 3], color='red')
    axs[1].set_title('Yaw')
    axs[1].set_xlabel('Time step')
    axs[1].set_ylabel('Yaw (radians)')

    # Plot Speed
    axs[2].plot(target_states[:, 2], label='Speed')
    axs[2].scatter(exceed_indices, target_states[exceed_indices, 2], color='red')
    axs[2].set_title('Speed')
    axs[2].set_xlabel('Time step')
    axs[2].set_ylabel('Speed (m/s)')

    # Plot Acceleration
    axs[3].plot(target_actions[:, 0], label='Acceleration')
    axs[3].scatter(exceed_indices, target_actions[exceed_indices, 0], color='red')
    axs[3].set_title('Acceleration')
    axs[3].set_xlabel('Time step')
    axs[3].set_ylabel('Acceleration (m/s^2)')

    # Plot Turning Rate
    axs[4].plot(target_actions[:, 1], label='Turning Rate')
    axs[4].scatter(exceed_indices, target_actions[exceed_indices, 1], color='red')
    axs[4].set_title('Turning Rate')
    axs[4].set_xlabel('Time step')
    axs[4].set_ylabel('Turning Rate (rad/s)')

    plt.tight_layout()
    plt.savefig(f"turningrate_exceeded{np.abs(target_actions[:, 1]).max()}.png")
