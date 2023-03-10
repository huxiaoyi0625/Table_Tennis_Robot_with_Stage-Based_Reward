# You can change the data_path here to read different trajectories
data_path = './Trajectory_Experiments/Serve_plus_Spin'
import os
import csv
from matplotlib import pyplot as plt
import numpy as np


def _remove_outliners(data_array, data_windows, std_threshold):
    data_windows = min(len(data_array), data_windows)  # the length of the sliding window
    # std_threshold: the threshold to choose 2σ or 3σ

    for index in range(data_windows // 2, len(data_array) - data_windows // 2):
        xyz_array = data_array[index - data_windows // 2:index + data_windows // 2 + 1, :3].copy()
        bool_matrix_upper_bound = data_array[index, :3] > xyz_array.mean(axis=0) + std_threshold * xyz_array.std(axis=0)
        bool_matrix_lower_bound = data_array[index, :3] < xyz_array.mean(axis=0) - std_threshold * xyz_array.std(axis=0)
        bool_matrix = bool_matrix_upper_bound + bool_matrix_lower_bound
        data_array[index, :3] = np.where(bool_matrix,
                                         (xyz_array.sum(axis=0) - data_array[index, :3]) / (data_windows - 1),
                                         data_array[index, :3])
    return data_array

def _velocity_direct_estimmation(data_array):
    velocity_array = np.zeros_like(data_array[:, :3])
    for i in range(len(data_array) - 1):
        data_array_diff = data_array[i + 1] - data_array[i]
        velocity_array[i] = data_array_diff[:3] / data_array_diff[3]
    velocity_array = np.concatenate((velocity_array[:-1], data_array[1:, -1].reshape(-1, 1)), axis=1)  # 添加时间轴，实际应用时可以不要

    return velocity_array


def _velocity_estimation(data_array, vel_windows=3, std_threshold=1):
    velocity_array = np.zeros_like(data_array[:, :3])
    for i in range(len(data_array) - vel_windows + 1):
        data_array_diff = data_array[i + vel_windows - 1] - data_array[i]
        velocity_array[i] = np.where(data_array_diff[:3], data_array_diff[:3] / (data_array_diff[3] + 1e-9),
                                     velocity_array[
                                         i])  # prevent there is no difference in data and result in nan in data array

    velocity_array = np.concatenate((velocity_array[:-vel_windows + 1],
                                     data_array[vel_windows // 2:len(data_array) - vel_windows // 2, -1].reshape(-1,
                                                                                                                 1)),
                                    axis=1)  # Add time, maybe not necessary in application

    return velocity_array


def _force_direct_estimmation(data_array):
    velocity_array = np.zeros_like(data_array[:, :3])
    for i in range(len(data_array) - 1):
        data_array_diff = data_array[i + 1] - data_array[i]
        velocity_array[i] = (data_array_diff[:3] / data_array_diff[3]) * 2.7

    velocity_array = np.concatenate((velocity_array[:-1], data_array[1:, -1].reshape(-1, 1)), axis=1) # Add time, maybe not necessary in application

    return velocity_array


def _force_estimation(data_array, force_windows=3, std_threshold=1):
    force_array = np.zeros_like(data_array[:, :3])
    for i in range(len(data_array) - force_windows + 1):
        data_array_diff = data_array[i + force_windows - 1] - data_array[i]
        force_array[i] = np.where(data_array_diff[:3], (data_array_diff[:3] / (data_array_diff[3] + 1e-9)) * 1000,
                                  force_array[i])

    force_array = np.concatenate((force_array[:-force_windows + 1],
                                  data_array[force_windows // 2:len(data_array) - force_windows // 2, -1].reshape(-1,
                                                                                                                  1)),
                                 axis=1)  # Add time, maybe not necessary in application

    return force_array


def smoothing_windows_filter(ball_pos_array, remove_data_windows, vel_windows, force_windows, std_threshold):
    # It was used to integrate functions like:"_remove_outliners,_velocity_estimation,_force_estimation"
    outliner_remover = lambda data: _remove_outliners(data, remove_data_windows, std_threshold)
    estimate_ball_pos = outliner_remover(ball_pos_array)
    estimate_ball_vel = outliner_remover(_velocity_estimation(estimate_ball_pos, vel_windows, std_threshold))
    estimate_ball_force = outliner_remover(_force_estimation(estimate_ball_vel, force_windows, std_threshold))
    return estimate_ball_pos, estimate_ball_vel, estimate_ball_force


def predict_ball_state(estimate_ball_pos, estimate_ball_vel, estimate_ball_force, prediction_time, prediction_windows):
    predict_ball_pos = estimate_ball_pos[:, :3] + estimate_ball_vel[:, :3] * (prediction_time - estimate_ball_vel[3])

    return predict_ball_pos


def simulated_ball_current_state(removed_data_array, removed_outline_velocity_array,
                                 removed_outline_acceleration_array):
    prediction_time = removed_data_array[len(removed_data_array) // 2, 3] + 5  # simulated prediction time
    closest_index = np.argmin(abs(removed_data_array[:, 3] - prediction_time))
    return removed_data_array[:closest_index], removed_outline_velocity_array, removed_outline_acceleration_array

def _show_position_data(data_array, figure_scatter):
    figure_xyz_sub = figure_scatter.add_subplot(341, projection='3d')
    figure_xt_sub = figure_scatter.add_subplot(342)
    figure_yt_sub = figure_scatter.add_subplot(343)
    figure_zt_sub = figure_scatter.add_subplot(344)
    figure_xyz_sub.scatter(data_array[:, 0], data_array[:, 1], data_array[:, 2])
    figure_xt_sub.scatter(data_array[:, -1], data_array[:, 0])
    figure_yt_sub.scatter(data_array[:, -1], data_array[:, 1])
    figure_zt_sub.scatter(data_array[:, -1], data_array[:, 2])
    figure_xyz_sub.set_xlabel('x')
    figure_xyz_sub.set_ylabel('y')
    figure_xyz_sub.set_zlabel('z')
    figure_xyz_sub.set_title('xyz Scatter')
    figure_xt_sub.set_xlabel('t')
    figure_xt_sub.set_ylabel('x')
    figure_xt_sub.set_title('x-t Scatter')
    figure_yt_sub.set_xlabel('t')
    figure_yt_sub.set_ylabel('y')
    figure_yt_sub.set_title('y-t Scatter')
    figure_zt_sub.set_xlabel('t')
    figure_zt_sub.set_ylabel('z')
    figure_zt_sub.set_title('z-t Scatter')


def _show_velocity_data(data_array, figure_scatter):
    figure_xyz_sub = figure_scatter.add_subplot(345, projection='3d')
    figure_xt_sub = figure_scatter.add_subplot(346)
    figure_yt_sub = figure_scatter.add_subplot(347)
    figure_zt_sub = figure_scatter.add_subplot(348)
    figure_xyz_sub.scatter(data_array[:, 0], data_array[:, 1], data_array[:, 2])
    figure_xt_sub.scatter(data_array[:, -1], data_array[:, 0])
    figure_yt_sub.scatter(data_array[:, -1], data_array[:, 1])
    figure_zt_sub.scatter(data_array[:, -1], data_array[:, 2])
    figure_xyz_sub.set_xlabel('x')
    figure_xyz_sub.set_ylabel('y')
    figure_xyz_sub.set_zlabel('z')
    figure_xyz_sub.set_title('vel xyz Scatter')
    figure_xt_sub.set_xlabel('t')
    figure_xt_sub.set_ylabel('x')
    figure_xt_sub.set_title('vel x-t Scatter')
    figure_yt_sub.set_xlabel('t')
    figure_yt_sub.set_ylabel('y')
    figure_yt_sub.set_title('vel y-t Scatter')
    figure_zt_sub.set_xlabel('t')
    figure_zt_sub.set_ylabel('z')
    figure_zt_sub.set_title('vel z-t Scatter')


def _show_acceleration_data(data_array, figure_scatter):
    figure_xyz_sub = figure_scatter.add_subplot(3, 4, 9, projection='3d')
    figure_xt_sub = figure_scatter.add_subplot(3, 4, 10)
    figure_yt_sub = figure_scatter.add_subplot(3, 4, 11)
    figure_zt_sub = figure_scatter.add_subplot(3, 4, 12)
    figure_xyz_sub.scatter(data_array[:, 0], data_array[:, 1], data_array[:, 2])
    figure_xt_sub.scatter(data_array[:, -1], data_array[:, 0])
    figure_yt_sub.scatter(data_array[:, -1], data_array[:, 1])
    figure_zt_sub.scatter(data_array[:, -1], data_array[:, 2])
    figure_xyz_sub.set_xlabel('x')
    figure_xyz_sub.set_ylabel('y')
    figure_xyz_sub.set_zlabel('z')
    figure_xyz_sub.set_title('acceleration xyz Scatter')
    figure_xt_sub.set_xlabel('t')
    figure_xt_sub.set_ylabel('x')
    figure_xt_sub.set_title('acceleration x-t Scatter')
    figure_yt_sub.set_xlabel('t')
    figure_yt_sub.set_ylabel('y')
    figure_yt_sub.set_title('acceleration y-t Scatter')
    figure_zt_sub.set_xlabel('t')
    figure_zt_sub.set_ylabel('z')
    figure_zt_sub.set_title('acceleration z-t Scatter')


def data_loader(data_windows=3, std_threshold=0.75, show_data=True, remove_data=True):
    data_files = []
    min = [1000, 1000, 1000]
    max = [0, 0, 0]
    # try:
    for root, dirs, files in os.walk(data_path, topdown=False):
        for name in files:
            with open(os.path.join(root, name), mode="r", encoding="utf-8-sig") as f:
                reader = csv.reader(f)
                header = next(reader)
                data = []
                for row in reader:
                    row_list = []
                    for item in row:
                        row_list.append(eval(item))
                    data.append(row_list)
                data_files.append(data)
                data_array = np.array(data).astype(np.float16)  # use low accuracy data to reduce calculation cost
                figure_scatter = plt.figure()
                if show_data:
                    _show_position_data(data_array, figure_scatter)
            with open(os.path.join(root, name), mode="r", encoding="utf-8-sig") as f:
                reader = csv.reader(f)
                header = next(reader)
                data = []
                for row in reader:
                    row_list = []
                    for item in row:
                        row_list.append(eval(item))

                    if row_list[0] > 1250 or row_list[0] < -1480 or row_list[1] > 750 or row_list[1] < -750 or row_list[
                        2] < 0 or row_list[
                        2] > 1050:  # remove the data out of limitation. According to measurement , the distance
                        # between the camera and ground is 3.25m, while the height of the table is 0.76m. Therefore, the
                        # limitation in z-axis is 2.5m.
                        pass
                    else:
                        data.append(row_list)
                data_files.append(data)
                data_array = np.array(data)
                removed_data_array = _remove_outliners(data_array.copy(), data_windows, std_threshold)
                direct_estimate_velocity_array = _velocity_direct_estimmation(data_array.copy())
                direct_estimate_acceleration_array = _force_estimation(direct_estimate_velocity_array.copy())
                estimate_velocity_array = _velocity_estimation(removed_data_array.copy(), vel_windows=7)
                removed_outline_velocity_array = _remove_outliners(estimate_velocity_array.copy(),
                                                                   data_windows, std_threshold)

                estimate_acceleration_array = _force_estimation(removed_outline_velocity_array.copy(), force_windows=7)
                removed_outline_acceleration_array = _remove_outliners(estimate_acceleration_array.copy(),
                                                                       data_windows, std_threshold)
                if show_data:
                    _show_position_data(removed_data_array, figure_scatter)
                    _show_velocity_data(direct_estimate_velocity_array, figure_scatter)
                    _show_velocity_data(estimate_velocity_array, figure_scatter)
                    _show_velocity_data(removed_outline_velocity_array, figure_scatter)
                    _show_acceleration_data(direct_estimate_acceleration_array, figure_scatter)
                    _show_acceleration_data(estimate_acceleration_array, figure_scatter)
                    _show_acceleration_data(removed_outline_acceleration_array, figure_scatter)
                plt.show()
                data_array_min = data_array.min(axis=0)
                data_array_max = data_array.max(axis=0)
                for index in range(3):

                    if min[index] > data_array_min[index]:
                        min[index] = data_array_min[index]
                    if max[index] < data_array_max[index]:
                        max[index] = data_array_max[index]
                print(f'min:{min}')
                print(f'max:{max}')

if __name__ == '__main__':
    data_loader()
    pass
