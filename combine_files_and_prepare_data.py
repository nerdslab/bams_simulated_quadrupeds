import numpy as np


def filter_data(data):
    X, Y = data['base_pos'][:, 0, :-1], data['base_pos'][:, 1, :-1]

    # filter data
    fail_mask = np.any(data['base_pos'][:, 2, :-1] < -4, axis=1)
    respawn_mask = np.zeros_like(X)
    respawn_mask[:, 1:] = np.abs(X[:, 1:] - X[:, :-1]) > 1
    respawn_mask = np.any(respawn_mask, axis=1)

    edge_x = data['terrain_map'][(29, 1)]['origins'][0] + 12.
    edge_mask = np.any(data['base_pos'][:, 0, :-1] > edge_x, axis=1)

    rm_mask = np.logical_or(np.logical_or(respawn_mask, fail_mask), edge_mask)
    print(f'{data["task"]}-{data["command_vel"]}: Removing {np.sum(rm_mask)} robots out of {len(rm_mask)}.')
    keep_mask = np.logical_not(rm_mask)
    return keep_mask

def get_label_maps(terrain_map):
    num_rows = 30
    num_cols = 5

    terrain_type_map = np.ones((num_rows * 2, num_cols), dtype=np.uint8)
    terrain_type_map_2 = np.ones((num_rows * 2, num_cols), dtype=np.uint8)
    slope_map = np.zeros((num_rows * 2, num_cols), dtype=float)
    rough_map = np.zeros((num_rows * 2, num_cols), dtype=bool)
    difficulty_map = 0.1 * np.ones((num_rows * 2, num_cols), dtype=float)

    for (row_id, col_id), subterrain in terrain_map.items():
        terrain_type = subterrain['terrain_type']
        terrain_param = subterrain['terrain_param']
        terrain_difficulty = subterrain['difficulty']
        if terrain_type in ['flat', 'flat+noise']:
            terrain_type_map[2*row_id:2*(row_id+1), col_id] = 0
            terrain_type_map_2[2*row_id:2*(row_id+1), col_id] = 0
            if terrain_type == 'flat+noise':
                difficulty_map[2*row_id:2*(row_id+1), col_id] = 0.2
        elif terrain_type == 'discrete_obstacles':
            terrain_type_map[2*row_id:2*(row_id+1), col_id] = 0
            terrain_type_map_2[2 * row_id:2 * (row_id + 1), col_id] = 0
            rough_map[2*row_id:2*(row_id+1), col_id] = True
            difficulty_map[2*row_id:2*(row_id+1), col_id] = terrain_difficulty
        elif 'pyramid_slope_up' in terrain_type:
            if terrain_param <= 0.21:
                terrain_type_map[2*row_id:2*(row_id+1), col_id] = 0
                terrain_type_map_2[2 * row_id:2 * (row_id + 1), col_id] = 0
            else:
                terrain_type_map[2*row_id:2*(row_id+1), col_id] = 1
                terrain_type_map_2[2 * row_id, col_id] = 1
                terrain_type_map_2[2 * row_id+1, col_id] = 2
            slope_map[2*row_id, col_id] = terrain_param
            slope_map[2*row_id+1, col_id] = -terrain_param
            difficulty_map[2*row_id:2*(row_id+1), col_id] = terrain_difficulty
        elif 'pyramid_slope_down' in terrain_type:
            if np.abs(terrain_param) <= 0.21:
                terrain_type_map[2*row_id:2*(row_id+1), col_id] = 0
                terrain_type_map_2[2 * row_id:2 * (row_id + 1), col_id] = 0
            else:
                terrain_type_map[2*row_id:2*(row_id+1), col_id] = 2
                terrain_type_map_2[2 * row_id, col_id] = 2
                terrain_type_map_2[2 * row_id+1, col_id] = 1
            slope_map[2*row_id, col_id] = -np.abs(terrain_param)
            slope_map[2*row_id+1, col_id] = np.abs(terrain_param)
            difficulty_map[2*row_id:2*(row_id+1), col_id] = terrain_difficulty
        elif 'pyramid_stairs_up' in terrain_type:
            terrain_type_map[2*row_id:2*(row_id+1), col_id] = 3
            terrain_type_map_2[2 * row_id, col_id] = 3
            terrain_type_map_2[2 * row_id + 1, col_id] = 4
            slope_map[2*row_id, col_id] = terrain_param
            slope_map[2*row_id+1, col_id] = -terrain_param
            difficulty_map[2*row_id:2*(row_id+1), col_id] = terrain_difficulty
        elif 'pyramid_stairs_down' in terrain_type:
            terrain_type_map[2*row_id:2*(row_id+1), col_id] = 4
            terrain_type_map_2[2 * row_id, col_id] = 4
            terrain_type_map_2[2 * row_id + 1, col_id] = 3
            slope_map[2*row_id, col_id] = -np.abs(terrain_param)
            slope_map[2*row_id+1, col_id] = np.abs(terrain_param)
            difficulty_map[2*row_id:2*(row_id+1), col_id] = terrain_difficulty
        else:
            raise NotImplementedError

        if 'noise' in terrain_type:
            rough_map[2*row_id:2*(row_id+1), col_id] = True
    return terrain_type_map, terrain_type_map_2, slope_map, rough_map, difficulty_map
def extract_labels(data, keep_mask):
    X, Y = data['base_pos'][keep_mask, 0, :-1], data['base_pos'][keep_mask, 1, :-1]
    terrain_map = data['terrain_map']

    origin_x, origin_y, _ = terrain_map[(0, 0)]['origins']
    hx, hy, _ = (terrain_map[(1, 1)]['origins'] - terrain_map[(0, 0)]['origins'])

    row = ((X - origin_x) / (hx/2)).astype(int)
    col = ((Y - origin_y) / hy).astype(int)

    terrain_type_map, terrain_type_map_2, slope_map, rough_map, difficulty_map = get_label_maps(terrain_map)

    terrain_type = terrain_type_map[row, col]
    terrain_type_2 = terrain_type_map_2[row, col]
    terrain_difficulty = difficulty_map[row, col]
    terrain_slope = slope_map[row, col]
    terrain_rough = rough_map[row, col]
    return terrain_type, terrain_type_2, terrain_difficulty, terrain_slope, terrain_rough

def process(filename):
    data = np.load(filename, allow_pickle=True).item()

    keep_mask = filter_data(data)
    terrain_type, terrain_type_2, terrain_difficulty, terrain_slope, terrain_rough = extract_labels(data, keep_mask)

    robot_id = {'anymal_b': 0, 'anymal_c_rough': 1, 'a1': 2}

    data = dict(
        base_pos=data['base_pos'][keep_mask, :, :-1],
        base_lin_vel=data['base_lin_vel'][keep_mask, :, :-1],
        base_ang_vel=data['base_ang_vel'][keep_mask, :, :-1],
        dof_pos=data['dof_pos'][keep_mask, :, :-1],
        dof_vel=data['dof_vel'][keep_mask, :, :-1],
        # global
        body_mass=np.array(data['body_mass'])[keep_mask],
        robot_type=np.ones(np.sum(keep_mask)) * robot_id[data['task']],
        command_vel=np.ones(np.sum(keep_mask)) * data['command_vel'],
        # local
        terrain_type=terrain_type,
        terrain_type_2=terrain_type_2,
        terrain_difficulty=terrain_difficulty,
        terrain_slope=terrain_slope,
        terrain_rough=terrain_rough,
        reward=data['reward'][keep_mask, :-1],
        # metadata
        names=dict(
            dof_names=data['dof_names'],
            robot_names=['anymal_b', 'anymal_c'],
            terrain_names=['flat', 'pyramid_slope', 'inverted_pyramid_slope', 'pyramid_stairs', 'inverted_pyramid_stairs'],
            terrain_names_2=['flat', 'slope_up', 'slope_down', 'stairs_up', 'stairs_down'],
        )
    )
    return data

from glob import glob
filenames = glob('./data/*npy')

mega_data = process(filenames[0])

for filename in filenames[1:]:
    data = process(filename)
    for key in mega_data.keys():
        if key != 'names':
            mega_data[key] = np.concatenate([mega_data[key], data[key]], axis=0)

np.save('robot_dataset.npy', mega_data)
