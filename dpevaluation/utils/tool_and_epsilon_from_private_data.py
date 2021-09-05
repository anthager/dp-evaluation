def tool_and_epsilon_from_private_data(private_data_path: str):
    dataset_dir = private_data_path.split('/')[-1]
    (tool, epsilon, _) = dataset_dir.split('_')
    return {'tool': tool, 'epsilon': epsilon}