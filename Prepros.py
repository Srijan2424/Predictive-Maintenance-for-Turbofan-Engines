import pandas as pd

class Pre:
    def __init__(self):
        pass
    def pre_a_cl(self):
        # ===  Dataset Info ===
        DATASETS = {
            'FD001': {'path': '/Users/srijanchopra/Desktop/college projects/ml class project /ml project /CMaps/train_FD001.txt','ops_vary': False, 'fault_modes': 1},
            'FD002': {'path': '/Users/srijanchopra/Desktop/college projects/ml class project /ml project /CMaps/train_FD002.txt','ops_vary': True, 'fault_modes': 1},
            'FD003': {'path': '/Users/srijanchopra/Desktop/college projects/ml class project /ml project /CMaps/train_FD003.txt','ops_vary': False, 'fault_modes': 2},
            'FD004': {'path': '/Users/srijanchopra/Desktop/college projects/ml class project /ml project /CMaps/train_FD004.txt','ops_vary': True, 'fault_modes': 2},
        }
        COLUMNS = ['unit', 'time_in_cycles'] + \
                  [f'op_set_{i}' for i in range(1, 4)] + \
                  [f'sensor_{i}' for i in range(1, 22)]
        # ===  Load and Combine Datasets ===
        df_all = []

        for dataset_id, meta in DATASETS.items():
            df = pd.read_csv(meta['path'], sep='\s+', header=None)
            df.columns = COLUMNS
            df['dataset_id'] = dataset_id
            df['fault_modes'] = meta['fault_modes']
            df['ops_vary'] = meta['ops_vary']
            df_all.append(df)
        print(df)
        return df
    def pre_b_cl(self):
        REDUNDANT_COLS = [
            'unit', 'time_in_cycles',
            'op_set_1', 'op_set_2', 'op_set_3',  # Constant for FD001/FD003
            'sensor_1', 'sensor_5', 'sensor_6',
            'sensor_10', 'sensor_16', 'sensor_18', 'sensor_19'
        ]

        COLUMNS = ['unit', 'time_in_cycles'] + \
                  [f'op_set_{i}' for i in range(1, 4)] + \
                  [f'sensor_{i}' for i in range(1, 22)]

        # ===  Load Function ===
        def load_cmapss(file_path, dataset_id):
            df = pd.read_csv(file_path, sep='\s+', header=None)
            df.columns = COLUMNS
            df['dataset_id'] = dataset_id
            return df

        # ===  Load FD001 and FD003 ===
        df_fd001 = load_cmapss("/Users/srijanchopra/Desktop/college projects/ml class project /ml project /CMaps/train_FD001.txt", "FD001")
        df_fd003 = load_cmapss("/Users/srijanchopra/Desktop/college projects/ml class project /ml project /CMaps/train_FD003.txt", "FD003")
        df = pd.concat([df_fd001, df_fd003], ignore_index=True)
        print(f"✅ Loaded Group A (FD001 + FD003): {df.shape}")

        # === Drop Redundant Columns ===
        df_clean = df.drop(columns=REDUNDANT_COLS + ['dataset_id'])

        return df,df_clean

    def pre_c_cl(self):

        COLUMNS = ['unit', 'time_in_cycles'] + \
                  [f'op_set_{i}' for i in range(1, 4)] + \
                  [f'sensor_{i}' for i in range(1, 22)]

        REDUNDANT_COLS = [
            'unit', 'time_in_cycles',  # Still redundant
            'sensor_1', 'sensor_5', 'sensor_6',
            'sensor_10', 'sensor_16', 'sensor_18', 'sensor_19'
        ]
        # Note: DO NOT remove op_set_1–3 (they vary in FD002/FD004)

        # ===  Load Function ===
        def load_cmapss(file_path, dataset_id):
            df = pd.read_csv(file_path, sep='\s+', header=None)
            df.columns = COLUMNS
            df['dataset_id'] = dataset_id
            return df

        # ===  Load FD002 and FD004 ===
        df_fd002 = load_cmapss("/Users/srijanchopra/Desktop/college projects/ml class project /ml project /CMaps/train_FD002.txt", "FD002")
        df_fd004 = load_cmapss("/Users/srijanchopra/Desktop/college projects/ml class project /ml project /CMaps/train_FD004.txt", "FD004")
        df = pd.concat([df_fd002, df_fd004], ignore_index=True)
        print(f"✅ Loaded Group B (FD002 + FD004): {df.shape}")

        # ===  Drop Redundant Columns ===
        df_clean = df.drop(columns=REDUNDANT_COLS + ['dataset_id'])

        return df_clean,df

    def reg_a(self):
        # training data
        DATASETS = {
            'FD001': {'path': '/Users/srijanchopra/Desktop/college projects/ml class project /ml project /CMaps/train_FD001.txt','ops_vary': False, 'fault_modes': 1},
            # 'FD002': {'path': '/Users/srijanchopra/Desktop/college projects/ml class project /ml project /CMaps/train_FD002.txt','ops_vary': True, 'fault_modes': 1},
            # 'FD003': {'path': '/Users/srijanchopra/Desktop/college projects/ml class project /ml project /CMaps/train_FD003.txt','ops_vary': False, 'fault_modes': 2},
            # 'FD004': {'path': '/Users/srijanchopra/Desktop/college projects/ml class project /ml project /CMaps/train_FD004.txt','ops_vary': True, 'fault_modes': 2},
        }

        columns = ['unit_number', 'time_in_cycles'] + \
                  [f'op_setting_{i}' for i in range(1, 4)] + \
                  [f'sensor_{i}' for i in range(1, 22)]

        # ===  Load and Combine Datasets ===
        df_all = []
        for dataset_id, meta in DATASETS.items():
            df_train = pd.read_csv(meta['path'], sep='\s+', header=None)
            df_train.columns = columns
            df_all.append(df_train)

        # testing data
        DATASETS1 = {
            'FD001': {'path': '/Users/srijanchopra/Desktop/college projects/ml class project /ml project /CMaps/test_FD001.txt','ops_vary': False, 'fault_modes': 1},
            # 'FD002': {'path': '/Users/srijanchopra/Desktop/college projects/ml class project /ml project /CMaps/test_FD002.txt','ops_vary': True, 'fault_modes': 1},
            # 'FD003': {'path': '/Users/srijanchopra/Desktop/college projects/ml class project /ml project /CMaps/test_FD003.txt','ops_vary': False, 'fault_modes': 2},
            # 'FD004': {'path': '/Users/srijanchopra/Desktop/college projects/ml class project /ml project /CMaps/test_FD004.txt','ops_vary': True, 'fault_modes': 2},
        }
        # ===  Load and Combine Datasets ===
        df_all1 = []
        for dataset_id, meta in DATASETS1.items():
            df_test = pd.read_csv(meta['path'], sep='\s+', header=None)
            df_test.columns = columns
            df_all1.append(df_test)

        # RUL
        DATASETS2 = {
            'FD001': {'path': '/Users/srijanchopra/Desktop/college projects/ml class project /ml project /CMaps/RUL_FD001.txt','ops_vary': False, 'fault_modes': 1},
            # 'FD002': {'path': '/Users/srijanchopra/Desktop/college projects/ml class project /ml project /CMaps/RUL_FD002.txt','ops_vary': True, 'fault_modes': 1},
            # 'FD003': {'path': '/Users/srijanchopra/Desktop/college projects/ml class project /ml project /CMaps/RUL_FD003.txt','ops_vary': False, 'fault_modes': 2},
            # 'FD004': {'path': '/Users/srijanchopra/Desktop/college projects/ml class project /ml project /CMaps/RUL_FD004.txt','ops_vary': True, 'fault_modes': 2},
        }

        for dataset_id, meta in DATASETS2.items():
            df_rul= pd.read_csv(meta['path'], sep='\s+', header=None)

        return df_train,df_test,df_rul

    def reg_b(self):
        # training data
        DATASETS = {
            'FD001': {'path': '/Users/srijanchopra/Desktop/college projects/ml class project /ml project /CMaps/train_FD001.txt','ops_vary': False, 'fault_modes': 1},
            'FD003': {'path': '/Users/srijanchopra/Desktop/college projects/ml class project /ml project /CMaps/train_FD003.txt','ops_vary': False, 'fault_modes': 2},
        }

        columns = ['unit_number', 'time_in_cycles'] + \
                  [f'op_setting_{i}' for i in range(1, 4)] + \
                  [f'sensor_{i}' for i in range(1, 22)]

        # ===  Load and Combine Datasets ===
        df_all = []
        for dataset_id, meta in DATASETS.items():
            df_train = pd.read_csv(meta['path'], sep='\s+', header=None)
            df_train.columns = columns
            df_all.append(df_train)

        # testing data
        DATASETS1 = {
            'FD001': {'path': '/Users/srijanchopra/Desktop/college projects/ml class project /ml project /CMaps/test_FD001.txt','ops_vary': False, 'fault_modes': 1},
            'FD003': {'path': '/Users/srijanchopra/Desktop/college projects/ml class project /ml project /CMaps/test_FD003.txt','ops_vary': False, 'fault_modes': 2},
        }
        # ===  Load and Combine Datasets ===
        df_all1 = []
        for dataset_id, meta in DATASETS1.items():
            df_test = pd.read_csv(meta['path'], sep='\s+', header=None)
            df_test.columns = columns
            df_all1.append(df_test)

        # RUL
        DATASETS2 = {
            'FD001': {'path': '/Users/srijanchopra/Desktop/college projects/ml class project /ml project /CMaps/RUL_FD001.txt','ops_vary': False, 'fault_modes': 1},
            'FD003': {'path': '/Users/srijanchopra/Desktop/college projects/ml class project /ml project /CMaps/RUL_FD003.txt','ops_vary': False, 'fault_modes': 2},
        }

        for dataset_id, meta in DATASETS2.items():
            df_rul= pd.read_csv(meta['path'], sep='\s+', header=None)

        return df_train, df_test, df_rul

    def reg_c(self):
        # training data
        DATASETS = {
            'FD002': {'path': '/Users/srijanchopra/Desktop/college projects/ml class project /ml project /CMaps/train_FD002.txt','ops_vary': True, 'fault_modes': 1},
            'FD004': {'path': '/Users/srijanchopra/Desktop/college projects/ml class project /ml project /CMaps/train_FD004.txt','ops_vary': True, 'fault_modes': 2},
        }

        columns = ['unit_number', 'time_in_cycles'] + \
                  [f'op_setting_{i}' for i in range(1, 4)] + \
                  [f'sensor_{i}' for i in range(1, 22)]

        # ===  Load and Combine Datasets ===
        df_all = []
        for dataset_id, meta in DATASETS.items():
            df_train = pd.read_csv(meta['path'], sep='\s+', header=None)
            df_train.columns = columns
            df_all.append(df_train)

        # testing data
        DATASETS1 = {
            'FD002': {'path': '/Users/srijanchopra/Desktop/college projects/ml class project /ml project /CMaps/test_FD002.txt','ops_vary': True, 'fault_modes': 1},
            'FD004': {'path': '/Users/srijanchopra/Desktop/college projects/ml class project /ml project /CMaps/test_FD004.txt','ops_vary': True, 'fault_modes': 2},
        }
        # ===  Load and Combine Datasets ===
        df_all1 = []
        for dataset_id, meta in DATASETS1.items():
            df_test = pd.read_csv(meta['path'], sep='\s+', header=None)
            df_test.columns = columns
            df_all1.append(df_test)

        # RUL
        DATASETS2 = {
            'FD002': {'path': '/Users/srijanchopra/Desktop/college projects/ml class project /ml project /CMaps/RUL_FD002.txt','ops_vary': True, 'fault_modes': 1},
            'FD004': {'path': '/Users/srijanchopra/Desktop/college projects/ml class project /ml project /CMaps/RUL_FD004.txt','ops_vary': True, 'fault_modes': 2},
        }

        for dataset_id, meta in DATASETS2.items():
            df_rul = pd.read_csv(meta['path'], sep='\s+', header=None)

        return df_train, df_test, df_rul