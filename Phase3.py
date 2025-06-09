from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import numpy as np
from sklearn.svm import SVR

class Three:
    def __init__(self, train, test, RUL):
        self.data_training = train
        self.data_test = test
        self.data_RUL = RUL

    def next_failure_prediction(self):
        # ==== PREPROCESS ====
        op_cols = [f'op_setting_{i}' for i in range(1, 4)]
        sensor_cols = [col for col in self.data_training.columns if 'sensor_' in col]

        # Check for missing columns
        missing = [col for col in op_cols if col not in self.data_training.columns]
        if missing:
            raise ValueError(f"Missing columns in data_training: {missing}")

        # Normalize operational settings
        op_scaler = StandardScaler()
        self.data_training[op_cols] = op_scaler.fit_transform(self.data_training[op_cols])

        # Add RUL to training data
        rul_max = self.data_training.groupby('unit_number')['time_in_cycles'].max().reset_index()
        rul_max.columns = ['unit_number', 'max_cycle']
        self.data_training = self.data_training.merge(rul_max, on='unit_number', how='left')
        self.data_training['RUL'] = self.data_training['max_cycle'] - self.data_training['time_in_cycles']
        self.data_training.drop('max_cycle', axis=1, inplace=True)

        # Add rolling stats
        def add_rolling(df, sensors, window=30):
            grouped = df.groupby('unit_number')
            for sensor in sensors:
                df[f'{sensor}_mean_{window}'] = grouped[sensor].rolling(window).mean().reset_index(0, drop=True)
                df[f'{sensor}_std_{window}'] = grouped[sensor].rolling(window).std().reset_index(0, drop=True)
            return df.dropna()

        self.data_training = add_rolling(self.data_training, sensor_cols)
        self.data_test = add_rolling(self.data_test, sensor_cols)

        # ==== TRAIN RIDGE REGRESSION ====
        feature_cols = [col for col in self.data_training.columns if (
            col in sensor_cols or col in op_cols or 'mean' in col or 'std' in col)]

        X_train = self.data_training[feature_cols]
        y_train = self.data_training['RUL']

        X_test_full = self.data_test.groupby('unit_number').tail(1).copy()
        self.data_RUL.columns = ['RUL']
        self.data_RUL['unit_number'] = range(1, len(self.data_RUL) + 1)
        X_test_full = X_test_full.merge(self.data_RUL, on='unit_number', how='left')

        X_test = X_test_full[feature_cols]

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Define and train SVR model
        svr = SVR(kernel='rbf', C=100, epsilon=5)
        svr.fit(X_train_scaled, y_train)

        # Predict RUL
        X_test_full['predicted_rul'] = svr.predict(X_test_scaled)

        # ==== STAGE MAPPING ====
        def label_stage(rul):
            if rul > 150:
                return 4  # Stage 1 - Normal
            elif 100 < rul <= 150:
                return 3
            elif 50 < rul <= 100:
                return 2
            elif 20 < rul <= 50:
                return 1
            else:
                return 0  # Failure

        X_test_full['predicted_stage'] = X_test_full['predicted_rul'].apply(label_stage)

        # ==== SIMULATE TIME TO NEXT STAGE ====
        def estimate_cycles_to_next_stage(rul, current_stage, drop_rate=1.0, max_cycles=50):
            for i in range(1, max_cycles + 1):
                next_rul = rul - i * drop_rate
                if label_stage(next_rul) < current_stage:
                    return i
            return None

        X_test_full['cycles_to_next_stage'] = X_test_full.apply(
            lambda row: estimate_cycles_to_next_stage(row['predicted_rul'], row['predicted_stage']),
            axis=1
        )

        # ==== EVALUATE RMSE ====

        if 'RUL' in X_test_full.columns:
            y_true = X_test_full['RUL']
            y_pred = X_test_full['predicted_rul']
            rmse = np.sqrt(mean_squared_error(y_true, y_pred))
            print(f"[Phase 3] RMSE for RUL prediction: {rmse:.2f}")
        else:
            print("Warning: RUL column missing in test set for RMSE evaluation.")

        # ==== OUTPUT ====
        output = X_test_full[['unit_number', 'predicted_rul', 'predicted_stage', 'cycles_to_next_stage']]

        return output