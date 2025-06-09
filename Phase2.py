from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

class Two:
    def __init__(self,train,test,rul):
        self.data_training=train
        self.data_testing=test
        self.data_rul=rul

    def model(self):
        def label_stage(rul):
            if rul > 150:
                return 4
            elif 101 < rul < 150:
                return 3
            elif 51 < rul < 100:
                return 2
            elif 21 < rul < 50:
                return 1
            else:
                return 0

        sensor_cols = [col for col in self.data_training.columns if 'sensor_' in col]
        op_cols = [col for col in self.data_training.columns if 'operational_setting' in col]
        window = 31

        group = self.data_training.groupby('unit_number')
        for sensor in sensor_cols:
            self.data_training[f'{sensor}_mean_{window}'] = group[sensor].rolling(window).mean().reset_index(0,
                                                                                                             drop=True)
            self.data_training[f'{sensor}_std_{window}'] = group[sensor].rolling(window).std().reset_index(0, drop=True)

        self.data_training.dropna(inplace=True)

        rul_max = self.data_training.groupby('unit_number')['time_in_cycles'].max().reset_index()
        rul_max.columns = ['unit_number', 'max_cycle']
        self.data_training = self.data_training.merge(rul_max, on='unit_number', how='left')
        self.data_training['RUL'] = self.data_training['max_cycle'] - self.data_training['time_in_cycles']
        self.data_training['stage'] = self.data_training['RUL'].apply(label_stage)
        self.data_training.drop('max_cycle', axis=1, inplace=True)

        feature_cols = [col for col in self.data_training.columns if (
                col in sensor_cols or col in op_cols or
                ('mean' in col or 'std' in col))]

        X_train = self.data_training[feature_cols]
        y_train = self.data_training['stage']

        group = self.data_testing.groupby('unit_number')
        for sensor in sensor_cols:
            self.data_testing[f'{sensor}_mean_{window}'] = group[sensor].rolling(window).mean().reset_index(0,
                                                                                                            drop=True)
            self.data_testing[f'{sensor}_std_{window}'] = group[sensor].rolling(window).std().reset_index(0, drop=True)

        self.data_testing.dropna(inplace=True)
        test_last = self.data_testing.groupby('unit_number').tail(1).copy()

        if isinstance(self.data_rul, pd.DataFrame) and self.data_rul.shape[1] == 1:
            self.data_rul.columns = ['RUL']
        self.data_rul['unit_number'] = range(1, len(self.data_rul) + 1)

        test_last = test_last.merge(self.data_rul, on='unit_number', how='left')
        test_last['stage'] = test_last['RUL'].apply(label_stage)
        X_test = test_last[feature_cols]

        model = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
        model.fit(X_train, y_train)
        test_last['predicted_stage'] = model.predict(X_test)
        test_last['failure_probability'] = model.predict_proba(X_test)[:, 0]

        # print the classification report
        print(classification_report(test_last['stage'] ,test_last['predicted_stage'], target_names=["Stage 1", "Stage 2", "Stage 3", "Stage 4","Stage 5"]))

        # forming the confusion matrix
        cm = confusion_matrix(test_last['stage'] ,test_last['predicted_stage'])
        # plotting of the diagram with all the specifications
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=["Stage 1", "Stage 2", "Stage 3", "Stage 4","Stage 5"],
                    yticklabels=["Stage 1", "Stage 2", "Stage 3", "Stage 4","Stage 5"])
        plt.xlabel('Predicted Stage')
        plt.ylabel('True Stage')
        plt.title('Confusion Matrix - Degradation Stage Classification')
        plt.show()

        print(test_last[['unit_number', 'predicted_stage', 'failure_probability']])
        return test_last[['unit_number', 'predicted_stage', 'failure_probability']]
