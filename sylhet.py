# -*- coding: utf-8 -*-
"""Sylhet.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1fDuYwTJRvCkpmk7Ss_Wil32YHSwottM3

Exploratory Data Analysis (EDA)+ ML model developnment on Sylhet

import all the required libraries
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import seaborn as sbs

Sylhet=pd.read_csv(r"C:/Users/ThinkPad/Downloads/streamlit_sylhet/Sylhet.csv")
Sylhet.info() #3895 entries, 45 columns

"""Removing columns with no meaning and also having data mssing more than 70%"""

Sylhet.drop(['name', 'Unnamed: 0', 'stations', 'severerisk', 'snow', 'snowdepth', 'windgust', 'sunrise', 'sunset'], axis=1,inplace=True)
Sylhet.info() #45 to 36 columns

"""Change the data type of datetime column"""

Sylhet['datetime'] = pd.to_datetime(Sylhet['datetime'])
Sylhet.info()

"""To check correlations"""


# Compute the correlation matrix using Pandas
correlation_matrix = Sylhet.corr()

# Extract upper triangular part and keep only values > 0.85
upper_triangular = correlation_matrix.where(np.triu(np.ones(correlation_matrix.shape), k=1).astype(bool))
upper_triangular_high_correlations = upper_triangular[upper_triangular > 0.85]

# Create a mask to identify unique pairs
mask = ~upper_triangular_high_correlations.duplicated(keep="first")

# Display only unique pairs with correlations greater than 0.85
pd.set_option("display.max_rows", None)
print(upper_triangular_high_correlations[mask].stack())
pd.reset_option("display.max_rows")  # Reset the display setting

#remove high correlations columns
#precip sum and rain sum is same, if we keep both, then rain_sum will be entirely responsible for predicting preciptation sum
columns_to_drop = ['temp', 'feelslikemax', 'feelslikemin', 'feelslike', 'solarenergy', 'uvindex', 'precipitation_sum', 'temperature_2m_max',
                   'temperature_2m_min', 'temperature_2m_mean', 'apparent_temperature_mean']

# Drop the specified columns
Sylhet_1 = Sylhet.drop(columns=columns_to_drop, axis =1)
Sylhet_1.info() #25 columns remaining

from dython import nominal
corr_matrix = nominal.associations(Sylhet_1, figsize=(18, 18), mark_columns=True)

#remove high correlations columns
columns_to_drop = ['preciptype', 'conditions', 'description', 'icon']

# Drop the specified columns
Sylhet_2 = Sylhet_1.drop(columns=columns_to_drop, axis =1)
Sylhet_2.info() #21 columns remaining

"""Handle missing values"""

Sylhet_2.isnull().sum()

# Impute with Monthly Mean for 'sealevelpressure'
Sylhet_2['sealevelpressure'] = Sylhet_2.groupby(Sylhet_2['datetime'].dt.month)['sealevelpressure'].transform(lambda x: x.fillna(x.mean()))

# Impute with Monthly Mean for 'visibility'
Sylhet_2['visibility'] = Sylhet_2.groupby(Sylhet_2['datetime'].dt.month)['visibility'].transform(lambda x: x.fillna(x.mean()))
Sylhet_2.isnull().sum()

"""Boxplot for outliers

Time series plot
"""

fig = px.line(Sylhet_2, x='datetime', y='rain_sum', title='Time Series Plot for rain')
fig.update_xaxes(title_text='Datetime')
fig.update_yaxes(title_text='rain_sum')

fig.show()

fig = px.line(Sylhet_2, x='datetime', y='river_discharge', title='Time Series Plot for river discharge')
fig.update_xaxes(title_text='Datetime')
fig.update_yaxes(title_text='river_discharge')

fig.show()

fig = px.line(Sylhet_2, x='datetime', y='precip', title='Time Series Plot for river precipitation')
fig.update_xaxes(title_text='Datetime')
fig.update_yaxes(title_text='precipitation')

fig.show()

Sylhet_2.describe().transpose()

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()

# Identify and store the datetime column, if any
datetime_column = Sylhet_2.select_dtypes(include=['datetime64']).columns.tolist()

# Exclude the datetime column from scaling
numeric_columns = Sylhet_2.select_dtypes(include=['float64', 'int64']).columns.difference(datetime_column)
Sylhet_scaled = Sylhet_2.copy()
Sylhet_scaled[numeric_columns] = scaler.fit_transform(Sylhet_2[numeric_columns])

Sylhet_scaled.head(30)

"""Divding into train and test
As per instructed We are following this splitting:

Training Set: 2012-2019 Validation Set: 2020-2021 Testing Set: 2022-2023
"""

# Define target variables
target_variables = ['rain_sum', 'river_discharge', 'precip']

# Split the dataset based on the 'Year' column
# Set 'datetime' as the index for the training set
training_set = Sylhet_scaled[Sylhet_scaled['datetime'].dt.year <= 2019].set_index('datetime')

# Set 'datetime' as the index for the validation set
validation_set = Sylhet_scaled[(Sylhet_scaled['datetime'].dt.year >= 2020) & (Sylhet_scaled['datetime'].dt.year <= 2021)].set_index('datetime')

# Set 'datetime' as the index for the testing set
testing_set = Sylhet_scaled[Sylhet_scaled['datetime'].dt.year >= 2022].set_index('datetime')

# Separate input features and target variables for each set
X_train, y_train = training_set.drop(target_variables, axis=1), training_set[target_variables]
X_validation, y_validation = validation_set.drop(target_variables, axis=1), validation_set[target_variables]
X_test, y_test = testing_set.drop(target_variables, axis=1), testing_set[target_variables]

# Display the first few rows of X_train, y_train, X_validation, y_validation, X_test, y_test
print("Training Set:")
print("Input Features (X_train):")
print(X_train.info())
print("\nTarget Variables (y_train):")
print(y_train.info())

print("\nValidation Set:")
print("Input Features (X_validation):")
print(X_validation.info())
print("\nTarget Variables (y_validation):")
print(y_validation.info())

print("\nTesting Set:")
print("Input Features (X_test):")
print(X_test.info())
print("\nTarget Variables (y_test):")
print(y_test.info())

training_set.to_csv('/content/drive/MyDrive/data_Sylhet/train.csv')
validation_set.to_csv('/content/drive/MyDrive/data_Sylhet/val.csv')
testing_set.to_csv('/content/drive/MyDrive/data_Sylhet/test.csv')

"""Model building

XGBoost model referenced from Kyaw Htet Paing Win
"""

import xgboost as xgb
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import uniform, randint

# Define the parameter grid to search
param_grid = {
    'n_estimators': randint(100, 300),  # Smaller range for n_estimators
    'max_depth': randint(3, 7),  # Smaller range for max_depth
    'learning_rate': uniform(0.01, 0.2),  # Narrow range for learning_rate
    'subsample': uniform(0.6, 0.4),  # Range for subsample ratio
    'colsample_bytree': uniform(0.6, 0.4),  # Range for colsample_bytree
    'gamma': uniform(0, 1),  # Smaller range for gamma
    'min_child_weight': randint(1, 6)  # Smaller range for min_child_weight
}

# Define the model to tune
xgb_model = xgb.XGBRegressor(objective='reg:squarederror')

# Setup the randomized search with cross-validation
random_search = RandomizedSearchCV(
    estimator=xgb_model,
    param_distributions=param_grid,
    n_iter=50,  # Reduced number of iterations for initial search
    scoring='neg_mean_squared_error',  # Mean Squared Error as scoring
    n_jobs=-1,  # Use all cores
    cv=5,  # Increased number of folds in cross-validation for reliability
    verbose=1,  # Reduced verbosity for less output
    random_state=42  # For reproducibility
)

# Perform the randomized search over the parameter grid
random_search.fit(X_train, y_train)

# Print the best parameters and lowest RMSE
best_parameters = random_search.best_params_
lowest_rmse = np.sqrt(-random_search.best_score_)

print(f"Best parameters found: {best_parameters}")
print(f"Lowest RMSE found: {lowest_rmse}")

# Retrain with the best parameters on the full dataset
final_model = xgb.XGBRegressor(
    objective='reg:squarederror',
    **best_parameters  # Unpack the best parameters
)
final_model.fit(X_train, y_train)

fi = pd.DataFrame(final_model.feature_importances_,
                  index=X_train.columns, columns=['importance'])

fi = fi.sort_values('importance')

fig, ax = plt.subplots(figsize=(8, 12))

fi.plot.barh(title="Feature Importance", ax=ax)

ax.set_yticklabels(fi.index, rotation=0)
ax.tick_params(axis='both', which='major', labelsize=12)

ax.set_ylabel('')
ax.set_xlabel('Importance')

fig.tight_layout()

plt.show()

# Generate predictions on the test set
y_pred = final_model.predict(X_test)

# Calculate the RMSE (Root Mean Squared Error)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

# Calculate the MAE (Mean Absolute Error)
mae = mean_absolute_error(y_test, y_pred)

# Calculate the R^2 score
r2 = r2_score(y_test, y_pred)

# Print out the metrics
print(f"RMSE: {rmse}") #0.0442
print(f"MAE: {mae}") #0.0221
print(f"R^2: {r2}") #0.655

# Print shapes of y_test and y_pred
print("y_test shape:", y_test.shape)
print("y_pred shape:", y_pred.shape)

# Convert the NumPy array to a DataFrame
y_pred_df = pd.DataFrame(y_pred, columns=['precip', 'rain_sum', 'river_discharge'])

# Now you can use head() on the DataFrame
print("y_pred_df head:\n", y_pred_df.head())

"""plot for actual vs predicted values for test dataset"""

def create_individual_plot(y_test, y_pred, variable_name):
    df_plot = pd.DataFrame({
        'Datetime': X_test.index,
        f'Actual_{variable_name}': y_test[variable_name],
        f'Predicted_{variable_name}': y_pred[:, y_test.columns.get_loc(variable_name)],
    })

    # Create an interactive plot using Plotly Express
    fig = px.line(df_plot, x='Datetime', y=[f'Actual_{variable_name}', f'Predicted_{variable_name}'],
                  labels={'value': 'Rainfall (mm)', 'Datetime': 'Datetime'},
                  title=f'Actual vs Predicted {variable_name} Rainfall')

    # Show the plot
    fig.show()

# Create individual plots for each variable
for variable_name in ['precip', 'rain_sum', 'river_discharge']:
    create_individual_plot(y_test, y_pred, variable_name)

"""Precip is predicted less than actual values"""

# Initialize dictionaries to store metrics for each variable
mse_dict = {}
mae_dict = {}

# Function to calculate metrics for each variable
def calculate_metrics(y_true, y_pred, variable_name):
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)

    mse_dict[variable_name] = mse
    mae_dict[variable_name] = mae

    print(f"{variable_name} MSE: {mse}")
    print(f"{variable_name} MAE: {mae}")
    print("")

# Calculate metrics for each variable
for i, variable_name in enumerate(['precip', 'rain_sum', 'river_discharge']):
    y_true = y_test[variable_name]
    y_pred_single_variable = y_pred[:, i]

    calculate_metrics(y_true, y_pred_single_variable, variable_name)

# Combine metrics if needed
combined_mse = sum(mse_dict.values()) / len(mse_dict)
combined_mae = sum(mae_dict.values()) / len(mae_dict)

print("Combined MSE: ", combined_mse)
print("Combined MAE: ", combined_mae)

import joblib
cwd = r'C:\Users\ThinkPad\Downloads\streamlit_sylhet'
joblib.dump(final_model, cwd + '/final_xgboost_sylhet1.joblib')

"""Tensorflow and LSTM (referenced from Vijay Mamillia)"""

import tensorflow as tf

from tensorflow.keras import Model, Sequential

from tensorflow.keras.optimizers import Adam
from tensorflow import keras
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.metrics import MeanAbsoluteError

from tensorflow.keras.layers import Dense, Conv1D, LSTM, Lambda, Reshape, RNN, LSTMCell

import warnings
warnings.filterwarnings('ignore')

train_df = pd.read_csv('/content/drive/MyDrive/data_Sylhet/train.csv', index_col='datetime',parse_dates=True)
val_df = pd.read_csv('/content/drive/MyDrive/data_Sylhet/val.csv', index_col='datetime',parse_dates=True)
test_df = pd.read_csv('/content/drive/MyDrive/data_Sylhet/test.csv', index_col='datetime',parse_dates=True)

print(train_df.shape, val_df.shape, test_df.shape)

tf.random.set_seed(42)
np.random.seed(42)

class DataWindow():
    def __init__(self, input_width, label_width, shift,
                 train_df=train_df, val_df=val_df, test_df=test_df,
                 label_columns=None):

        self.train_df = train_df
        self.val_df = val_df
        self.test_df = test_df

        self.label_columns = label_columns
        if label_columns is not None:
            self.label_columns_indices = {name: i for i, name in enumerate(label_columns)}
        self.column_indices = {name: i for i, name in enumerate(train_df.columns)}

        self.input_width = input_width
        self.label_width = label_width
        self.shift = shift

        self.total_window_size = input_width + shift

        self.input_slice = slice(0, input_width)
        self.input_indices = np.arange(self.total_window_size)[self.input_slice]

        self.label_start = self.total_window_size - self.label_width
        self.labels_slice = slice(self.label_start, None)
        self.label_indices = np.arange(self.total_window_size)[self.labels_slice]

    def split_to_inputs_labels(self, features):
        inputs = features[:, self.input_slice, :]
        labels = features[:, self.labels_slice, :]
        if self.label_columns is not None:
            labels = tf.stack(
                [labels[:,:,self.column_indices[name]] for name in self.label_columns],
                axis=-1
            )
        inputs.set_shape([None, self.input_width, None])
        labels.set_shape([None, self.label_width, None])

        return inputs, labels


    def make_dataset(self, data):
        data = np.array(data, dtype=np.float32)
        ds = tf.keras.preprocessing.timeseries_dataset_from_array(
            data=data,
            targets=None,
            sequence_length=self.total_window_size,
            sequence_stride=1,
            shuffle=True,
            batch_size=32
        )

        ds = ds.map(self.split_to_inputs_labels)
        return ds

    @property
    def train(self):
        return self.make_dataset(self.train_df)

    @property
    def val(self):
        return self.make_dataset(self.val_df)

    @property
    def test(self):
        return self.make_dataset(self.test_df)

    @property
    def sample_batch(self):
        result = getattr(self, '_sample_batch', None)
        if result is None:
            result = next(iter(self.train))
            self._sample_batch = result
        return result

class Baseline(Model):
    def __init__(self, label_index=None):
        super().__init__()
        self.label_index = label_index

    def call(self, inputs):
        if self.label_index is None:
            return inputs

        elif isinstance(self.label_index, list):
            tensors = []
            for index in self.label_index:
                result = inputs[:, :, index]
                result = result[:, :, tf.newaxis]
                tensors.append(result)
            return tf.concat(tensors, axis=-1)

        result = inputs[:, :, self.label_index]
        return result[:,:,tf.newaxis]

"""Multi-output baseline model"""

mo_single_step_window = DataWindow(input_width=1, label_width=1, shift=1, label_columns=['precip','rain_sum','river_discharge'])
mo_wide_window = DataWindow(input_width=14, label_width=14, shift=1, label_columns=['precip','rain_sum','river_discharge'])

column_indices = {name: i for i, name in enumerate(train_df.columns)}
print(column_indices['precip'])
print(column_indices['rain_sum'])
print(column_indices['river_discharge'])

mo_baseline_last = Baseline(label_index=[4,15,19])

mo_baseline_last.compile(loss=MeanSquaredError(), metrics=[MeanAbsoluteError()])

mo_val_performance = {}
mo_performance = {}

mo_val_performance['Baseline - Last'] = mo_baseline_last.evaluate(mo_wide_window.val)
mo_performance['Baseline - Last'] = mo_baseline_last.evaluate(mo_wide_window.test, verbose=0)

print(mo_performance['Baseline - Last'][1])

"""Implementing a deep neural network as a multi-output model"""

def compile_and_fit(model, window, patience=3, max_epochs=50):
    early_stopping = EarlyStopping(monitor='val_loss',
                                   patience=patience,
                                   mode='min')

    model.compile(loss=MeanSquaredError(),
                  optimizer=Adam(),
                  metrics=[MeanAbsoluteError()])

    history = model.fit(window.train,
                       epochs=max_epochs,
                       validation_data=window.val,
                       callbacks=[early_stopping])

    return history

mo_dense = Sequential([
    Dense(units=64, activation='relu'),
    Dense(units=64, activation='relu'),
    Dense(units=3)
])

history = compile_and_fit(mo_dense, mo_single_step_window)

mo_val_performance['Dense'] = mo_dense.evaluate(mo_single_step_window.val)
mo_performance['Dense'] = mo_dense.evaluate(mo_single_step_window.test, verbose=0)

"""LSTM"""

mo_lstm_model = Sequential([
    LSTM(32, return_sequences=True),
    Dense(units = 3)
])

history = compile_and_fit(mo_lstm_model, mo_wide_window)

mo_val_performance = {}
mo_performance = {}

mo_val_performance['LSTM'] = mo_lstm_model.evaluate(mo_wide_window.val)
mo_performance['LSTM'] = mo_lstm_model.evaluate(mo_wide_window.test, verbose=0)

predicted_results = mo_lstm_model.predict(mo_wide_window.test)
predicted_array= predicted_results[0]

my_array = np.array(predicted_array)

df_raw = pd.DataFrame(my_array)

df = df_raw.rename(columns={0: "river_discharge", 1: "rain_sum",2:"precip"})


df.head(14)

mo_lstm_model.save("/content/drive/MyDrive/data_Sylhet/lstm_sylhet_model.h5")

"""XGBoost gives MAE as 0.021 and LSTM gives 0.029, we can use XGBoost as it is more simple"""

sylhet_model = keras.models.load_model("/content/drive/MyDrive/data_Sylhet/lstm_sylhet_model.h5")

class DataWindow:
    def __init__(self, input_width, label_width, shift, test_df, label_columns=None):
        self.test_df = test_df
        self.label_columns = label_columns
        if label_columns is not None:
            self.label_columns_indices = {name: i for i, name in enumerate(label_columns)}
        self.column_indices = {name: i for i, name in enumerate(test_df.columns)}
        self.input_width = input_width
        self.label_width = label_width
        self.shift = shift
        self.total_window_size = input_width + shift
        self.input_slice = slice(0, input_width)
        self.input_indices = np.arange(self.total_window_size)[self.input_slice]
        self.label_start = self.total_window_size - self.label_width
        self.labels_slice = slice(self.label_start, None)
        self.label_indices = np.arange(self.total_window_size)[self.labels_slice]

    def split_to_inputs_labels(self, features):
        inputs = features[:, self.input_slice, :]
        labels = features[:, self.labels_slice, :]
        if self.label_columns is not None:
            labels = tf.stack(
                [labels[:, :, self.column_indices[name]] for name in self.label_columns],
                axis=-1
            )
        inputs.set_shape([None, self.input_width, None])
        labels.set_shape([None, self.label_width, None])

        return inputs, labels

    def make_dataset(self, data):
        data = np.array(data, dtype=np.float32)
        ds = tf.keras.preprocessing.timeseries_dataset_from_array(
            data=data,
            targets=None,
            sequence_length=self.total_window_size,
            sequence_stride=1,
            shuffle=True,
            batch_size=32
        )

        ds = ds.map(self.split_to_inputs_labels)
        return ds

    @property
    def test(self):
        return self.make_dataset(self.test_df)

def predict(days: int):
    custom_mo_wide_window = DataWindow(input_width=days, label_width=days, shift=days, test_df=test_df,
                                       label_columns=['precip','rain_sum','river_discharge'])

    predicted_results = sylhet_model.predict(custom_mo_wide_window.test)
    predicted_array= predicted_results[0]

    predicted_numpy_array = np.array(predicted_array)

    df_scaled = pd.DataFrame(predicted_numpy_array)

    df = df_scaled.rename(columns={0: "river_discharge", 1: "rain_sum",2:"precip"})

    '''RD_max_train = 1.0
    RD_min_train = 0.0008077544426494
    RD_max_test =  0.7019386106623586
    RD_min_test =  0.0

    R_max_train = 1.0
    R_min_train = 0.0
    R_max_test = 0.4720101781170483
    R_min_test = 0.0

    P_max_train = 0.78
    P_min_train = 0.0
    P_max_test = 0.765
    P_min_test = 0.0'''

    RD_max_train = 25.61
    RD_min_train = 0.85

    R_max_train = 235.80
    R_min_train = 0.0

    P_max_train = 200.00
    P_min_train = 0.0



    df['river_discharge'] = df['river_discharge'].apply(lambda x: x*(RD_max_train - RD_min_train) + RD_min_train)
    df['rain_sum'] = df['rain_sum'].apply(lambda x: x*(R_max_train - R_min_train) + R_min_train)
    df['precip'] = df['precip'].apply(lambda x: x*(P_max_train - P_min_train) + P_min_train)
    df['floods'] = df['precip'] >2;

    return df

predict(14)