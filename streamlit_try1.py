import os
import pandas as pd 
import numpy as np 
import datetime as dt
from datetime import datetime
import joblib
import streamlit as st
import sklearn
import plotly.express as px
import plotly.graph_objs as go
# from keras.models import load_model


# bangladesh_image = os.path.join(ARTIFACTORY_DIR, "/mount/src/bangladesh-flood-guard/src/tasks/task-4-model-deployment/streamlit/app/artifactory/bangladesh_map.jpg")

st.header("Flood Guard - Bangladesh Sylhet Region")

#st.image("/mount/src/bangladesh-flood-guard/src/tasks/task-4-model-deployment/streamlit/app/artifactory/bangladesh_map.jpg")
st.image("Shah_Jalal_Mazar_at_Sylhet.jpeg", caption='Shah Jalal Mazar', use_column_width=True)


st.write("""
    Bangladesh is a country in South Asia that is known for its rich natural beauty and biodiversity. 
    It has the world's largest delta, formed by the confluence of the Ganges, Brahmaputra and Meghna rivers, which supports a variety of ecosystems and wildlife. 
    Sylhet is a metropolitan city located in the northeastern region of Bangladesh. 
    It is situated on the banks of the Surma River. The city has a population of approximately 700,000 people, making it fifth-largest city in Bangladesh.
    """)

st.write("""
    **Problem Statement:**
    The repercussions of flooding in Bangladesh give rise to pressing issues, encompassing the loss of lives, devastation of crops, infrastructure damage, 
    and displacement of communities. Accurate and timely flood prediction, along with waterbody forecasting, is paramount to mitigating the impact of floods. 
    It enhances disaster preparedness, allowing for more effective resource allocation.
    """)

st.write("""
    **Model Overview:**
    Data spanning from January 1, 2013, to August 17, 2023, served as the foundation for predicting daily average precipitation, the cumulative sum 
    of rainfall, and river discharge in Sylhet through the application of the XGBoost Regressor. The model's efficacy was assessed, yielding commendable
    results: an R2 score of 0.655, a mean squared error of 0.0442, and a mean absolute error of 0.0221.
    """)

st.write("""
    **Further Consideration:**
    Sylhet, situated on the borders of Bangladesh, is intricately connected to the dynamics of its surroundings. The likelihood of floods in this region is profoundly shaped by the atmospheric conditions 
    prevailing not only within its boundaries but also by the influences emanating from neighboring states in India. Recognizing the interconnectedness of environmental factors, expanding the geographical 
    scope to include data from these neighboring regions holds the potential to significantly enhance the accuracy of flood predictions.
    """)
    
    
target_variables = ['precip', 'rain_sum', 'river_discharge']
#@st.cache(suppress_st_warning=True)
def load_test_df():
    test_df = pd.read_csv('test.csv', index_col='datetime', parse_dates=True)
    return test_df

    
def get_date_range(df):
    start_date = df.index.min()
    end_date = df.index.max()
    start_end_date = [start_date, end_date]
    return start_end_date

def get_date(start_end_date):
    default_date = dt.date(2023, 8, 17)  # Use the alias here
    date = st.date_input("Please select a date between 2022 Jan 1 and 2023 Aug 17 to see the precipitation forecast:", default_date)
    if (date >= start_end_date[0].date()) & (date <= start_end_date[1].date()):
        return date
    else:
        st.write("No data available for the selected date.")

# get_data function
def get_data(df, date):
    date = pd.Timestamp(date)
    if date in df.index:
        weather_data = df[df.index == date]
        return weather_data
    else:
        st.write("The selected date is not in the DataFrame.")

# load model
def load_model(model_file):
    with open(model_file, 'rb') as file:
        model = joblib.load(file)
    return model 

def get_prediction(model, data):
    pred = model.predict(data)
    return pred


def create_individual_plot(data, y_test, y_pred, variable_name):

        
        # Check if y_pred is a 1D array and convert it to a DataFrame
        if len(y_pred.shape) == 1:
            y_pred_df = pd.DataFrame({
                'Datetime': data.index,
                f'Actual_{variable_name}': y_test,
                f'Predicted_{variable_name}': y_pred,
            })
        else:
            col_index = target_variables.index(variable_name)
            # Access the first column of y_pred
            y_pred_df = pd.DataFrame({
                'Datetime': data.index,
                f'Actual_{variable_name}': y_test,
                f'Predicted_{variable_name}': y_pred[:, col_index],
            })

        
        fig = go.Figure()

        # Add traces for actual and predicted values
        fig.add_trace(go.Scatter(x=y_pred_df['Datetime'], y=y_pred_df[f'Actual_{variable_name}'],
                                 mode='markers', name=f'Actual {variable_name}', marker=dict(color='blue', symbol='circle')))

        fig.add_trace(go.Scatter(x=y_pred_df['Datetime'], y=y_pred_df[f'Predicted_{variable_name}'],
                         mode='markers', name=f'Predicted {variable_name}',
                         marker=dict(color='rgba(0, 128, 128, 0.7)', symbol='triangle-up-open', size=10)))


        # Set the y-axis range to [0, 1]
        fig.update_layout(yaxis=dict(range=[0, 1]))

        # Set the layout
        fig.update_layout(title=f'Actual vs Predicted {variable_name} (for scaled data)',
                          xaxis=dict(title='Datetime'),
                          yaxis=dict(title=f'{variable_name}'))

        # Display the figure
        st.plotly_chart(fig)

# Main function to display the web app UI  
def main(): 
    
    # Set up directories
    #working_dir = os.path.dirname(os.path.realpath(__file__))
    #app_dir = os.path.dirname(working_dir)
    #ARTIFACTORY_DIR = os.path.join(app_dir, 'artifactory')
    MODEL_FILE = r"C:/Users/ThinkPad/Downloads/streamlit_sylhet/final_xgboost_sylhet1.joblib"
    
    #ACTUAL_DATASET = os.path.join(ARTIFACTORY_DIR, "central_actual.csv")

    # # Load datasets
    test = load_test_df()
    y_test = test[target_variables]

    # # Load the model
    model = load_model(MODEL_FILE)

    start_end_date = get_date_range(test)

    # Get date to predict
    date = get_date(start_end_date)
    
    # Convert date to the same format as the index in y_test
    date = pd.Timestamp(date)
        
    # # Get the values for prediction
    data = get_data(test, date)
    
   # Check if data is not None and contains the required columns
    if data is not None and all(col in data.columns for col in target_variables):
        data1 = data.drop(target_variables, axis=1)

        # Predict
        y_pred = get_prediction(model, data1)
        
        y_test_single_date = None

        # Extract y_test for the selected date
        try:
            y_test_single_date = y_test.loc[date]
        except KeyError:
            st.write(f"No data available for the selected date: {date}")

        predicted_numpy_array = np.array(y_pred)

        df_scaled = pd.DataFrame(predicted_numpy_array)

        df = df_scaled.rename(columns={0: "precipitation(mm)", 1: "rain sum(mm)", 2: "river discharge"})
        RD_max_train = 25.61
        RD_min_train = 0.85

        R_max_train = 235.80
        R_min_train = 0.0

        P_max_train = 200.00
        P_min_train = 0.0

        df['river discharge'] = df['river discharge'].apply(lambda x: x * (RD_max_train - RD_min_train) + RD_min_train)
        df['rain sum(mm)'] = df['rain sum(mm)'].apply(lambda x: x * (R_max_train - R_min_train) + R_min_train)
        df['precipitation(mm)'] = df['precipitation(mm)'].apply(lambda x: x * (P_max_train - P_min_train) + P_min_train)
        st.write("Predicted results: ", df)

        # Draw line charts
        for variable_name in target_variables:
            create_individual_plot(data1, y_test_single_date[variable_name], y_pred, variable_name)
    else:
        st.write("Cannot proceed with the prediction. Check if the required columns are present in the data.")
        # st.write(df)
        

if __name__ == '__main__':
    main()
