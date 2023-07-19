import streamlit as st
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Sidebar title
st.sidebar.header("Maven Cloud Limited")

# Allow users to upload a file via the sidebar
uploaded_file = st.sidebar.file_uploader("Upload Dataset", type=['csv', 'txt', 'xlsx'])

if uploaded_file is not None:
    # Read the uploaded file as a DataFrame
    df = pd.read_csv(uploaded_file)
    # Show the DataFrame contents
    st.subheader("Uploaded Dataset")
    st.write(df)
else:
    # If no file is uploaded, display a message and stop the application
    st.subheader("Kindly upload the Dataset using the sidebar to for this application to run.")
    st.stop()


# Add other contents of the main page here
st.subheader("Introductory Remarks")
st.write("Welcome to Maven Cloud Limited call Direction App! Here's some introductory text.")

st.title('A CALL DIRECTION PREDICTION APP FOR MAVEN CLOUD LIMITED')
st.write("""

This app predicts **The most likely call direction (whether incoming or outgoing calls)** at a Telecomunication Company

The Data is obtained from Maven Cloud Limited-Telecomunication company's call data 
""")
st.subheader('The entire Dataset')
st.write(df)

st.subheader('A general shape composition of the data')
st.markdown('Display data')
st.write(df.head())

# shape of data
if st.checkbox("Show shape and the corresponding Descriptive Statistics "):
    st.write('Data Shape')
    st.write(f'{df.shape[0]} rows and {df.shape[1]} columns')
    
    # data description
    st.markdown ("Descriptive statistics ")
    st.write(df.describe())

def main():
    st.title("Call Direction Visualization")
    st.write("This app displays the distribution of call directions.")
    
    # Display the visualization using seaborn's displot
    fig = sns.displot(df, x="call_direction", shrink=.5)
    
    # Display the plot using Streamlit
    st.subheader('Visualisation')
    st.pyplot(fig.fig)  # Use fig.fig to pass the Matplotlib figure to Streamlit

if __name__ == "__main__":
    main()

# But first let us drop insignificant features
# dropping the columns that are insignificant to my analysis
df = df.drop(columns=[ 'call_id', 'destination_person_id', 'source_person_id', 'occupation', 
                      'language', 'call_end_date', 'call_topics'])

#Let us encode the categorical features using LabelEncoder

le = LabelEncoder()

def encode(data_set, column):
    data_set[str(column)] = le.fit_transform(data_set[str(column)])
list_of_columns = ['account_age', 'account_state', 'call_direction', 'call_duration',
       'call_outcome', 'call_topic_group']

for item in list_of_columns:
    encode(df,item)
#We select the values for X and y sets 
X = df.drop(labels='call_direction', axis=1)
y = df.call_direction

#Splitting the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

# Create the DecisionTreeClassifier 
DT = DecisionTreeClassifier(random_state=0)

# Fitting Model
DT.fit(X_train, y_train)

# Getting Predictions
DT_pred = DT.predict(X_test)

# Model Accuracy, how often is the classifier correct
DT_ac = accuracy_score(y_test, DT_pred)

print(f'Accuracy Score: {DT_ac:.3f}\n')

st.subheader('Accuracy:')
accuracy = DT_ac
st.write(f'Accuracy: {accuracy * 100:.2f}%')

def get_f1_score(class_report, class_name):
    report_lines = class_report.split('\n')
    for line in report_lines:
        if line.strip().startswith(class_name):
            # Split by whitespace and filter out empty elements
            elements = [elem for elem in line.split() if elem]
            if len(elements) >= 4:
                f1_score = elements[3]
                if f1_score == 'None':
                    return None
                return float(f1_score)
    return None

def print_classification_report(report):
    report_lines = report.split('\n')
    headers = report_lines[0].split()
    data_lines = report_lines[2:-3]  # Exclude the headers and summary lines

    # Print the headers
    print(' '.join(headers))

    # Print the data lines
    for line in data_lines:
        print(line)

    return report_lines

# Generate the classification report
report = classification_report(y_test, DT_pred, target_names=["Incoming", "Outgoing"])

# Print the classification report to the console and get it as a list of lines
report_lines = print_classification_report(report)

# Calculate the F1-score for each class
incoming_f1_score = get_f1_score(report, "Incoming")
outgoing_f1_score = get_f1_score(report, "Outgoing")

# Check if both F1-scores are valid before making the comparison
if incoming_f1_score is not None and outgoing_f1_score is not None:
    # Determine the preferred response
    if incoming_f1_score > outgoing_f1_score:
        preferred_response = "You are likely to receive incoming calls than outgoing calls"
    else:
        preferred_response = "You are likely to receive outgoing calls than incoming calls"
else:
    # Handle the case when one or both F1-scores are missing
    preferred_response = "Unable to determine the preferred response due to missing F1-scores."

# Display the classification report
print("Classification Report:")
for line in report_lines:
    print(line)

# Display the results
print("F1-score for Incoming calls:", incoming_f1_score)
print("F1-score for Outgoing calls:", outgoing_f1_score)

# Display the preferred response in your streamlit app
st.subheader('Your Report:')
st.write(preferred_response)

# # Display Progress time 
import time
my_bar = st.progress (0)
for percent_complete in range (100):
    time.sleep(0.1)
    my_bar.progress(percent_complete + 1)

st.success("Kind regards")
st.success("Designed by Ezra Kikonyogo Kasirye")

st.balloons()


