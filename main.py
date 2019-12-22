import json, pickle, math, os, time, datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as md
from datetime import datetime, timedelta
from matplotlib.dates import DateFormatter
from matplotlib import style
from sklearn import preprocessing, svm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

def serialize_df(filename, output):  # Create and serialize dataframe using pickle

	# Create empty dataframe with column names
	df = pd.DataFrame(columns=['Date', 'Open', 'High', 'Low', 'Close', 'Volume'])

	with open(filename) as json_file:
		data = json.load(json_file) # Open and load json file
		for day in data:  # Loop through days in the data

			volume = int(day['Volume'].replace(',',''))  # Remove commas and convert volume to int

			new_row = {'Date':day['Date'],
					   'Open':day['Open'],
					   'High':day['High'],
					   'Low':day['Low'],
					   'Close':day['Close'],
					   'Volume':volume
					   }
			df = df.append(new_row, ignore_index=True)  # Append new row to dataframe

	df = df.reindex(index=df.index[::-1])  # Flip data old to new (for graph)
	df.to_pickle(output)  # Serialize data into dataframe to speed up parsing speeds


def create_df(filename):  # Parse pickled dataframe to features of dataset
	df_old = pickle.load(open(filename, 'rb'))
	df = df_old[['Date','Open','High','Low', 'Close', 'Volume']]
	df['HL_PCT'] = (df['High'] - df['Low']) / df['Close'] * 100.0  # High/Low percent change (volatility)
	df['PCT_change'] = (df['Close'] - df['Open']) / df['Open'] * 100.0  # PCT Change (increase in day)
	df = df[['Date','Close', 'HL_PCT', 'PCT_change', 'Volume']]  # Isolate important rows for training
	return df

def create_training_data(df, predictor_col, pct_train, day_shift):
	predictor_col = "Close"
	df.fillna(-99999, inplace=True)
	df['label'] = df[predictor_col].shift(-day_shift)

	# Prepare X training data
	X = np.array(df.drop(['label', 'Date'], 1))  # Features (everything except labels)
	X = preprocessing.scale(X)  # Scale and Normalize x values
	X = X[:-day_shift]  # Rest of data (not shifted)
	X_predict = X[-day_shift:]  # Empty columns to predict

	df.dropna(inplace=True)  # Remove NaN values
	y = np.array(df['label'])  # Create y values for training

	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=pct_train) # Use x% of data as testing data
	model = LinearRegression(n_jobs=-1)
	model.fit(X_train, y_train)
	accuracy = model.score(X_test, y_test)
	predictions = model.predict(X_predict)
	return predictions, accuracy


def plot_results(df, predictions):

	plt.style.use('seaborn-notebook')  # Set style scheme of graph

	# Calculate the dates of the predicted values
	last_day = df.iloc[-1]['Date']  # Get most recent day
	last_day = datetime.strptime(last_day, "%m/%d/%Y")  # Format last day to use delta change
	next_day = last_day + timedelta(days=1)  # Get next datetime
	prediction_days = []  # Empty array to store x values for graph

	for i in predictions:  #  Loop through all the predictions
		formatted_date = datetime.strftime(next_day, "%m/%d/%Y")  # Format the date to be mapped
		prediction_days.append(formatted_date)  # Append the date to the x values
		next_day = next_day + timedelta(days=1)  # Increment day by 1
	
	actual_dates = list(map(datetime.strptime, df['Date'], len(df['Date'])*['%m/%d/%Y']))
	predicted_dates = list(map(datetime.strptime, prediction_days, len(prediction_days)*['%m/%d/%Y']))

	formatter = md.DateFormatter('%Y-%m-%d')  # Format date string for axis

	plt.xlabel('Date', fontsize=16)
	plt.ylabel('Closing Price ($)', fontsize=16)

	plt.title('Date vs. Closing Price of GLD', fontsize=20)
	plt.plot(actual_dates, df['Close'], '-', label="Historic Price")  # Plot the values as two seperate lines (Time vs Price)
	plt.plot(predicted_dates, predictions, '-', label="Predicted Price")
	plt.legend(loc="upper right")

	ax = plt.gcf().axes[0]  # Define axis for format
	ax.xaxis.set_major_formatter(formatter)  # Apply date formatter to axis
	plt.gcf().autofmt_xdate(rotation=25)  # Auto fit dates to axis (no overlap)
	plt.show()



if __name__ == "__main__":

	outputFile = "data"  # Name of output serialize file
	dataPath = "json/gldSymbol.json"  # Path to json file for data parsing

	if not os.path.isfile(outputFile):  # If the file does not exist
		print("Serializing dataframe!")
		serialize_df(dataPath, outputFile)  # Create serialized dataframe

	df = create_df("data")  # Format dataframe
	predictions, accuracy = create_training_data(df, "Close", 0.2, 5)  # Get predictions
	plot_results(df, predictions)
		













