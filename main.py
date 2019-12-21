import json, pickle, math
import pandas as pd
import numpy as np
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
	df = df_old[['Open','High','Low', 'Close', 'Volume']]
	df['HL_PCT'] = (df['High'] - df['Low']) / df['Close'] * 100.0  # High/Low percent change (volatility)
	df['PCT_change'] = (df['Close'] - df['Open']) / df['Open'] * 100.0  # PCT Change (increase in day)
	df = df[['Close', 'HL_PCT', 'PCT_change', 'Volume']]  # Isolate important rows for training
	return df

def create_training_data(df, predictor_col, pct_train, day_shift):
	predictor_col = "Close"
	df.fillna(-99999, inplace=True)
	df['label'] = df[predictor_col].shift(-day_shift)

	# Prepare X training data
	X = np.array(df.drop(['label'], 1))  # Features (everything except labels)
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


#serialize_df("json/gldSymbol.json", "data")
df = create_df("data")
predictions, accuracy = create_training_data(df, "Close", 0.2, 8)

print(predictions)












