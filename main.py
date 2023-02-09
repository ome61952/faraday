# -*- coding: utf-8 -*-
"""
Created on Thu Feb  9 18:12:48 2023

@author: Ankit
"""
import pandas as pd
import sqlite3 as sql
from statsmodels.tsa.arima.model import ARIMA

class ARIMAForecaster:
    def __init__(self, p: int = 1, d: int = 1, q: int = 0):
        """
        Initializer for ARIMAForecater

        Parameters
        ----------
        p : int, optional
            Lag parameter for ARIMA model. The default is 1.
        d : int, optional
            Difference parameter for ARIMA model. The default is 1.
        q : int, optional
            Window size parameter for ARIMA model. The default is 0.

        Returns
        -------
        None.

        """
        self._p = p
        self._d = d
        self._q = q
        self.conn = sql.connect("faraday.db")
        self.model = None # Trained model
        self._n = None # number of training samples
        
    def load_history(self, filepath: str):
        """
        loads the historical data csv file to sqlite table

        Parameters
        ----------
        filepath : str
            filepath of the censor csv data.
        """
        history = pd.read_csv(filepath)
        history.to_sql("voltage", self.conn, if_exists="replace")
        
    def train(self):
        """
        Trains an ARIMA forecasting model on historical data 

        Returns
        -------
        None.

        """
                
        train  = pd.read_sql("SELECT * from voltage;", self.conn)["voltage"] # fetch the voltage column from table
        self._n = len(train) # sample size of training
        m  = ARIMA(train, order=(self._p, self._d, self._q))
        self.model = m.fit()
        
    def _infer(self, iterations: int = 1):
        """
        Forecast future for n iterations 

        Parameters
        ----------
        iterations : int, optional
            Number of predictions in future. The default is 100.

        Returns
        -------
            list of predictions.

        """
        if not self.model:
            raise Exception("ERROR: Model not trained yet")

        start_period = self._n
        end_period = start_period + iterations

        return self.model.predict(start_period, end_period)

    
    def predict(self, iterations: int = 1):
        """
        Run model inference and store predictions to 'voltage_pred' table
        """
        try:
            n_pred = self._infer(iterations)       
            n_pred.to_sql("voltage_pred", self.conn, if_exists="append", index=True)    
        except Exception as ex:
            print("Unable to save to Sqlite")
            raise ex