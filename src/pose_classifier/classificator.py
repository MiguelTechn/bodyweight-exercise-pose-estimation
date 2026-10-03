import joblib
import pandas as pd
from pathlib import Path

class ExcersisesClassifier:
    """
    Classifies the exercise using a pretrained machine learning model (Random Forest).

    The purpose of this class is to predict an element based on a data entry. As this class 
    works with a Random Forest model but it has to understand data based on time (windows of a video streaming)
    before predict, the class transforms 60 video frames in a single statistics data row.
    """
    def __init__(self, classifier_path: str | Path = Path(__file__).parent.parent.parent / 'models' / 'rf_classifier' / 'rf_model.joblib'):
        """
        Initializes the ExcersisesClassifier.
        Args:
            classifier_path (str | Path, optional): _description_. Defaults to Path(__file__).parent.parent.parent/'models'/'rf_classifier'/'rf_model.joblib': 
                Machine Learning model (predictor)

        Raises:
            Exception: No classifier model file found.
        """
        try:
            self.model = joblib.load(classifier_path)
            self.feature_names_in = self.model.feature_names_in_
            print("Clasifier iniciated")
        except FileNotFoundError:
            raise Exception(f"No path to classifier found: {classifier_path}") 
    
    def predict(self, X):
        """
        Transforms multiple data rows into a single data row and make a prediction giving a class.

        Args:
            X (_type_): Data used to predict. It contains 60 video frames wich has been proseced.

        Returns:
            numpy.ndarray: Class prediction.
        """
        X_selected = self._select_cols(X)
        y_pred = self.model.predict(X_selected)
        return y_pred

    def _select_cols (self, X):
        """
        Combine the next method _data_flattening() to just select the columns needed by the model.

        Args:
            X (_type_): Data used to predict. It contains 60 video frames wich has been proseced.

        Returns:
            numpy.ndarray: Array with data flattened and filtered.
        """
        X_planned = self._data_flattening(X)
        return X_planned[self.feature_names_in]
    
    def _data_flattening(self, X):
        """_summary_

        Args:
            X (_type_): _description_

        Returns:
            numpy.ndarray: Array with data flattened.
        """
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        #Agregation dict
        agg_dict = {col: ['mean', 'std', 'min', 'max', 'median'] for col in X.columns} #Agg features to group windows

        # Features metrics calculation
        df_grouped = X.agg(agg_dict).unstack().to_frame().T

        # Var flatenned
        df_grouped.columns = [f"{col}_{stat}" for col, stat in df_grouped.columns]

        return df_grouped
    
