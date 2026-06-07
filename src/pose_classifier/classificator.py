import joblib
import pandas as pd
from pathlib import Path

class ExcersisesClassifier:
    def __init__(self, classifier_path: str | Path = Path(__file__).parent.parent.parent / 'models' / 'rf_classifier' / 'rf_model.joblib'):
        try:
            self.model = joblib.load(classifier_path)
            self.feature_names_in = self.model.feature_names_in_
            print("Clasifier iniciated")
        except FileNotFoundError:
            raise Exception(f"No path to classifier found: {classifier_path}") 
    
    
    
    def predict(self, X):
        X_selected = self._select_cols(X)
        y_pred = self.model.predict(X_selected)
        return y_pred

    def _select_cols (self, X):
        X_planned = self._data_planning(X)
        return X_planned[self.feature_names_in]
    
    def _data_planning(self, X):
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)


        #Agregation dict
        agg_dict = {col: ['mean', 'std', 'min', 'max', 'median'] for col in X.columns} #Agg features to group windows

        # Features metrics calculation
        df_grouped = X.agg(agg_dict).unstack().to_frame().T

        # Var flatenned
        df_grouped.columns = [f"{col}_{stat}" for col, stat in df_grouped.columns]

        return df_grouped
    
