import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from extended_pipeline import EstimatorAttributeMixin, ExtendedPipeline

hernia_optimum_threshold = 0.4322


class HierarchicalModel(EstimatorAttributeMixin):
    def __init__(
        self,
        spondy_pipeline: Pipeline,
        hernia_pipeline: Pipeline,
        hernia_threshold: float,
    ):
        self.spondy_pipeline = spondy_pipeline
        self.hernia_pipeline = hernia_pipeline
        self.hernia_threshold = hernia_threshold
        self._coef_ = self._compute_coef()
        self._intercept_ = self._compute_intercept_()


    def scale_spondy(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Scales the features used by the spondy pipeline.
        Returns a DataFrame with the scaled 'degree_spondylolisthesis' feature.
        """
        X_scaled = X.copy()
        spondy_scaler: StandardScaler = self.spondy_pipeline.named_steps['scaler']
        X_scaled["degree_spondylolisthesis"] = spondy_scaler.transform(
            X[["degree_spondylolisthesis"]]
        )
        return X_scaled

    def scale_hernia(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Scales the features used by the hernia pipeline.
        Returns a DataFrame with scaled features ['pelvic_tilt', 'lumbar_lordosis_angle', 'sacral_slope', 'pelvic_radius'].
        """
        X_scaled = X.copy()
        hernia_scaler: StandardScaler = self.hernia_pipeline.named_steps['scaler']
        feature_columns = [
            "pelvic_tilt",
            "lumbar_lordosis_angle",
            "sacral_slope",
            "pelvic_radius",
        ]
        X_scaled[feature_columns] = hernia_scaler.transform(X[feature_columns])
        return X_scaled

    def scale(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Scales the input data using the scalers in the spondy and hernia pipelines.
        This function calls both scale_spondy and scale_hernia.
        """
        # Scale for Spondy part
        X_scaled = self.scale_spondy(X)
        
        # Scale for Hernia part
        X_scaled = self.scale_hernia(X_scaled)

        return X_scaled

    def predict_proba(self, X_test: pd.DataFrame) -> pd.DataFrame:
        # Initialize a DataFrame to hold probabilities
        df_probabilities = X_test.reset_index(drop=True).copy()

        # Spondy vs Non-Spondy classification
        df_probabilities["class"] = self.spondy_pipeline.predict(
            X_test[["degree_spondylolisthesis"]]
        )
        df_probabilities["spondy_proba"] = self.spondy_pipeline.predict_proba(
            X_test[["degree_spondylolisthesis"]]
        )[:, 1]
        df_probabilities["hernia_proba"] = 0.0
        df_probabilities["normal_proba"] = 0.0

        # Non-Spondy bifurcation
        non_spondy_df = df_probabilities[df_probabilities["class"] == "Non-Spondy"].copy()
        feature_columns = [
            "pelvic_tilt",
            "lumbar_lordosis_angle",
            "sacral_slope",
            "pelvic_radius",
        ]
        features = non_spondy_df[feature_columns]

        if not features.empty:
            hernia_probs = self.hernia_pipeline.predict_proba(features)
            non_spondy_df["hernia_proba"], non_spondy_df["normal_proba"] = hernia_probs[:, 0], hernia_probs[:, 1]

        # Update probabilities for Non-Spondy samples
        df_probabilities.loc[
            df_probabilities["class"] == "Non-Spondy", "hernia_proba"
        ] = non_spondy_df["hernia_proba"]
        df_probabilities.loc[
            df_probabilities["class"] == "Non-Spondy", "normal_proba"
        ] = non_spondy_df["normal_proba"]
        # df_probabilities.loc[
        #     df_probabilities["class"] == "Non-Spondy", "spondy_proba"
        # ] = 1 - (non_spondy_df["hernia_proba"] + non_spondy_df["normal_proba"])
        # df_probabilities.loc[
        #     df_probabilities["class"] == "Spondylolisthesis", "spondy_proba"
        # ] = 1

        return df_probabilities

    def predict(self, X_test: pd.DataFrame) -> pd.DataFrame:
        # Get probabilities
        df_probabilities = self.predict_proba(X_test)

        # Determine final predictions
        df_probabilities["prediction"] = np.where(
            df_probabilities["hernia_proba"] >= self.hernia_threshold,
            "Hernia",
            "Normal",
        )
        df_probabilities["final_prediction"] = np.where(
            df_probabilities["class"] == "Non-Spondy",
            df_probabilities["prediction"],
            df_probabilities["class"],
        )
        return df_probabilities

    def save(self, file_path: str) -> None:
        joblib.dump(self, file_path)

    @staticmethod
    def load(file_path: str) -> Pipeline:
        return joblib.load(file_path)
    
    def _compute_coef(self):
        """
        Combine and retrieve the coef_ from both the spondy and hernia pipelines.
        This returns a single array combining the coefficients of both pipelines.
        """
        spondy_coef = ExtendedPipeline(self.spondy_pipeline).coef_
        hernia_coef = ExtendedPipeline(self.hernia_pipeline).coef_
        
        combined_coef = np.zeros((3, spondy_coef.shape[1] + hernia_coef.shape[1]))
        combined_coef[0, :spondy_coef.shape[1]] = spondy_coef.flatten()
        combined_coef[1, spondy_coef.shape[1]:] = hernia_coef.flatten()
        combined_coef[2, spondy_coef.shape[1]:] = hernia_coef.flatten()

        return combined_coef

    @property
    def coef_(self):
        """
        Return the precomputed combined coef_.
        """
        return self._coef_

    def _compute_intercept_(self):
        """
        Combine and retrieve the intercept_ from both the spondy and hernia pipelines.
        This returns a single intercept value for the hierarchical model.
        """
        # Get intercepts from both spondy and hernia estimators
        spondy_intercept = ExtendedPipeline(self.spondy_pipeline).intercept_
        hernia_intercept = ExtendedPipeline(self.hernia_pipeline).intercept_

        combined_intercept = np.zeros(3)
        combined_intercept[0] = spondy_intercept
        combined_intercept[1] = hernia_intercept
        combined_intercept[2] = hernia_intercept

        return combined_intercept
    
    @property
    def intercept_(self):
        """
        Return the precomputed combined intercept_.
        """
        return self._intercept_
