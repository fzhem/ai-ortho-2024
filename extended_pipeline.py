## wrapper for pipeline (to be used for RFE and SHAP)
import sys
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.pipeline import Pipeline
from sklearn.utils.validation import check_is_fitted

if sys.version_info >= (3, 11):
    from typing import Self
else:
    from typing_extensions import Self


class EstimatorAttributeMixin:
    """
    A mixin to provide functionality for accessing attributes from the final estimator in a pipeline.
    """
    def _get_estimator_attribute(self, attr_name: str):
        """
        Retrieve an attribute from the final estimator in the pipeline.

        Args:
            attr_name (str): The attribute name to retrieve.

        Returns:
            The value of the attribute if it exists.

        Raises:
            AttributeError: If the attribute does not exist on the final estimator.
            ValueError: If the pipeline is not fitted before accessing the attribute.
        """
        if not hasattr(self, 'pipeline') or not hasattr(self, 'estimator_name'):
            raise AttributeError(
                "The mixin requires 'pipeline' and 'estimator_name' attributes to be defined."
            )
        
        estimator = self.pipeline.named_steps[self.estimator_name]
        check_is_fitted(estimator)
        if hasattr(estimator, attr_name):
            return getattr(estimator, attr_name)
        raise AttributeError(
            f"{self.estimator_name} does not have attribute `{attr_name}`."
        )


class ExtendedPipeline(BaseEstimator, ClassifierMixin, EstimatorAttributeMixin):
    def __init__(self, pipeline: Pipeline, estimator_name: str = "logreg"):
        """
        A wrapper to make a pipeline compatible with RFE by exposing the final estimator.

        Args:
            pipeline (Pipeline): A pipeline with preprocessing and a final estimator.
            estimator_name (str): The name of the final estimator step in the pipeline.
        """
        self.pipeline = pipeline
        self.estimator_name = estimator_name

    def fit(
        self,
        X: list[str] | np.ndarray | Iterable | pd.DataFrame,
        y: list[int] | Iterable | pd.Series | np.ndarray | None,
    ) -> Self:
        self.pipeline.fit(X, y)
        return self

    def predict(
        self, X: list[str] | np.ndarray | Iterable | pd.DataFrame
    ) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
        return self.pipeline.predict(X)

    def predict_proba(
        self, X: list[str] | np.ndarray | Iterable | pd.DataFrame
    ) -> np.ndarray:
        return self.pipeline.predict_proba(X)

    @property
    def coef_(self):
        return self._get_estimator_attribute("coef_")

    @property
    def feature_importances_(self):
        return self._get_estimator_attribute("feature_importances_")

    @property
    def classes_(self):
        return self._get_estimator_attribute("classes_")

    @property
    def intercept_(self):
        return self._get_estimator_attribute("intercept_")