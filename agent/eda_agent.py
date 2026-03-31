import logging
import pandas as pd
import numpy as np
from scipy import stats
from sklearn.ensemble import IsolationForest
from statsmodels.stats.outliers_influence import variance_inflation_factor
from typing import Tuple, Dict, Any, List

logger = logging.getLogger(__name__)

def _analyze_missingness(df: pd.DataFrame) -> Dict[str, Any]:
    """Analyzes missing values and calculates missingness correlations (Proxy for MAR)."""
    missing_stats = df.isnull().sum()
    missing_cols = missing_stats[missing_stats > 0].index.tolist()
    
    result = {
        "total_missing_cells": int(missing_stats.sum()),
        "columns_with_missing": missing_cols,
        "missingness_correlation": None
    }
    
    # If multiple columns have missing data, check if missingness is correlated (MAR indicator)
    if len(missing_cols) > 1:
        missing_indicators = df[missing_cols].isnull().astype(int)
        corr_matrix = missing_indicators.corr().fillna(0).to_dict()
        result["missingness_correlation"] = corr_matrix
        
    return result

def _detect_outliers_advanced(series: pd.Series, df: pd.DataFrame, col_name: str) -> Dict[str, int]:
    """Detects outliers using IQR, Z-Score, and Isolation Forest."""
    clean_series = series.dropna()
    if len(clean_series) < 50:
        return {"iqr": 0, "z_score": 0, "isolation_forest": 0}

    # 1. IQR Method
    Q1, Q3 = clean_series.quantile(0.25), clean_series.quantile(0.75)
    IQR = Q3 - Q1
    iqr_outliers = int(((clean_series < (Q1 - 1.5 * IQR)) | (clean_series > (Q3 + 1.5 * IQR))).sum())

    # 2. Z-Score Method
    z_scores = np.abs(stats.zscore(clean_series))
    z_outliers = int((z_scores > 3).sum())

    # 3. Isolation Forest (using the single column reshaped)
    iso_forest = IsolationForest(contamination='auto', random_state=42)
    preds = iso_forest.fit_predict(clean_series.values.reshape(-1, 1))
    if_outliers = int((preds == -1).sum())

    return {
        "iqr": iqr_outliers,
        "z_score": z_outliers,
        "isolation_forest": if_outliers
    }

def _calculate_vif(df: pd.DataFrame, numeric_cols: List[str]) -> Dict[str, float]:
    """Calculates Variance Inflation Factor to check multicollinearity."""
    # VIF requires dropping NaNs and needs at least 2 variables
    clean_df = df[numeric_cols].dropna()
    if len(clean_df) < 50 or len(numeric_cols) < 2:
        return {}
        
    vif_data = {}
    # Cap the features to prevent massive compute times on wide datasets
    cols_to_check = numeric_cols[:20] 
    
    for i, col in enumerate(cols_to_check):
        try:
            vif = variance_inflation_factor(clean_df[cols_to_check].values, i)
            # Replace infinities with a high arbitrary number for JSON serialization
            vif_data[col] = float(vif) if not np.isinf(vif) else 999.0
        except Exception as e:
            logger.debug(f"Could not calculate VIF for {col}: {e}")
            
    return vif_data

def _detect_time_series(df: pd.DataFrame) -> List[str]:
    """Attempts to identify datetime columns or string columns that represent dates."""
    ts_cols = []
    for col in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            ts_cols.append(col)
        elif df[col].dtype == 'object':
            # Sample 100 non-null rows to test for date parsing
            sample = df[col].dropna().head(100)
            if not sample.empty:
                try:
                    pd.to_datetime(sample, format='mixed', errors='raise')
                    ts_cols.append(col)
                except (ValueError, TypeError):
                    continue
    return ts_cols

def _generate_narrative(results: Dict[str, Any]) -> List[str]:
    """Auto-generates 5-10 key observations based on the calculated stats."""
    observations = []
    
    # 1. Dataset size
    observations.append(f"The dataset contains {results['row_count']} rows and {results['column_count']} columns.")
    
    # 2. Missing data
    missing_cols = results.get("missingness", {}).get("columns_with_missing", [])
    if missing_cols:
        observations.append(f"Missing values detected in {len(missing_cols)} columns. Requires imputation before modeling.")
    else:
        observations.append("No missing values detected in the dataset.")
        
    # 3. Time Series
    ts_cols = results.get("time_series_columns", [])
    if ts_cols:
        observations.append(f"Detected potential time-series data in columns: {', '.join(ts_cols[:3])}.")
        
    # 4. Multicollinearity
    vif_data = results.get("multicollinearity_vif", {})
    high_vif = [col for col, val in vif_data.items() if val > 10.0]
    if high_vif:
        observations.append(f"High multicollinearity (VIF > 10) detected in: {', '.join(high_vif[:3])}. Consider dropping highly correlated features.")
        
    # 5. Outliers
    outlier_heavy = []
    for col, data in results.get("columns", {}).items():
        if "outliers" in data and data["outliers"].get("isolation_forest", 0) > (results["row_count"] * 0.05):
            outlier_heavy.append(col)
    
    if outlier_heavy:
        observations.append(f"Significant outliers (>5% of data) detected in: {', '.join(outlier_heavy[:3])}.")
        
    return observations

def run_eda(dataset_path: str) -> Tuple[Dict[str, Any], str]:
    """Main entry point for the autonomous EDA agent."""
    logger.info(f"Starting advanced EDA on {dataset_path}")
    
    try:
        if dataset_path.endswith('.csv'):
            df = pd.read_csv(dataset_path)
        elif dataset_path.endswith(('.xls', '.xlsx')):
            df = pd.read_excel(dataset_path)
        elif dataset_path.endswith('.json'):
            df = pd.read_json(dataset_path)
        else:
            raise ValueError("Unsupported file format.")
            
        if df.empty:
            raise ValueError("Dataset is empty.")

        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        results = {
            "row_count": len(df),
            "column_count": len(df.columns),
            "memory_usage_mb": float(df.memory_usage(deep=True).sum() / (1024 ** 2)),
            "missingness": _analyze_missingness(df),
            "time_series_columns": _detect_time_series(df),
            "multicollinearity_vif": _calculate_vif(df, numeric_cols),
            "columns": {}
        }

        for col in df.columns:
            series = df[col]
            missing_count = int(series.isnull().sum())
            
            col_data = {
                "dtype": str(series.dtype),
                "missing_percentage": float((missing_count / len(df)) * 100),
                "unique_values": int(series.nunique())
            }

            if col in numeric_cols:
                col_data.update({
                    "mean": float(series.mean()) if not pd.isna(series.mean()) else None,
                    "std_dev": float(series.std()) if not pd.isna(series.std()) else None,
                    "skewness": float(series.skew()) if not pd.isna(series.skew()) else None,
                    "kurtosis": float(series.kurtosis()) if not pd.isna(series.kurtosis()) else None,
                    "outliers": _detect_outliers_advanced(series, df, col)
                })
            else:
                top_values = series.value_counts().head(5).to_dict()
                col_data["top_values"] = {str(k): int(v) for k, v in top_values.items()}
                
            results["columns"][col] = col_data

        # Generate narrative
        results["narrative_observations"] = _generate_narrative(results)

        log_msg = f"Completed EDA. Generated {len(results['narrative_observations'])} key observations."
        return results, log_msg

    except Exception as e:
        logger.error(f"EDA Pipeline failed: {str(e)}")
        raise e
