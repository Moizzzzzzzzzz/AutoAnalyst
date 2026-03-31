import logging
import pandas as pd
import numpy as np
from scipy import stats
from typing import Tuple, Dict, Any, List, Optional

logger = logging.getLogger(__name__)

def _get_plain_english_conclusion(test_name: str, p_value: float, var1: str, var2: str, alpha: float = 0.05) -> str:
    """Translates statistical results into readable business conclusions."""
    if p_value < alpha:
        return f"Statistically significant relationship found between {var1} and {var2} (p < {alpha})."
    else:
        return f"No significant relationship detected between {var1} and {var2} (p >= {alpha})."

def _test_categorical_vs_numeric(df: pd.DataFrame, cat_col: str, num_col: str) -> Optional[Dict[str, Any]]:
    """Runs T-test or ANOVA depending on the number of groups in the categorical column."""
    groups = df.groupby(cat_col)[num_col].apply(list).to_dict()
    valid_groups = {k: v for k, v in groups.items() if len(v) >= 5} # Require minimum sample size per group
    
    if len(valid_groups) < 2:
        return None

    group_arrays = list(valid_groups.values())
    
    try:
        if len(group_arrays) == 2:
            # 2 groups -> Independent T-test
            stat, p_val = stats.ttest_ind(group_arrays[0], group_arrays[1], nan_policy='omit')
            test_name = "Independent T-Test"
        else:
            # 3+ groups -> One-way ANOVA
            stat, p_val = stats.f_oneway(*group_arrays)
            test_name = "One-Way ANOVA"
            
        return {
            "hypothesis": f"Does {num_col} differ across groups of {cat_col}?",
            "test_used": test_name,
            "statistic": float(stat),
            "p_value": float(p_val),
            "conclusion": _get_plain_english_conclusion(test_name, p_val, cat_col, num_col)
        }
    except Exception as e:
        logger.debug(f"Failed to run {cat_col} vs {num_col}: {str(e)}")
        return None

def _test_categorical_vs_categorical(df: pd.DataFrame, col1: str, col2: str) -> Optional[Dict[str, Any]]:
    """Runs Chi-Square Test of Independence for two categorical variables."""
    # Prevent massive cross-tabulations that crash memory
    if df[col1].nunique() > 10 or df[col2].nunique() > 10:
        return None
        
    contingency_table = pd.crosstab(df[col1], df[col2])
    
    # Chi-square requires expected frequencies > 5 in mostly all cells, but we run it as a heuristic
    try:
        stat, p_val, dof, expected = stats.chi2_contingency(contingency_table)
        return {
            "hypothesis": f"Are {col1} and {col2} independent?",
            "test_used": "Chi-Square Test of Independence",
            "statistic": float(stat),
            "p_value": float(p_val),
            "conclusion": _get_plain_english_conclusion("Chi-Square", p_val, col1, col2)
        }
    except Exception as e:
        logger.debug(f"Failed to run Chi-Square for {col1} vs {col2}: {str(e)}")
        return None

def _test_numeric_vs_numeric(df: pd.DataFrame, col1: str, col2: str) -> Optional[Dict[str, Any]]:
    """Runs Pearson correlation test for two continuous variables."""
    clean_df = df[[col1, col2]].dropna()
    if len(clean_df) < 10:
        return None
        
    try:
        stat, p_val = stats.pearsonr(clean_df[col1], clean_df[col2])
        return {
            "hypothesis": f"Is there a linear correlation between {col1} and {col2}?",
            "test_used": "Pearson Correlation",
            "statistic": float(stat),
            "p_value": float(p_val),
            "conclusion": _get_plain_english_conclusion("Pearson Correlation", p_val, col1, col2)
        }
    except Exception as e:
        logger.debug(f"Failed to run Pearson for {col1} vs {col2}: {str(e)}")
        return None

def generate_auto_hypotheses(df: pd.DataFrame, max_tests: int = 5) -> List[Dict[str, Any]]:
    """
    Autonomously scans the dataframe to find the most interesting variable pairs to test.
    Prioritizes columns with higher variance to avoid testing static data.
    """
    results = []
    
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()
    
    # Filter out IDs or heavily missing columns
    num_cols = [c for c in num_cols if df[c].nunique() > 5 and df[c].isnull().mean() < 0.3][:5]
    cat_cols = [c for c in cat_cols if 1 < df[c].nunique() < 10][:5]

    # 1. Test Cat vs Num (ANOVA/T-test)
    if cat_cols and num_cols:
        res = _test_categorical_vs_numeric(df, cat_cols[0], num_cols[0])
        if res: results.append(res)
            
    # 2. Test Num vs Num (Pearson)
    if len(num_cols) >= 2:
        res = _test_numeric_vs_numeric(df, num_cols[0], num_cols[1])
        if res: results.append(res)
            
    # 3. Test Cat vs Cat (Chi-Square)
    if len(cat_cols) >= 2:
        res = _test_categorical_vs_categorical(df, cat_cols[0], cat_cols[1])
        if res: results.append(res)

    return results[:max_tests]

def run_stats(dataset_path: str) -> Tuple[Dict[str, Any], str]:
    """
    Main entry point for the Statistical Testing agent.
    Returns results dictionary and a reasoning log.
    """
    logger.info(f"Starting Stats Agent on {dataset_path}")
    
    try:
        # File ingestion (assuming valid path passed by Orchestrator)
        if dataset_path.endswith('.csv'):
            df = pd.read_csv(dataset_path)
        elif dataset_path.endswith(('.xls', '.xlsx')):
            df = pd.read_excel(dataset_path)
        else:
            raise ValueError("Unsupported file format for Stats Agent.")

        # In a fully connected graph, we might receive specific hypotheses from the LLM Planner.
        # For autonomous fallback, we generate our own.
        hypotheses_results = generate_auto_hypotheses(df)
        
        significant_findings = sum(1 for h in hypotheses_results if h['p_value'] < 0.05)
        
        results = {
            "tests_run": len(hypotheses_results),
            "significant_findings": significant_findings,
            "detailed_results": hypotheses_results
        }
        
        log_msg = f"Ran {len(hypotheses_results)} statistical tests. Found {significant_findings} significant relationships."
        return results, log_msg

    except Exception as e:
        logger.error(f"Stats Pipeline failed: {str(e)}")
        raise e