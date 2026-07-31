"""
Tools module for Agentic BI System
Each tool is a function that agents can call to perform specific tasks.
"""
import pandas as pd
import json
import os

# ========================
# DATA PATH
# ========================
DATA_PATH = "stripe_payments_dataset.csv"

# ========================
# TOOL IMPLEMENTATIONS
# ========================

def read_csv_data(file_path: str = None, sample_rows: int = 5) -> dict:
    """
    Read CSV data and return sample records with metadata.
    
    Args:
        file_path: Ignored - always uses stripe_payments_dataset.csv
        sample_rows: Number of sample rows to return (default: 5, max: 10)
    
    Returns:
        Dictionary with sample data, columns, and row count
    """
    # Always use the correct data path, ignore file_path parameter
    path = DATA_PATH
    try:
        df = pd.read_csv(path)
        # Limit sample size to avoid token overflow
        sample_size = min(sample_rows or 5, 10)
        sample_df = df.head(sample_size)
        return {
            "success": True,
            "columns": list(df.columns),
            "total_row_count": len(df),
            "sample_rows_returned": sample_size,
            "sample_data": sample_df.to_dict(orient="records")
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


def get_data_summary(file_path: str = None) -> dict:
    """
    Get statistical summary and metadata of the dataset.
    
    Returns:
        Dictionary with data types, missing values, and statistics
    """
    # Always use the correct data path, ignore file_path parameter
    path = DATA_PATH
    try:
        df = pd.read_csv(path)
        
        # Basic info
        summary = {
            "success": True,
            "shape": {"rows": len(df), "columns": len(df.columns)},
            "columns": list(df.columns),
            "dtypes": df.dtypes.astype(str).to_dict(),
            "missing_values": df.isnull().sum().to_dict(),
            "numeric_summary": {},
            "categorical_summary": {}
        }
        
        # Numeric columns summary
        numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
        for col in numeric_cols:
            summary["numeric_summary"][col] = {
                "min": float(df[col].min()),
                "max": float(df[col].max()),
                "mean": float(df[col].mean()),
                "median": float(df[col].median()),
                "std": float(df[col].std())
            }
        
        # Categorical columns summary
        cat_cols = df.select_dtypes(include=['object']).columns
        for col in cat_cols:
            value_counts = df[col].value_counts().head(10).to_dict()
            summary["categorical_summary"][col] = {
                "unique_count": int(df[col].nunique()),
                "top_values": value_counts
            }
        
        return summary
    except Exception as e:
        return {"success": False, "error": str(e)}


def execute_analysis(analysis_type: str, group_by: str = None, metric: str = None) -> dict:
    """
    Execute predefined analysis on the payment data.
    
    Args:
        analysis_type: Type of analysis ('revenue_by_country', 'revenue_by_segment', 
                      'revenue_by_channel', 'status_distribution', 'time_series', 'custom')
        group_by: Column to group by (for custom analysis)
        metric: Column to aggregate (for custom analysis)
    
    Returns:
        Analysis results as dictionary
    """
    try:
        df = pd.read_csv(DATA_PATH)
        
        # Convert amount from cents to euros
        df['amount_eur'] = df['amount'] / 100
        
        if analysis_type == "revenue_by_country":
            result = df.groupby('metadata.country')['amount_eur'].agg(['sum', 'count', 'mean']).reset_index()
            result.columns = ['country', 'total_revenue', 'transaction_count', 'avg_transaction']
            return {"success": True, "analysis": analysis_type, "data": result.to_dict(orient="records")}
        
        elif analysis_type == "revenue_by_segment":
            result = df.groupby('metadata.segment')['amount_eur'].agg(['sum', 'count', 'mean']).reset_index()
            result.columns = ['segment', 'total_revenue', 'transaction_count', 'avg_transaction']
            return {"success": True, "analysis": analysis_type, "data": result.to_dict(orient="records")}
        
        elif analysis_type == "revenue_by_channel":
            result = df.groupby('metadata.channel')['amount_eur'].agg(['sum', 'count', 'mean']).reset_index()
            result.columns = ['channel', 'total_revenue', 'transaction_count', 'avg_transaction']
            return {"success": True, "analysis": analysis_type, "data": result.to_dict(orient="records")}
        
        elif analysis_type == "status_distribution":
            result = df.groupby('status').agg({
                'amount_eur': ['sum', 'count'],
                'id': 'count'
            }).reset_index()
            result.columns = ['status', 'total_amount', 'count', 'transactions']
            return {"success": True, "analysis": analysis_type, "data": result.to_dict(orient="records")}
        
        elif analysis_type == "segment_channel_matrix":
            result = pd.pivot_table(df, values='amount_eur', 
                                   index='metadata.segment', 
                                   columns='metadata.channel', 
                                   aggfunc='sum', fill_value=0)
            return {"success": True, "analysis": analysis_type, "data": result.to_dict()}
        
        elif analysis_type == "custom" and group_by and metric:
            if group_by in df.columns and metric in df.columns:
                result = df.groupby(group_by)[metric].agg(['sum', 'count', 'mean']).reset_index()
                return {"success": True, "analysis": "custom", "data": result.to_dict(orient="records")}
        
        return {"success": False, "error": f"Unknown analysis type: {analysis_type}"}
    
    except Exception as e:
        return {"success": False, "error": str(e)}


def calculate_kpis() -> dict:
    """
    Calculate key performance indicators from the payment data.
    
    Returns:
        Dictionary with KPIs
    """
    try:
        df = pd.read_csv(DATA_PATH)
        df['amount_eur'] = df['amount'] / 100
        
        total_revenue = df['amount_eur'].sum()
        total_transactions = len(df)
        successful_transactions = len(df[df['status'] == 'succeeded'])
        
        kpis = {
            "success": True,
            "kpis": {
                "total_revenue_eur": round(total_revenue, 2),
                "total_transactions": total_transactions,
                "successful_transactions": successful_transactions,
                "success_rate": round(successful_transactions / total_transactions * 100, 2),
                "average_transaction_value": round(total_revenue / total_transactions, 2),
                "revenue_by_segment": df.groupby('metadata.segment')['amount_eur'].sum().to_dict(),
                "top_country": df.groupby('metadata.country')['amount_eur'].sum().idxmax(),
                "top_channel": df.groupby('metadata.channel')['amount_eur'].sum().idxmax(),
            }
        }
        return kpis
    except Exception as e:
        return {"success": False, "error": str(e)}


def detect_anomalies(threshold_std: float = 2.0) -> dict:
    """
    Detect anomalies in transaction amounts using statistical methods.
    
    Args:
        threshold_std: Number of standard deviations for anomaly threshold
    
    Returns:
        Dictionary with anomaly details
    """
    try:
        df = pd.read_csv(DATA_PATH)
        df['amount_eur'] = df['amount'] / 100
        
        mean_amount = df['amount_eur'].mean()
        std_amount = df['amount_eur'].std()
        
        # Identify anomalies
        upper_bound = mean_amount + (threshold_std * std_amount)
        lower_bound = mean_amount - (threshold_std * std_amount)
        
        anomalies = df[(df['amount_eur'] > upper_bound) | (df['amount_eur'] < lower_bound)]
        
        return {
            "success": True,
            "statistics": {
                "mean_amount": round(mean_amount, 2),
                "std_amount": round(std_amount, 2),
                "upper_bound": round(upper_bound, 2),
                "lower_bound": round(lower_bound, 2)
            },
            "anomaly_count": len(anomalies),
            "anomalies": anomalies[['id', 'amount_eur', 'metadata.country', 'metadata.segment', 'status']].to_dict(orient="records")
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


def generate_report_data() -> dict:
    """
    Generate comprehensive data for business report.
    
    Returns:
        Dictionary with all report data
    """
    try:
        df = pd.read_csv(DATA_PATH)
        df['amount_eur'] = df['amount'] / 100
        
        # Segment performance - flatten to avoid tuple keys
        segment_agg = df.groupby('metadata.segment')['amount_eur'].agg(['sum', 'mean', 'count']).reset_index()
        segment_perf = {}
        for _, row in segment_agg.iterrows():
            segment_perf[row['metadata.segment']] = {
                "total_revenue": round(row['sum'], 2),
                "avg_transaction": round(row['mean'], 2),
                "transaction_count": int(row['count'])
            }
        
        # Geographic performance
        geo_agg = df.groupby('metadata.country')['amount_eur'].agg(['sum', 'mean', 'count']).reset_index()
        geo_perf = {}
        for _, row in geo_agg.iterrows():
            geo_perf[row['metadata.country']] = {
                "total_revenue": round(row['sum'], 2),
                "avg_transaction": round(row['mean'], 2),
                "transaction_count": int(row['count'])
            }
        
        # Channel performance
        channel_agg = df.groupby('metadata.channel')['amount_eur'].agg(['sum', 'mean', 'count']).reset_index()
        channel_perf = {}
        for _, row in channel_agg.iterrows():
            channel_perf[row['metadata.channel']] = {
                "total_revenue": round(row['sum'], 2),
                "avg_transaction": round(row['mean'], 2),
                "transaction_count": int(row['count'])
            }
        
        report = {
            "success": True,
            "executive_summary": {
                "total_revenue": round(df['amount_eur'].sum(), 2),
                "total_transactions": len(df),
                "date_range": f"{df['metadata.batch_date'].min()} to {df['metadata.batch_date'].max()}"
            },
            "segment_performance": segment_perf,
            "geographic_performance": geo_perf,
            "channel_performance": channel_perf,
            "status_breakdown": df['status'].value_counts().to_dict()
        }
        return report
    except Exception as e:
        return {"success": False, "error": str(e)}


# ========================
# TOOL METADATA FOR LLM
# ========================

def get_data_engineer_tools():
    """Tools available to the Data Engineer Agent"""
    return [
        {
            "type": "function",
            "function": {
                "name": "read_csv_data",
                "description": "Read the payment CSV data and return records with metadata",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "file_path": {"type": "string", "description": "Path to CSV file (optional)"},
                        "sample_rows": {"type": "integer", "description": "Number of rows to sample (optional)"}
                    }
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "get_data_summary",
                "description": "Get statistical summary including data types, missing values, and basic statistics",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "file_path": {"type": "string", "description": "Path to CSV file (optional)"}
                    }
                }
            }
        }
    ]


def get_data_scientist_tools():
    """Tools available to the Data Scientist Agent"""
    return [
        {
            "type": "function",
            "function": {
                "name": "execute_analysis",
                "description": "Execute analysis on payment data. Types: revenue_by_country, revenue_by_segment, revenue_by_channel, status_distribution, segment_channel_matrix, custom",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "analysis_type": {"type": "string", "description": "Type of analysis to perform"},
                        "group_by": {"type": "string", "description": "Column to group by (for custom)"},
                        "metric": {"type": "string", "description": "Column to aggregate (for custom)"}
                    },
                    "required": ["analysis_type"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "calculate_kpis",
                "description": "Calculate key performance indicators from payment data",
                "parameters": {
                    "type": "object",
                    "properties": {}
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "detect_anomalies",
                "description": "Detect anomalies in transaction amounts using statistical methods",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "threshold_std": {"type": "number", "description": "Standard deviation threshold (default: 2.0)"}
                    }
                }
            }
        }
    ]


def get_business_manager_tools():
    """Tools available to the Business Manager Agent"""
    return [
        {
            "type": "function",
            "function": {
                "name": "calculate_kpis",
                "description": "Calculate key performance indicators",
                "parameters": {
                    "type": "object",
                    "properties": {}
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "generate_report_data",
                "description": "Generate comprehensive data for business report",
                "parameters": {
                    "type": "object",
                    "properties": {}
                }
            }
        }
    ]


# ========================
# TOOL CALL DISPATCHER
# ========================

TOOLS_MAP = {
    "read_csv_data": read_csv_data,
    "get_data_summary": get_data_summary,
    "execute_analysis": execute_analysis,
    "calculate_kpis": calculate_kpis,
    "detect_anomalies": detect_anomalies,
    "generate_report_data": generate_report_data,
}


def handle_tool_call(tool_call):
    """Execute a tool call and return the result"""
    function_name = tool_call.function.name
    arguments = json.loads(tool_call.function.arguments) if tool_call.function.arguments else {}
    
    if function_name in TOOLS_MAP:
        return TOOLS_MAP[function_name](**arguments)
    else:
        return {"error": f"Unknown tool: {function_name}"}


def create_tool_response_message(tool_call, tool_result):
    """Create a tool response message for the conversation"""
    return {
        "role": "tool",
        "tool_call_id": tool_call.id,
        "name": tool_call.function.name,
        "content": json.dumps(tool_result)
    }
