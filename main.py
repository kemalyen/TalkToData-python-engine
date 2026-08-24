import logging
import json
import os
import re
from typing import Any

import duckdb
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="AI Data Warehouse Engine")

# Request Models
class ProfileRequest(BaseModel):
    file_path: str

class AnalysisRequest(BaseModel):
    file_path: str
    schema_json: dict
    prompt: str


def json_records(dataframe: pd.DataFrame) -> list[dict[str, Any]]:
    return json.loads(dataframe.head(5).to_json(orient="records", date_format="iso"))


def build_numeric_summary(dataframe: pd.DataFrame) -> dict[str, dict[str, Any]]:
    summary: dict[str, dict[str, Any]] = {}

    for column in dataframe.select_dtypes(include="number").columns:
        values = dataframe[column].dropna()
        summary[column] = {
            "count": int(values.count()),
            "null_count": int(dataframe[column].isna().sum()),
            "min": float(values.min()) if not values.empty else None,
            "max": float(values.max()) if not values.empty else None,
            "mean": float(values.mean()) if not values.empty else None,
            "sum": float(values.sum()) if not values.empty else None,
        }

    return summary


def is_dataset_question(prompt: str, schema_json: dict) -> bool:
    prompt_tokens = set(re.findall(r"[a-zA-Z0-9_]+", prompt.lower()))
    analysis_terms = {
        "average", "column", "columns", "count", "dataset", "field", "fields",
        "highest", "lowest", "maximum", "mean", "metric", "metrics", "minimum",
        "numerical", "number", "numbers", "null", "percent", "percentage", "record",
        "records", "row", "rows", "sum", "summarize", "summary", "table", "total",
    }
    column_names = {
        token
        for column in schema_json.get("columns", [])
        for token in re.findall(r"[a-zA-Z0-9_]+", str(column.get("name", "")).lower())
    }

    return bool(prompt_tokens & (analysis_terms | column_names))

# 1. Health Check Endpoint
@app.get("/health")
def health_check():
    return {"status": "ok", "engine": "FastAPI + Python", "timestamp": pd.Timestamp.now().isoformat()}

# 2. Profile Dataset Endpoint (Called right after Laravel upload)
@app.post("/profile-dataset")
def profile_dataset(req: ProfileRequest):
    if not os.path.exists(req.file_path):
        raise HTTPException(status_code=404, detail="File not found")

    try:
        # Read first 100 rows for fast schema inference
        df = pd.read_csv(req.file_path, nrows=100)
        
        # Get total row count
        total_rows = sum(1 for _ in open(req.file_path)) - 1

        columns = [
            {"name": col, "type": str(dtype)} 
            for col, dtype in zip(df.columns, df.dtypes)
        ]

        return {
            "status": "success",
            "row_count": total_rows,
            "schema": {
                "columns": columns,
                "sample_rows": df.head(3).to_dict(orient="records")
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# 3. Analyze Endpoint (Executed when Laravel AI Tool calls Python)
@app.post("/analyze")
def analyze_dataset(req: AnalysisRequest):
    logger.info("analyze_dataset called: %s", req.model_dump())

    if not os.path.exists(req.file_path):
        logger.error("analyze_dataset file not found: %s", req.file_path)
        raise HTTPException(status_code=404, detail="File not found")

    if not is_dataset_question(req.prompt, req.schema_json):
        response = {
            "status": "rejected",
            "reason": "I can only answer questions about this uploaded dataset.",
            "prompt": req.prompt,
        }
        logger.info("analyze_dataset rejected out-of-scope prompt: %s", response)
        return response

    try:
        # Read the complete dataset so summary metrics reflect all rows.
        query = f"SELECT * FROM read_csv_auto('{req.file_path}')"
        result_df = duckdb.query(query).to_df()
        numeric_summary = build_numeric_summary(result_df)
        columns = [
            {"name": column, "type": str(result_df[column].dtype)}
            for column in result_df.columns
        ]

        chart_column = next(iter(numeric_summary), None)
        chart_config = None
        if chart_column:
            chart_config = {
                "type": "bar",
                "data": {
                    "labels": [chart_column],
                    "datasets": [{
                        "label": f"Sum of {chart_column}",
                        "data": [numeric_summary[chart_column]["sum"]],
                    }],
                },
            }

        response = {
            "status": "success",
            "executed_query": query,
            "prompt": req.prompt,
            "row_count": int(len(result_df)),
            "columns": columns,
            "numeric_summary": numeric_summary,
            "result_sample": json_records(result_df),
            "chart_config": chart_config,
        }

        logger.info("analyze_dataset response: %s", response)

        return response
    except Exception as e:
        logger.exception("analyze_dataset failed for %s", req.file_path)
        raise HTTPException(status_code=500, detail=str(e))