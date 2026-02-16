from fastapi import FastAPI, UploadFile, File, Request
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.templating import Jinja2Templates
import pandas as pd
import numpy as np
import os
import joblib
import math
from rapidfuzz import process as rf_process

app = FastAPI()



app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)



templates = Jinja2Templates(directory="templates")

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

MODEL_DIR = "models"

vectorizer = joblib.load(os.path.join(MODEL_DIR, "tfidf_vectorizer.pkl"))
model = joblib.load(os.path.join(MODEL_DIR, "logistic_classifier.pkl"))
label_encoder = joblib.load(os.path.join(MODEL_DIR, "label_encoder.pkl"))

uploaded_files = {}



def clean_nan(obj):
    if isinstance(obj, list):
        return [clean_nan(item) for item in obj]

    if isinstance(obj, dict):
        return {k: clean_nan(v) for k, v in obj.items()}

    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None

    return obj



def match_column(text, columns):
    text = text.lower()


    for col in columns:
        if col.lower() in text:
            return col

    columns_lower = [c.lower() for c in columns]
    best_match = rf_process.extractOne(text, columns_lower)

    if best_match and best_match[1] >= 60:   
        return columns[columns_lower.index(best_match[0])]

    return None



TYPE_OPTIONS = [
    "int", "integer",
    "float", "double",
    "string", "str",
    "bool", "boolean",
    "datetime"
]

TYPE_MAP = {
    "int": "int",
    "integer": "int",
    "float": "float",
    "double": "float",
    "string": "string",
    "str": "string",
    "bool": "bool",
    "boolean": "bool",
    "datetime": "datetime"
}

FILL_OPTIONS = ["mean", "median", "zero", "0"]

COLUMN_OPERATIONS = {
    "FILL_NA",
    "LABEL_ENCODE_COLUMN",
    "STANDARDIZE_COLUMN",
    "NORMALIZE_COLUMN",
    "TYPE_CAST",
    "DROP_COLUMN",
    "REMOVE_OUTLIERS_COLUMN"
}



@app.get("/")
def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})



@app.post("/upload")
async def upload_csv(file: UploadFile = File(...)):
    file_path = os.path.join(UPLOAD_DIR, file.filename)

    with open(file_path, "wb") as f:
        f.write(await file.read())

    try:
        df = pd.read_csv(file_path)
        df.columns = df.columns.str.strip()
        uploaded_files[file_path] = df
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=400)

    return {"file_path": file_path}



@app.get("/get_page")
async def get_page(file_path: str, page: int = 1, page_size: int = 5):
    if file_path not in uploaded_files:
        return JSONResponse({"error": "File not found"}, status_code=400)

    df = uploaded_files[file_path]

    if page_size > 100:
        page_size = 100

    total_rows = len(df)
    total_pages = max((total_rows + page_size - 1) // page_size, 1)

    if page < 1 or page > total_pages:
        return JSONResponse({"error": "Invalid page number"}, status_code=400)

    start = (page - 1) * page_size
    end = start + page_size

    page_df = df.iloc[start:end]
    data = page_df.to_dict(orient="records")
    data = clean_nan(data)

    return {
        "page": page,
        "page_size": page_size,
        "total_rows": total_rows,
        "total_pages": total_pages,
        "data": data
    }



@app.post("/process")
async def process_operation(payload: dict):

    file_path = payload.get("file_path")
    text = payload.get("text")

    if not file_path or not text:
        return JSONResponse({"error": "file_path and text required"}, status_code=400)

    if file_path not in uploaded_files:
        return JSONResponse({"error": "File not uploaded."}, status_code=400)

    df = uploaded_files[file_path]

    text_vector = vectorizer.transform([text])
    pred = model.predict(text_vector)
    confidence = model.predict_proba(text_vector).max()
    operation = label_encoder.inverse_transform(pred)[0]

    if confidence < 0.8:
        return JSONResponse(
            {"error": f"Low confidence ({confidence:.2f}). Try rephrasing."},
            status_code=400
        )

    matched_column = None

    if operation in COLUMN_OPERATIONS:
        matched_column = match_column(text, df.columns.tolist())

        if not matched_column:
            return JSONResponse(
                {"error": f"Could not match a column. Available columns: {df.columns.tolist()}"},
                status_code=400
            )

    try:

        # REMOVE DUPLICATES
        if operation == "REMOVE_DUPLICATES_ROWS":
            df = df.drop_duplicates()

        # FILL NA (MEAN / MEDIAN / ZERO)
        elif operation == "FILL_NA":

            match = rf_process.extractOne(text.lower(), FILL_OPTIONS)

            if not match or match[1] < 60:
                return JSONResponse(
                    {"error": "Specify fill method: mean, median, or zero."},
                    status_code=400
                )

            method = match[0]

            if method in ["mean", "median"]:
                if not pd.api.types.is_numeric_dtype(df[matched_column]):
                    return JSONResponse(
                        {"error": "Mean/Median only valid for numeric columns."},
                        status_code=400
                    )

            if method == "mean":
                value = df[matched_column].mean()

            elif method == "median":
                value = df[matched_column].median()

            elif method in ["zero", "0"]:
                value = 0

            df[matched_column] = df[matched_column].fillna(value)

        # LABEL ENCODE
        elif operation == "LABEL_ENCODE_COLUMN":
            from sklearn.preprocessing import LabelEncoder
            le = LabelEncoder()
            df[matched_column] = le.fit_transform(df[matched_column].astype(str))

        # STANDARDIZE
        elif operation == "STANDARDIZE_COLUMN":
            std = df[matched_column].std()
            if std == 0:
                return JSONResponse({"error": "Standard deviation is zero."}, status_code=400)

            df[matched_column] = (
                df[matched_column] - df[matched_column].mean()
            ) / std

        # NORMALIZE
        elif operation == "NORMALIZE_COLUMN":
            min_val = df[matched_column].min()
            max_val = df[matched_column].max()

            if max_val == min_val:
                return JSONResponse({"error": "Cannot normalize constant column."}, status_code=400)

            df[matched_column] = (
                df[matched_column] - min_val
            ) / (max_val - min_val)

        # TYPE CAST
        elif operation == "TYPE_CAST":

            match = rf_process.extractOne(text.lower(), TYPE_OPTIONS)

            if not match or match[1] < 60:
                return JSONResponse(
                    {"error": "Cannot detect target type."},
                    status_code=400
                )

            target_dtype = TYPE_MAP[match[0]]

            if target_dtype == "int":
                df[matched_column] = pd.to_numeric(df[matched_column], errors="coerce").astype("Int64")

            elif target_dtype == "float":
                df[matched_column] = pd.to_numeric(df[matched_column], errors="coerce")

            elif target_dtype == "bool":
                df[matched_column] = df[matched_column].astype(str).str.lower().map({
                    "true": True,
                    "1": True,
                    "false": False,
                    "0": False
                })

            elif target_dtype == "string":
                df[matched_column] = df[matched_column].astype("string")

            elif target_dtype == "datetime":
                df[matched_column] = pd.to_datetime(
                    df[matched_column],
                    format="mixed",
                    dayfirst=True,
                    errors="coerce"
                )

        # DROP COLUMN
        elif operation == "DROP_COLUMN":
            df = df.drop(columns=[matched_column])
            matched_column = None

        # REMOVE OUTLIERS
        elif operation == "REMOVE_OUTLIERS_COLUMN":
            q1 = df[matched_column].quantile(0.25)
            q3 = df[matched_column].quantile(0.75)
            iqr = q3 - q1

            df = df[
                (df[matched_column] >= q1 - 1.5 * iqr) &
                (df[matched_column] <= q3 + 1.5 * iqr)
            ]

        else:
            return JSONResponse(
                {"error": f"Operation '{operation}' not supported."},
                status_code=400
            )

    except Exception as e:
        return JSONResponse(
            {"error": f"Operation failed: {str(e)}"},
            status_code=400
        )

    uploaded_files[file_path] = df

    new_dtype = None
    if matched_column and matched_column in df.columns:
        new_dtype = str(df[matched_column].dtype)

    return clean_nan({
        "predicted_operation": operation,
        "matched_column": matched_column,
        "new_dtype": new_dtype
    })



@app.get("/download")
async def download_file(file_path: str):
    if file_path not in uploaded_files:
        return JSONResponse({"error": "File not found"}, status_code=400)

    df = uploaded_files[file_path]
    df.to_csv(file_path, index=False)

    return FileResponse(file_path, filename=os.path.basename(file_path))

# ------------------ Get Info ------------------

@app.get("/get_info")
async def get_info(file_path: str):
    if file_path not in uploaded_files:
        return JSONResponse({"error": "File not found"}, status_code=400)

    df = uploaded_files[file_path]

    info = {
        "rows": df.shape[0],
        "columns": df.shape[1],
        "column_names": df.columns.tolist(),
        "dtypes": df.dtypes.astype(str).to_dict(),
        "missing_values": df.isna().sum().to_dict()
    }

    return clean_nan(info)


# from fastapi import FastAPI, UploadFile, File, Request
# from fastapi.responses import JSONResponse, FileResponse
# from fastapi.middleware.cors import CORSMiddleware
# from fastapi.templating import Jinja2Templates
# import pandas as pd
# import os
# import joblib
# from rapidfuzz import process as rf_process

# app = FastAPI()

# # CORS middleware
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],
#     allow_methods=["*"],
#     allow_headers=["*"]
# )

# # Store uploaded and processed DataFrames
# uploaded_files = {}

# # Templates
# templates = Jinja2Templates(directory="templates")

# # Directories
# UPLOAD_DIR = "uploads"
# os.makedirs(UPLOAD_DIR, exist_ok=True)
# MODEL_DIR = "models"

# # Load ML models
# vectorizer = joblib.load(os.path.join(MODEL_DIR, "tfidf_vectorizer.pkl"))
# model = joblib.load(os.path.join(MODEL_DIR, "logistic_classifier.pkl"))
# label_encoder = joblib.load(os.path.join(MODEL_DIR, "label_encoder.pkl"))

# # Allowed typecast types
# TYPE_OPTIONS = ["int", "integer", "float", "double", "string", "str", "bool", "boolean"]
# TYPE_MAP = {
#     "int": "Int64",
#     "integer": "Int64",
#     "float": "float",
#     "double": "float",
#     "string": "str",
#     "str": "str",
#     "bool": "boolean",
#     "boolean": "boolean"
# }

# @app.get("/")
# def index(request: Request):
#     return templates.TemplateResponse("index.html", {"request": request})

# # Upload CSV
# @app.post("/upload")
# async def upload_csv(file: UploadFile = File(...)):
#     file_path = os.path.join(UPLOAD_DIR, file.filename)
#     with open(file_path, "wb") as f:
#         f.write(await file.read())
#     try:
#         df = pd.read_csv(file_path)
#         uploaded_files[file_path] = df  # Store dataframe
#         table = df.to_dict(orient="records") 
#     except Exception as e:
#         return JSONResponse({"error": str(e)}, status_code=400)
#     return {"file_path": file_path, "table": table}

# # Process operation
# @app.post("/process")
# async def process_operation(payload: dict):
#     file_path = payload.get("file_path")
#     text = payload.get("text")

#     if not os.path.exists(file_path):
#         return JSONResponse({"error": "File not found"}, status_code=400)

#     df = pd.read_csv(file_path)

#     # Predict operation with ML
#     text_vector = vectorizer.transform([text])
#     pred = model.predict(text_vector)
#     pred_proba = model.predict_proba(text_vector).max()
#     operation = label_encoder.inverse_transform(pred)[0]

#     if pred_proba < 0.8:
#         return JSONResponse({"error": f"Cannot confidently detect operation. Probability: {pred_proba:.2f}"}, status_code=400)

#     matched_column = None
#     if "COLUMN" in operation or "column" in operation.upper() or operation=="TYPE_CAST":
#         column_names = df.columns.tolist()
#         matched_column = rf_process.extractOne(text, column_names)[0]

#     try:
#         if operation == "REMOVE_DUPLICATES_ROWS":
#             df = df.drop_duplicates()
#         elif operation == "FILL_NA" and matched_column:
#             df[matched_column].fillna(0, inplace=True)
#         elif operation == "LABEL_ENCODE_COLUMN" and matched_column:
#             from sklearn.preprocessing import LabelEncoder
#             le = LabelEncoder()
#             df[matched_column] = le.fit_transform(df[matched_column])
#         elif operation == "STANDARDIZE_COLUMN" and matched_column:
#             df[matched_column] = (df[matched_column] - df[matched_column].mean()) / df[matched_column].std()
#         elif operation == "NORMALIZE_COLUMN" and matched_column:
#             df[matched_column] = (df[matched_column] - df[matched_column].min()) / (df[matched_column].max() - df[matched_column].min())
#         elif operation == "TYPE_CAST" and matched_column:
#             # Detect target type from text using rapidfuzz
#             matched_type, score, _ = rf_process.extractOne(text.lower(), TYPE_OPTIONS)
#             if score < 80:
#                 return JSONResponse({"error": f"Cannot detect target type from instruction '{text}'"}, status_code=400)
#             target_dtype = TYPE_MAP[matched_type]

#             try:
#                 if target_dtype == "Int64":  # integer
#                     if df[matched_column].isnull().any():
#                         return JSONResponse({"error": f"Column '{matched_column}' contains missing values, cannot convert to int."}, status_code=400)
#                     df[matched_column] = df[matched_column].astype(int)
#                 elif target_dtype == "float":  # float
#                     df[matched_column] = pd.to_numeric(df[matched_column], errors='raise')
#                 elif target_dtype == "bool":  # boolean
#                     # Convert common truthy/falsy strings or numeric 1/0
#                     df[matched_column] = df[matched_column].map(lambda x: True if str(x).lower() in ["true", "1"] else False if str(x).lower() in ["false", "0"] else None)
#                     if df[matched_column].isnull().any():
#                         return JSONResponse({"error": f"Cannot convert column '{matched_column}' to bool due to invalid values."}, status_code=400)
#                 elif target_dtype == "string":  # string
#                     df[matched_column] = df[matched_column].astype(str)
#                 elif target_dtype == "datetime":  # datetime
#                     df[matched_column] = pd.to_datetime(df[matched_column], errors='raise')
#                 else:
#                     return JSONResponse({"error": f"Unsupported target type '{target_dtype}'."}, status_code=400)
#             except Exception as e:
#                 return JSONResponse({"error": f"Cannot convert column '{matched_column}' to {target_dtype}. Error: {str(e)}"}, status_code=400)


#         elif operation == "DROP_COLUMN" and matched_column:
#             df.drop(columns=[matched_column], inplace=True)
#         elif operation == "REMOVE_OUTLIERS_COLUMN" and matched_column:
#             q1 = df[matched_column].quantile(0.25)
#             q3 = df[matched_column].quantile(0.75)
#             iqr = q3 - q1
#             df = df[(df[matched_column] >= q1 - 1.5 * iqr) & (df[matched_column] <= q3 + 1.5 * iqr)]
#     except Exception as e:
#         return JSONResponse({"error": f"Operation '{operation}' cannot be performed. Check the instruction or column. Details: {str(e)}"}, status_code=400)

#     # Save and update stored DataFrame
#     df.to_csv(file_path, index=False)
#     uploaded_files[file_path] = df
#     table = df.head(5).to_dict(orient="records")

#     return {"updated_table": table, "predicted_operation": operation, "matched_column": matched_column}

# # Download CSV
# @app.get("/download")
# async def download_file(file_path: str):
#     if os.path.exists(file_path):
#         return FileResponse(file_path, filename=os.path.basename(file_path))
#     return {"error": "File not found"}

# # Get DataFrame info
# @app.get("/get_info")
# async def get_info(file_path: str):
#     if not os.path.exists(file_path):
#         return {"error": "File not found."}

#     df = pd.read_csv(file_path)

#     numeric_cols = df.select_dtypes(include='number')
#     categorical_cols = df.select_dtypes(exclude='number')

#     numeric_summary = numeric_cols.describe().T.apply(lambda x: {
#         "count": int(x["count"]),
#         "mean": round(x["mean"],3) if pd.notna(x["mean"]) else None,
#         "std": round(x["std"],3) if pd.notna(x["std"]) else None,
#         "min": x["min"],
#         "25%": x["25%"],
#         "50%": x["50%"],
#         "75%": x["75%"],
#         "max": x["max"]
#     }, axis=1).to_dict()

#     categorical_summary = {}
#     for col in categorical_cols.columns:
#         categorical_summary[col] = {
#             "count": int(df[col].count()),
#             "unique": int(df[col].nunique()),
#             "top": df[col].mode()[0] if not df[col].mode().empty else None,
#             "freq": int(df[col].value_counts().iloc[0]) if not df[col].value_counts().empty else None
#         }

#     info_dict = {
#         "rows": df.shape[0],
#         "columns": df.shape[1],
#         "column_names": df.columns.tolist(),
#         "dtypes": df.dtypes.astype(str).to_dict(),
#         "missing_values": df.isna().sum().to_dict(),
#         "numeric_summary": numeric_summary,
#         "categorical_summary": categorical_summary
#     }

#     return JSONResponse(info_dict)
