from fastapi import FastAPI, UploadFile, File, Request
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.templating import Jinja2Templates
import pandas as pd
import numpy as np
import os
import math
import pickle
from rapidfuzz import process as rf_process
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

app = FastAPI()

# ---------------- CORS ----------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

# ---------------- Templates ----------------
templates = Jinja2Templates(directory="templates")

# ---------------- Directories ----------------
UPLOAD_DIR = "uploads"
MODEL_DIR = "models"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# ---------------- Load Models ----------------
with open(os.path.join(MODEL_DIR, "tokenizer.pkl"), "rb") as f:
    tokenizer = pickle.load(f)

model = load_model(os.path.join(MODEL_DIR, "intent_lstm_model.keras"))

with open(os.path.join(MODEL_DIR, "label_encoder.pkl"), "rb") as f:
    label_encoder = pickle.load(f)

MAX_SEQUENCE_LENGTH = 30  

# ---------------- Global Variables ----------------
uploaded_files = {}

TYPE_OPTIONS = ["int", "integer", "float", "double", "string", "str", "bool", "boolean", "datetime"]
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

# ---------------- Utility Functions ----------------
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

# ---------------- Endpoints ----------------
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
    page_size = min(page_size, 100)
    total_rows = len(df)
    total_pages = max((total_rows + page_size - 1) // page_size, 1)

    if page < 1 or page > total_pages:
        return JSONResponse({"error": "Invalid page number"}, status_code=400)

    start = (page - 1) * page_size
    end = start + page_size
    page_df = df.iloc[start:end]
    data = clean_nan(page_df.to_dict(orient="records"))

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

    # ----- Tokenize and pad text for LSTM -----
    seq = tokenizer.texts_to_sequences([text])
    padded_seq = pad_sequences(seq, maxlen=MAX_SEQUENCE_LENGTH, padding="post")

    # ----- Predict operation -----
    pred_probs = model.predict(padded_seq)
    confidence = float(pred_probs.max())
    pred_label_idx = pred_probs.argmax(axis=1)[0]
    operation = label_encoder.inverse_transform([pred_label_idx])[0]

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
        # ---------------- Operations ----------------
        # REMOVE DUPLICATES
        if operation == "REMOVE_DUPLICATES_ROWS":
            df = df.drop_duplicates()

        # FILL NA
        elif operation == "FILL_NA":
            match = rf_process.extractOne(text.lower(), FILL_OPTIONS)
            if not match or match[1] < 60:
                return JSONResponse({"error": "Specify fill method: mean, median, or zero."}, status_code=400)
            method = match[0]
            if method in ["mean", "median"] and not pd.api.types.is_numeric_dtype(df[matched_column]):
                return JSONResponse({"error": "Mean/Median only valid for numeric columns."}, status_code=400)
            if method == "mean":
                value = df[matched_column].mean()
            elif method == "median":
                value = df[matched_column].median()
            else:
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
            df[matched_column] = (df[matched_column] - df[matched_column].mean()) / std

        # NORMALIZE
        elif operation == "NORMALIZE_COLUMN":
            min_val, max_val = df[matched_column].min(), df[matched_column].max()
            if max_val == min_val:
                return JSONResponse({"error": "Cannot normalize constant column."}, status_code=400)
            df[matched_column] = (df[matched_column] - min_val) / (max_val - min_val)

        # TYPE CAST
        elif operation == "TYPE_CAST":
            match = rf_process.extractOne(text.lower(), TYPE_OPTIONS)
            if not match or match[1] < 60:
                return JSONResponse({"error": "Cannot detect target type."}, status_code=400)
            target_dtype = TYPE_MAP[match[0]]
            if target_dtype == "int":
                df[matched_column] = pd.to_numeric(df[matched_column], errors="coerce").astype("Int64")
            elif target_dtype == "float":
                df[matched_column] = pd.to_numeric(df[matched_column], errors="coerce")
            elif target_dtype == "bool":
                df[matched_column] = df[matched_column].astype(str).str.lower().map({"true": True,"1": True,"false": False,"0": False})
            elif target_dtype == "string":
                df[matched_column] = df[matched_column].astype("string")
            elif target_dtype == "datetime":
                df[matched_column] = pd.to_datetime(df[matched_column], errors="coerce", dayfirst=True)

        # DROP COLUMN
        elif operation == "DROP_COLUMN":
            df = df.drop(columns=[matched_column])
            matched_column = None

        # REMOVE OUTLIERS
        elif operation == "REMOVE_OUTLIERS_COLUMN":
            q1, q3 = df[matched_column].quantile([0.25, 0.75])
            iqr = q3 - q1
            df = df[(df[matched_column] >= q1 - 1.5 * iqr) & (df[matched_column] <= q3 + 1.5 * iqr)]

        else:
            return JSONResponse({"error": f"Operation '{operation}' not supported."}, status_code=400)

    except Exception as e:
        return JSONResponse({"error": f"Operation failed: {str(e)}"}, status_code=400)

    uploaded_files[file_path] = df
    new_dtype = str(df[matched_column].dtype) if matched_column and matched_column in df.columns else None

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
# import numpy as np
# import os
# import joblib
# import math
# from rapidfuzz import process as rf_process

# app = FastAPI()



# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],
#     allow_methods=["*"],
#     allow_headers=["*"]
# )



# templates = Jinja2Templates(directory="templates")

# UPLOAD_DIR = "uploads"
# os.makedirs(UPLOAD_DIR, exist_ok=True)

# MODEL_DIR = "models"

# vectorizer = joblib.load(os.path.join(MODEL_DIR, "tfidf_vectorizer.pkl"))
# model = joblib.load(os.path.join(MODEL_DIR, "logistic_classifier.pkl"))
# label_encoder = joblib.load(os.path.join(MODEL_DIR, "label_encoder.pkl"))

# uploaded_files = {}



# def clean_nan(obj):
#     if isinstance(obj, list):
#         return [clean_nan(item) for item in obj]

#     if isinstance(obj, dict):
#         return {k: clean_nan(v) for k, v in obj.items()}

#     if isinstance(obj, float):
#         if math.isnan(obj) or math.isinf(obj):
#             return None

#     return obj



# def match_column(text, columns):
#     text = text.lower()


#     for col in columns:
#         if col.lower() in text:
#             return col

#     columns_lower = [c.lower() for c in columns]
#     best_match = rf_process.extractOne(text, columns_lower)

#     if best_match and best_match[1] >= 60:   
#         return columns[columns_lower.index(best_match[0])]

#     return None



# TYPE_OPTIONS = [
#     "int", "integer",
#     "float", "double",
#     "string", "str",
#     "bool", "boolean",
#     "datetime"
# ]

# TYPE_MAP = {
#     "int": "int",
#     "integer": "int",
#     "float": "float",
#     "double": "float",
#     "string": "string",
#     "str": "string",
#     "bool": "bool",
#     "boolean": "bool",
#     "datetime": "datetime"
# }

# FILL_OPTIONS = ["mean", "median", "zero", "0"]

# COLUMN_OPERATIONS = {
#     "FILL_NA",
#     "LABEL_ENCODE_COLUMN",
#     "STANDARDIZE_COLUMN",
#     "NORMALIZE_COLUMN",
#     "TYPE_CAST",
#     "DROP_COLUMN",
#     "REMOVE_OUTLIERS_COLUMN"
# }



# @app.get("/")
# def index(request: Request):
#     return templates.TemplateResponse("index.html", {"request": request})



# @app.post("/upload")
# async def upload_csv(file: UploadFile = File(...)):
#     file_path = os.path.join(UPLOAD_DIR, file.filename)

#     with open(file_path, "wb") as f:
#         f.write(await file.read())

#     try:
#         df = pd.read_csv(file_path)
#         df.columns = df.columns.str.strip()
#         uploaded_files[file_path] = df
#     except Exception as e:
#         return JSONResponse({"error": str(e)}, status_code=400)

#     return {"file_path": file_path}



# @app.get("/get_page")
# async def get_page(file_path: str, page: int = 1, page_size: int = 5):
#     if file_path not in uploaded_files:
#         return JSONResponse({"error": "File not found"}, status_code=400)

#     df = uploaded_files[file_path]

#     if page_size > 100:
#         page_size = 100

#     total_rows = len(df)
#     total_pages = max((total_rows + page_size - 1) // page_size, 1)

#     if page < 1 or page > total_pages:
#         return JSONResponse({"error": "Invalid page number"}, status_code=400)

#     start = (page - 1) * page_size
#     end = start + page_size

#     page_df = df.iloc[start:end]
#     data = page_df.to_dict(orient="records")
#     data = clean_nan(data)

#     return {
#         "page": page,
#         "page_size": page_size,
#         "total_rows": total_rows,
#         "total_pages": total_pages,
#         "data": data
#     }



# @app.post("/process")
# async def process_operation(payload: dict):

#     file_path = payload.get("file_path")
#     text = payload.get("text")

#     if not file_path or not text:
#         return JSONResponse({"error": "file_path and text required"}, status_code=400)

#     if file_path not in uploaded_files:
#         return JSONResponse({"error": "File not uploaded."}, status_code=400)

#     df = uploaded_files[file_path]

#     text_vector = vectorizer.transform([text])
#     pred = model.predict(text_vector)
#     confidence = model.predict_proba(text_vector).max()
#     operation = label_encoder.inverse_transform(pred)[0]

#     if confidence < 0.8:
#         return JSONResponse(
#             {"error": f"Low confidence ({confidence:.2f}). Try rephrasing."},
#             status_code=400
#         )

#     matched_column = None

#     if operation in COLUMN_OPERATIONS:
#         matched_column = match_column(text, df.columns.tolist())

#         if not matched_column:
#             return JSONResponse(
#                 {"error": f"Could not match a column. Available columns: {df.columns.tolist()}"},
#                 status_code=400
#             )

#     try:

#         # REMOVE DUPLICATES
#         if operation == "REMOVE_DUPLICATES_ROWS":
#             df = df.drop_duplicates()

#         # FILL NA (MEAN / MEDIAN / ZERO)
#         elif operation == "FILL_NA":

#             match = rf_process.extractOne(text.lower(), FILL_OPTIONS)

#             if not match or match[1] < 60:
#                 return JSONResponse(
#                     {"error": "Specify fill method: mean, median, or zero."},
#                     status_code=400
#                 )

#             method = match[0]

#             if method in ["mean", "median"]:
#                 if not pd.api.types.is_numeric_dtype(df[matched_column]):
#                     return JSONResponse(
#                         {"error": "Mean/Median only valid for numeric columns."},
#                         status_code=400
#                     )

#             if method == "mean":
#                 value = df[matched_column].mean()

#             elif method == "median":
#                 value = df[matched_column].median()

#             elif method in ["zero", "0"]:
#                 value = 0

#             df[matched_column] = df[matched_column].fillna(value)

#         # LABEL ENCODE
#         elif operation == "LABEL_ENCODE_COLUMN":
#             from sklearn.preprocessing import LabelEncoder
#             le = LabelEncoder()
#             df[matched_column] = le.fit_transform(df[matched_column].astype(str))

#         # STANDARDIZE
#         elif operation == "STANDARDIZE_COLUMN":
#             std = df[matched_column].std()
#             if std == 0:
#                 return JSONResponse({"error": "Standard deviation is zero."}, status_code=400)

#             df[matched_column] = (
#                 df[matched_column] - df[matched_column].mean()
#             ) / std

#         # NORMALIZE
#         elif operation == "NORMALIZE_COLUMN":
#             min_val = df[matched_column].min()
#             max_val = df[matched_column].max()

#             if max_val == min_val:
#                 return JSONResponse({"error": "Cannot normalize constant column."}, status_code=400)

#             df[matched_column] = (
#                 df[matched_column] - min_val
#             ) / (max_val - min_val)

#         # TYPE CAST
#         elif operation == "TYPE_CAST":

#             match = rf_process.extractOne(text.lower(), TYPE_OPTIONS)

#             if not match or match[1] < 60:
#                 return JSONResponse(
#                     {"error": "Cannot detect target type."},
#                     status_code=400
#                 )

#             target_dtype = TYPE_MAP[match[0]]

#             if target_dtype == "int":
#                 df[matched_column] = pd.to_numeric(df[matched_column], errors="coerce").astype("Int64")

#             elif target_dtype == "float":
#                 df[matched_column] = pd.to_numeric(df[matched_column], errors="coerce")

#             elif target_dtype == "bool":
#                 df[matched_column] = df[matched_column].astype(str).str.lower().map({
#                     "true": True,
#                     "1": True,
#                     "false": False,
#                     "0": False
#                 })

#             elif target_dtype == "string":
#                 df[matched_column] = df[matched_column].astype("string")

#             elif target_dtype == "datetime":
#                 df[matched_column] = pd.to_datetime(
#                     df[matched_column],
#                     format="mixed",
#                     dayfirst=True,
#                     errors="coerce"
#                 )

#         # DROP COLUMN
#         elif operation == "DROP_COLUMN":
#             df = df.drop(columns=[matched_column])
#             matched_column = None

#         # REMOVE OUTLIERS
#         elif operation == "REMOVE_OUTLIERS_COLUMN":
#             q1 = df[matched_column].quantile(0.25)
#             q3 = df[matched_column].quantile(0.75)
#             iqr = q3 - q1

#             df = df[
#                 (df[matched_column] >= q1 - 1.5 * iqr) &
#                 (df[matched_column] <= q3 + 1.5 * iqr)
#             ]

#         else:
#             return JSONResponse(
#                 {"error": f"Operation '{operation}' not supported."},
#                 status_code=400
#             )

#     except Exception as e:
#         return JSONResponse(
#             {"error": f"Operation failed: {str(e)}"},
#             status_code=400
#         )

#     uploaded_files[file_path] = df

#     new_dtype = None
#     if matched_column and matched_column in df.columns:
#         new_dtype = str(df[matched_column].dtype)

#     return clean_nan({
#         "predicted_operation": operation,
#         "matched_column": matched_column,
#         "new_dtype": new_dtype
#     })



# @app.get("/download")
# async def download_file(file_path: str):
#     if file_path not in uploaded_files:
#         return JSONResponse({"error": "File not found"}, status_code=400)

#     df = uploaded_files[file_path]
#     df.to_csv(file_path, index=False)

#     return FileResponse(file_path, filename=os.path.basename(file_path))

# # ------------------ Get Info ------------------

# @app.get("/get_info")
# async def get_info(file_path: str):
#     if file_path not in uploaded_files:
#         return JSONResponse({"error": "File not found"}, status_code=400)

#     df = uploaded_files[file_path]

#     info = {
#         "rows": df.shape[0],
#         "columns": df.shape[1],
#         "column_names": df.columns.tolist(),
#         "dtypes": df.dtypes.astype(str).to_dict(),
#         "missing_values": df.isna().sum().to_dict()
#     }

#     return clean_nan(info)

