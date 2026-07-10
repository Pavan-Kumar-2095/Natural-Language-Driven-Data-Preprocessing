# ClearData

ClearData is a FastAPI-based application that enables users to clean and preprocess CSV datasets using **natural language commands**. Instead of writing data processing code, users can simply describe the operation they want to perform, such as *"fill missing values with mean"* or *"remove outliers from age"*.

![Application Screenshot](Images/Screenshot_1.png)

---

## Features

- Upload CSV files
- Perform data cleaning using natural language commands
- Remove duplicate rows
- Fill missing values (mean, median, or zero)
- Label encoding
- Standardization
- Normalization
- Type casting (int, float, bool, string, datetime)
- Drop columns
- Remove outliers using the IQR method
- Preview data with pagination
- Download the processed CSV
- View dataset summary, statistics, and correlation matrix

---

## Project Structure

```text
ClearData/
├── main.py
├── models/
│   ├── tokenizer.pkl
│   ├── label_encoder.pkl
│   └── intent_lstm_model.keras
├── templates/
│   └── index.html
├── uploads/
├── Images/
└── README.md
```

---

## Requirements

- Python 3.9+
- FastAPI
- Pandas
- NumPy
- Scikit-learn
- Joblib
- RapidFuzz
- Jinja2
- Python Multipart

> **Note:** Ensure the following trained model files are available inside the `models/` directory:
>
> - `intent_lstm_model.keras`
> - `label_encoder.pkl`
> - `tokenizer.pkl`

---

## Installation

### Clone the Repository

```bash
git clone <repository-url>
cd ClearData
```

### Install Dependencies

```bash
pip install fastapi uvicorn pandas numpy scikit-learn joblib rapidfuzz jinja2 python-multipart
```

### Run the Application

```bash
uvicorn main:app --reload
```

The application will be available at:

```
http://127.0.0.1:8000
```

---

## Screenshots

### Main Interface

![Main Interface](Images/Screenshot_1.png)

### Data Processing

![Screenshot](Images/Screenshot_3.png)

![Screenshot](Images/Screenshot_4.png)

![Screenshot](Images/Screenshot_5.png)

![Screenshot](Images/Screenshot_6.png)

---

## Supported Operations

Examples of natural language commands:

- Remove duplicate rows
- Fill missing values with mean
- Fill missing values with median
- Normalize salary column
- Standardize age column
- Label encode gender column
- Remove outliers from age
- Drop email column
- Convert salary to integer
- Convert date to datetime

---

## LinkedIn Post

Read more about this project on LinkedIn:

https://www.linkedin.com/feed/update/urn:li:activity:7430480763772559360/

---