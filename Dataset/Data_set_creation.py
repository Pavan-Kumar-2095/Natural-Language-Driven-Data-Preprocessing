# import random
# import pandas as pd

# # ----------------------------
# # Column Pools
# # ----------------------------

# COLUMNS = [
#     "age", "salary", "income", "country",
#     "experience", "department", "rating",
#     "score", "price", "quantity"
# ]

# DATA_TYPES = ["int", "integer", "float", "string"]

# FILL_METHODS = {
#     "mean": ["mean", "average"],
#     "median": ["median"],
#     "mode": ["mode", "most frequent value", "most common value"]
# }

# MISSING_WORDS = ["missing", "null", "NaN", "empty", "blank"]

# # ----------------------------
# # Sentence Generators
# # ----------------------------

# def generate_fill_na():
#     column = random.choice(COLUMNS)
#     method_key = random.choice(list(FILL_METHODS.keys()))
#     method_word = random.choice(FILL_METHODS[method_key])
#     missing_word = random.choice(MISSING_WORDS)

#     templates = [
#         f"fill {missing_word} values in {column} using {method_word}",
#         f"replace {missing_word} values in {column} with {method_word}",
#         f"handle {missing_word} data in {column} using {method_word}",
#         f"fix {missing_word} entries in {column} with {method_word}",
#         f"use {method_word} to fill {missing_word} values in {column}"
#     ]

#     return random.choice(templates), "FILL_NA"


# def generate_drop_rows_missing():
#     missing_word = random.choice(MISSING_WORDS)

#     templates = [
#         f"drop rows with {missing_word} values",
#         f"remove rows containing {missing_word} data",
#         f"delete rows where values are {missing_word}",
#         f"eliminate rows with {missing_word} entries",
#         f"remove records having {missing_word} values"
#     ]

#     return random.choice(templates), "DROP_ROWS_MISSING"


# def generate_drop_column():
#     column = random.choice(COLUMNS)

#     templates = [
#         f"drop column {column}",
#         f"remove {column} column",
#         f"delete the {column} column",
#         f"get rid of column {column}",
#         f"exclude {column} from dataset"
#     ]

#     return random.choice(templates), "DROP_COLUMN"


# def generate_type_cast():
#     column = random.choice(COLUMNS)
#     dtype = random.choice(DATA_TYPES)

#     templates = [
#         f"convert {column} to {dtype}",
#         f"change {column} to {dtype} type",
#         f"cast {column} as {dtype}",
#         f"transform {column} into {dtype}",
#         f"set data type of {column} to {dtype}"
#     ]

#     return random.choice(templates), "TYPE_CAST"


# def generate_remove_duplicates_all():

#     templates = [
#         "remove duplicate rows",
#         "drop duplicates",
#         "delete duplicate entries",
#         "eliminate duplicate records",
#         "remove repeated rows",
#         "drop all duplicate rows"
#     ]

#     return random.choice(templates), "REMOVE_DUPLICATES_ALL"


# def generate_remove_duplicates_column():
#     column = random.choice(COLUMNS)

#     templates = [
#         f"remove duplicate values in {column}",
#         f"drop duplicates based on {column}",
#         f"delete duplicate entries in {column}",
#         f"eliminate duplicates in {column}",
#         f"remove repeated values in {column}",
#         f"drop duplicate records in {column}"
#     ]

#     return random.choice(templates), "REMOVE_DUPLICATES_COLUMN"


# # ----------------------------
# # Main Dataset Generator
# # ----------------------------

# def generate_dataset(samples_per_intent=500):

#     generators = [
#         generate_fill_na,
#         generate_drop_rows_missing,
#         generate_drop_column,
#         generate_type_cast,
#         generate_remove_duplicates_all,
#         generate_remove_duplicates_column
#     ]

#     data = []

#     for generator in generators:
#         for _ in range(samples_per_intent):
#             sentence, intent = generator()
#             data.append((sentence, intent))

#     random.shuffle(data)

#     df = pd.DataFrame(data, columns=["sentence", "intent"])
#     return df


# # ----------------------------
# # Create Dataset
# # ----------------------------

# if __name__ == "__main__":

#     df = generate_dataset(samples_per_intent=500)

#     df.to_csv("intent_dataset_final_v3.csv", index=False)

#     print("Dataset Created!")
#     print("Total Samples:", len(df))
#     print(df.head())
















import random
import pandas as pd

# ----------------------------
# Column Pools
# ----------------------------

NUMERIC_COLUMNS = [
    "age", "salary", "income", "experience",
    "rating", "score", "price", "quantity"
]

CATEGORICAL_COLUMNS = [
    "country", "department", "category",
    "city", "gender", "status"
]

DATA_TYPES = ["int", "integer", "float", "string"]

FILL_METHODS = {
    "mean": ["mean", "average"],
    "median": ["median"],
    "mode": ["mode", "most frequent value", "most common value"]
}

MISSING_WORDS = [
    "missing", "null", "NaN", "empty", "blank", 
    "unknown", "undefined"
]

POLITE_PREFIX = [
    "please", 
    "kindly", 
    "can you", 
    "could you", 
    "I want to", 
    "I need to",
    "help me to"
]

# Utility
def maybe_prefix(sentence):
    if random.random() < 0.4:
        return random.choice(POLITE_PREFIX) + " " + sentence
    return sentence

# ----------------------------
# INTENT GENERATORS
# ----------------------------

def generate_fill_na():
    column = random.choice(CATEGORICAL_COLUMNS)
    method_key = random.choice(list(FILL_METHODS.keys()))
    method = random.choice(FILL_METHODS[method_key])
    missing = random.choice(MISSING_WORDS)

    templates = [
        f"fill {missing} values in {column} using {method}",
        f"replace {missing} data in {column} with {method}",
        f"impute {missing} entries in {column} using {method}",
        f"handle {missing} values for {column} with {method}",
        f"substitute {missing} values in {column} by {method}",
        f"apply {method} imputation on {column}",
        f"use {method} strategy to treat {missing} values in {column}",
        f"perform missing value imputation on {column} using {method}"
    ]

    return maybe_prefix(random.choice(templates)), "FILL_NA"


def generate_drop_column():
    column = random.choice(CATEGORICAL_COLUMNS)

    templates = [
        f"drop column {column}",
        f"remove the {column} feature",
        f"delete {column} from the dataset",
        f"exclude {column} column",
        f"discard the {column} field",
        f"eliminate column named {column}",
        f"remove {column} attribute completely"
    ]

    return maybe_prefix(random.choice(templates)), "DROP_COLUMN"


def generate_type_cast():
    column = random.choice(CATEGORICAL_COLUMNS)
    dtype = random.choice(DATA_TYPES)

    templates = [
        f"convert {column} to {dtype}",
        f"cast {column} as {dtype}",
        f"change data type of {column} to {dtype}",
        f"transform {column} into {dtype} type",
        f"set {column} datatype as {dtype}",
        f"make {column} a {dtype} column",
        f"typecast {column} into {dtype}"
    ]

    return maybe_prefix(random.choice(templates)), "TYPE_CAST"


def generate_remove_duplicates_rows():

    templates = [
        "remove duplicate rows",
        "drop all duplicates",
        "delete repeated records",
        "eliminate duplicate entries from dataset",
        "remove redundant rows",
        "clean duplicate observations",
        "ensure dataset has no duplicate rows"
    ]

    return maybe_prefix(random.choice(templates)), "REMOVE_DUPLICATES_ROWS"


def generate_remove_outliers_column():
    column = random.choice(NUMERIC_COLUMNS)

    templates = [
        f"remove outliers in {column}",
        f"filter extreme values from {column}",
        f"drop anomalous data points in {column}",
        f"eliminate abnormal values in {column}",
        f"clean outliers from {column}",
        f"detect and remove outliers for {column}",
        f"trim extreme observations in {column}"
    ]

    return maybe_prefix(random.choice(templates)), "REMOVE_OUTLIERS_COLUMN"


def generate_normalize():
    column = random.choice(NUMERIC_COLUMNS)

    templates = [
        f"normalize {column}",
        f"scale {column} between 0 and 1",
        f"apply min max scaling to {column}",
        f"perform normalization on {column}",
        f"rescale {column} using min max normalization",
        f"convert {column} to 0-1 range",
        f"apply feature scaling to {column}"
    ]

    return maybe_prefix(random.choice(templates)), "NORMALIZE_COLUMN"


def generate_standardize():
    column = random.choice(NUMERIC_COLUMNS)

    templates = [
        f"standardize {column}",
        f"apply z score normalization to {column}",
        f"perform standard scaling on {column}",
        f"transform {column} to zero mean and unit variance",
        f"z-score normalize {column}",
        f"apply standard scaler to {column}",
        f"standardize the values of {column}"
    ]

    return maybe_prefix(random.choice(templates)), "STANDARDIZE_COLUMN"


def generate_label_encode_column():
    column = random.choice(CATEGORICAL_COLUMNS)

    templates = [
        f"label encode {column}",
        f"apply label encoding to {column}",
        f"convert {column} into numeric labels",
        f"encode categories in {column} as integers",
        f"transform {column} using label encoder",
        f"map {column} categories to numbers",
        f"convert categorical values in {column} to labels"
    ]

    return maybe_prefix(random.choice(templates)), "LABEL_ENCODE_COLUMN"


# ----------------------------
# Dataset Builder
# ----------------------------

def generate_dataset(samples_per_intent=600):

    generators = [
        generate_fill_na,
        generate_drop_column,
        generate_type_cast,
        generate_remove_duplicates_rows,
        generate_remove_outliers_column,
        generate_normalize,
        generate_standardize,
        generate_label_encode_column
    ]

    data = []

    for generator in generators:
        for _ in range(samples_per_intent):
            sentence, intent = generator()
            data.append((sentence, intent))

    random.shuffle(data)

    return pd.DataFrame(data, columns=["sentence", "intent"])


# ----------------------------
# Create Dataset
# ----------------------------

if __name__ == "__main__":

    df = generate_dataset(samples_per_intent=600)

    df.to_csv("intent_dataset_ml_diverse.csv", index=False)

    print("Dataset Created!")
    print("Total Samples:", len(df))
    print(df.head())
