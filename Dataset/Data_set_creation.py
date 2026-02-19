import random
import pandas as pd



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
    "missing", "null", "NaN", "empty",
    "blank", "unknown", "undefined"
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


def maybe_prefix(sentence):
    if random.random() < 0.4:
        return random.choice(POLITE_PREFIX) + " " + sentence
    return sentence




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
        f"perform missing value imputation on {column} using {method}",
        f"fix {missing} values in {column} with {method}",
        f"clean {missing} data in {column} using {method}",
        f"replace all {missing} in {column} by {method}",
        f"fill na in {column} using {method}",
        f"{column} fill missing with {method}",
        f"{method} imputation for {column}",
        f"resolve {missing} issue in {column}"
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
        f"remove {column} attribute completely",
        f"get rid of {column}",
        f"take out {column}",
        f"drop {column}",
        f"{column} remove",
        f"{column} delete",
        f"omit {column} field",
        f"erase {column}"
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
        f"typecast {column} into {dtype}",
        f"change {column} into {dtype}",
        f"turn {column} into {dtype}",
        f"{column} should be {dtype}",
        f"update {column} type to {dtype}",
        f"{column} convert to {dtype}"
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
        "ensure dataset has no duplicate rows",
        "get rid of duplicate rows",
        "remove row duplicates",
        "make dataset unique",
        "keep only unique rows",
        "deduplicate the dataset",
        "no duplicate rows",
        "clean duplicates"
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
        f"trim extreme observations in {column}",
        f"get rid of outliers in {column}",
        f"remove extreme {column} values",
        f"exclude abnormal {column} values",
        f"{column} remove outliers",
        f"drop extreme {column}"
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
        f"apply feature scaling to {column}",
        f"min max normalize {column}",
        f"bring {column} to range 0 and 1",
        f"rescale {column} to [0,1]",
        f"{column} normalization",
        f"make {column} between 0 and 1"
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
        f"standardize the values of {column}",
        f"z score scale {column}",
        f"convert {column} to z scores",
        f"standard scale {column}",
        f"{column} standardization"
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
        f"convert categorical values in {column} to labels",
        f"encode {column}",
        f"convert {column} to numbers",
        f"assign labels to {column}",
        f"{column} label encoding",
        f"{column} encode"
    ]

    return maybe_prefix(random.choice(templates)), "LABEL_ENCODE_COLUMN"




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






if __name__ == "__main__":

    df = generate_dataset(samples_per_intent=600)

    df.to_csv("intent_dataset_ml_diverse.csv", index=False)

    print("Dataset Created!")
    print("Total Samples:", len(df))
    print(df.head())










# import random
# import pandas as pd

# # ----------------------------
# # Column Pools
# # ----------------------------

# NUMERIC_COLUMNS = [
#     "age", "salary", "income", "experience",
#     "rating", "score", "price", "quantity"
# ]

# CATEGORICAL_COLUMNS = [
#     "country", "department", "category",
#     "city", "gender", "status"
# ]

# DATA_TYPES = ["int", "integer", "float", "string"]

# FILL_METHODS = {
#     "mean": ["mean", "average"],
#     "median": ["median"],
#     "mode": ["mode", "most frequent value", "most common value"]
# }

# MISSING_WORDS = [
#     "missing", "null", "NaN", "empty", "blank", 
#     "unknown", "undefined"
# ]

# POLITE_PREFIX = [
#     "please", 
#     "kindly", 
#     "can you", 
#     "could you", 
#     "I want to", 
#     "I need to",
#     "help me to"
# ]

# # Utility
# def maybe_prefix(sentence):
#     if random.random() < 0.4:
#         return random.choice(POLITE_PREFIX) + " " + sentence
#     return sentence

# # ----------------------------
# # INTENT GENERATORS
# # ----------------------------

# def generate_fill_na():
#     column = random.choice(CATEGORICAL_COLUMNS)
#     method_key = random.choice(list(FILL_METHODS.keys()))
#     method = random.choice(FILL_METHODS[method_key])
#     missing = random.choice(MISSING_WORDS)

#     templates = [
#         f"fill {missing} values in {column} using {method}",
#         f"replace {missing} data in {column} with {method}",
#         f"impute {missing} entries in {column} using {method}",
#         f"handle {missing} values for {column} with {method}",
#         f"substitute {missing} values in {column} by {method}",
#         f"apply {method} imputation on {column}",
#         f"use {method} strategy to treat {missing} values in {column}",
#         f"perform missing value imputation on {column} using {method}"
#     ]

#     return maybe_prefix(random.choice(templates)), "FILL_NA"


# def generate_drop_column():
#     column = random.choice(CATEGORICAL_COLUMNS)

#     templates = [
#         f"drop column {column}",
#         f"remove the {column} feature",
#         f"delete {column} from the dataset",
#         f"exclude {column} column",
#         f"discard the {column} field",
#         f"eliminate column named {column}",
#         f"remove {column} attribute completely"
#     ]

#     return maybe_prefix(random.choice(templates)), "DROP_COLUMN"


# def generate_type_cast():
#     column = random.choice(CATEGORICAL_COLUMNS)
#     dtype = random.choice(DATA_TYPES)

#     templates = [
#         f"convert {column} to {dtype}",
#         f"cast {column} as {dtype}",
#         f"change data type of {column} to {dtype}",
#         f"transform {column} into {dtype} type",
#         f"set {column} datatype as {dtype}",
#         f"make {column} a {dtype} column",
#         f"typecast {column} into {dtype}"
#     ]

#     return maybe_prefix(random.choice(templates)), "TYPE_CAST"


# def generate_remove_duplicates_rows():

#     templates = [
#         "remove duplicate rows",
#         "drop all duplicates",
#         "delete repeated records",
#         "eliminate duplicate entries from dataset",
#         "remove redundant rows",
#         "clean duplicate observations",
#         "ensure dataset has no duplicate rows"
#     ]

#     return maybe_prefix(random.choice(templates)), "REMOVE_DUPLICATES_ROWS"


# def generate_remove_outliers_column():
#     column = random.choice(NUMERIC_COLUMNS)

#     templates = [
#         f"remove outliers in {column}",
#         f"filter extreme values from {column}",
#         f"drop anomalous data points in {column}",
#         f"eliminate abnormal values in {column}",
#         f"clean outliers from {column}",
#         f"detect and remove outliers for {column}",
#         f"trim extreme observations in {column}"
#     ]

#     return maybe_prefix(random.choice(templates)), "REMOVE_OUTLIERS_COLUMN"


# def generate_normalize():
#     column = random.choice(NUMERIC_COLUMNS)

#     templates = [
#         f"normalize {column}",
#         f"scale {column} between 0 and 1",
#         f"apply min max scaling to {column}",
#         f"perform normalization on {column}",
#         f"rescale {column} using min max normalization",
#         f"convert {column} to 0-1 range",
#         f"apply feature scaling to {column}"
#     ]

#     return maybe_prefix(random.choice(templates)), "NORMALIZE_COLUMN"


# def generate_standardize():
#     column = random.choice(NUMERIC_COLUMNS)

#     templates = [
#         f"standardize {column}",
#         f"apply z score normalization to {column}",
#         f"perform standard scaling on {column}",
#         f"transform {column} to zero mean and unit variance",
#         f"z-score normalize {column}",
#         f"apply standard scaler to {column}",
#         f"standardize the values of {column}"
#     ]

#     return maybe_prefix(random.choice(templates)), "STANDARDIZE_COLUMN"


# def generate_label_encode_column():
#     column = random.choice(CATEGORICAL_COLUMNS)

#     templates = [
#         f"label encode {column}",
#         f"apply label encoding to {column}",
#         f"convert {column} into numeric labels",
#         f"encode categories in {column} as integers",
#         f"transform {column} using label encoder",
#         f"map {column} categories to numbers",
#         f"convert categorical values in {column} to labels"
#     ]

#     return maybe_prefix(random.choice(templates)), "LABEL_ENCODE_COLUMN"


# # ----------------------------
# # Dataset Builder
# # ----------------------------

# def generate_dataset(samples_per_intent=600):

#     generators = [
#         generate_fill_na,
#         generate_drop_column,
#         generate_type_cast,
#         generate_remove_duplicates_rows,
#         generate_remove_outliers_column,
#         generate_normalize,
#         generate_standardize,
#         generate_label_encode_column
#     ]

#     data = []

#     for generator in generators:
#         for _ in range(samples_per_intent):
#             sentence, intent = generator()
#             data.append((sentence, intent))

#     random.shuffle(data)

#     return pd.DataFrame(data, columns=["sentence", "intent"])


# # ----------------------------
# # Create Dataset
# # ----------------------------

# if __name__ == "__main__":

#     df = generate_dataset(samples_per_intent=600)

#     df.to_csv("intent_dataset_ml_diverse.csv", index=False)

#     print("Dataset Created!")
#     print("Total Samples:", len(df))
#     print(df.head())
