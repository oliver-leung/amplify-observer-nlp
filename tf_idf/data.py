from pyspark.sql.session import SparkSession
from pyspark.sql.dataframe import DataFrame
#  You may want to configure the Spark Context with the right credentials provider.
spark = SparkSession.builder.master('local').getOrCreate()

mode = None

def capture_stdout(func, *args, **kwargs):
    """Capture standard output to a string buffer"""

    from contextlib import redirect_stdout
    import io

    stdout_string = io.StringIO()
    with redirect_stdout(stdout_string):
        func(*args, **kwargs)
    return stdout_string.getvalue()


def convert_or_coerce(pandas_df, spark):
    """Convert pandas df to pyspark df and coerces the mixed cols to string"""
    import re

    try:
        return spark.createDataFrame(pandas_df)
    except TypeError as e:
        match = re.search(r".*field (\w+).*Can not merge type.*", str(e))
        if match is None:
            raise e
        mixed_col_name = match.group(1)
        # Coercing the col to string
        pandas_df[mixed_col_name] = pandas_df[mixed_col_name].astype("str")
        return pandas_df


def default_spark(value):
    return {"default": value}


def default_spark_with_stdout(df, stdout):
    return {
        "default": df,
        "stdout": stdout,
    }


def default_spark_with_trained_parameters(value, trained_parameters):
    return {"default": value, "trained_parameters": trained_parameters}


def default_spark_with_trained_parameters_and_state(df, trained_parameters, state):
    return {"default": df, "trained_parameters": trained_parameters, "state": state}


def dispatch(key_name, args, kwargs, funcs):
    """
    Dispatches to another operator based on a key in the passed parameters.
    This also slices out any parameters using the parameter_name passed in,
    and will reassemble the trained_parameters correctly after invocation.

    Args:
        key_name: name of the key in kwargs used to identify the function to use.
        args: dataframe that will be passed as the first set of parameters to the function.
        kwargs: keyword arguments that key_name will be found in; also where args will be passed to parameters.
                These are also expected to include trained_parameters if there are any.
        funcs: dictionary mapping from value of key_name to (function, parameter_name)

    """
    if key_name not in kwargs:
        raise OperatorCustomerError(f"Missing required parameter {key_name}")

    operator = kwargs[key_name]

    if operator not in funcs:
        raise OperatorCustomerError(f"Invalid choice selected for {key_name}. {operator} is not supported.")

    func, parameter_name = funcs[operator]

    # Extract out the parameters that should be available.
    func_params = kwargs.get(parameter_name, {})
    if func_params is None:
        func_params = {}

    # Extract out any trained parameters.
    specific_trained_parameters = None
    if "trained_parameters" in kwargs:
        trained_parameters = kwargs["trained_parameters"]
        if trained_parameters is not None and parameter_name in trained_parameters:
            specific_trained_parameters = trained_parameters[parameter_name]
    func_params["trained_parameters"] = specific_trained_parameters

    result = spark_operator_with_escaped_column(func, args, func_params)

    # Check if the result contains any trained parameters and remap them to the proper structure.
    if result is not None and "trained_parameters" in result:
        existing_trained_parameters = kwargs.get("trained_parameters")
        updated_trained_parameters = result["trained_parameters"]

        if existing_trained_parameters is not None or updated_trained_parameters is not None:
            existing_trained_parameters = existing_trained_parameters if existing_trained_parameters is not None else {}
            existing_trained_parameters[parameter_name] = result["trained_parameters"]

            # Update the result trained_parameters so they are part of the original structure.
            result["trained_parameters"] = existing_trained_parameters
        else:
            # If the given trained parameters were None and the returned trained parameters were None, don't return anything.
            del result["trained_parameters"]

    return result


def get_dataframe_with_sequence_ids(df: DataFrame):
    df_cols = df.columns
    rdd_with_seq = df.rdd.zipWithIndex()
    df_with_seq = rdd_with_seq.toDF()
    df_with_seq = df_with_seq.withColumnRenamed("_2", "_seq_id_")
    for col_name in df_cols:
        df_with_seq = df_with_seq.withColumn(col_name, df_with_seq["_1"].getItem(col_name))
    df_with_seq = df_with_seq.drop("_1")
    return df_with_seq


def get_execution_state(status: str, message=None):
    return {"status": status, "message": message}


def rename_invalid_column(df, orig_col):
    """Rename a given column in a data frame to a new valid name

    Args:
        df: Spark dataframe
        orig_col: input column name

    Returns:
        a tuple of new dataframe with renamed column and new column name
    """
    temp_col = orig_col
    if ESCAPE_CHAR_PATTERN.search(orig_col):
        idx = 0
        temp_col = ESCAPE_CHAR_PATTERN.sub("_", orig_col)
        name_set = set(list(df.columns))
        while temp_col in name_set:
            temp_col = f"{temp_col}_{idx}"
            idx += 1
        df = df.withColumnRenamed(orig_col, temp_col)
    return df, temp_col


def spark_operator_with_escaped_column(operator_func, func_args, func_params):
    """Invoke operator func with input dataframe that has its column names sanitized.

    This function rename column names with special char to an internal name and
    rename it back after invocation

    Args:
        operator_func: underlying operator function
        func_args: operator function positional args, this only contains one element `df` for now
        func_params: operator function kwargs

    Returns:
        a dictionary with operator results
    """
    renamed_columns = {}
    input_keys = ["input_column"]

    for input_col_key in input_keys:
        if input_col_key not in func_params:
            continue
        input_col_value = func_params[input_col_key]
        # rename this col if needed
        input_df, temp_col_name = rename_invalid_column(func_args[0], input_col_value)
        func_args[0] = input_df
        if temp_col_name != input_col_value:
            renamed_columns[input_col_value] = temp_col_name
            func_params[input_col_key] = temp_col_name

    # invoke underlying function
    result = operator_func(*func_args, **func_params)

    # put renamed columns back if applicable
    if result is not None and "default" in result:
        result_df = result["default"]
        # rename col
        for orig_col_name, temp_col_name in renamed_columns.items():
            if temp_col_name in result_df.columns:
                result_df = result_df.withColumnRenamed(temp_col_name, orig_col_name)

        result["default"] = result_df

    return result


class OperatorCustomerError(Exception):
    """Error type for Customer Errors in Spark Operators"""


import re


def featurize_text_character_statistics_word_count(text: str) -> float:
    return float(len(text.split()))


def featurize_text_character_statistics_char_count(text: str) -> float:
    return float(len(text))


def featurize_text_character_statistics_special_ratio(text: str) -> float:
    text = re.sub("\\s+", "", text)
    if not text:
        return 0.0
    new_str = re.sub(r"[\w]+", "", text)
    return len(new_str) / len(text)


def featurize_text_character_statistics_digit_ratio(text: str) -> float:
    text = re.sub("\\s+", "", text)
    if not text:
        return 0.0
    return sum(c.isdigit() for c in text) / len(text)


def featurize_text_character_statistics_lower_ratio(text: str) -> float:
    text = re.sub("\\s+", "", text)
    if not text:
        return 0.0
    return sum(c.islower() for c in text) / len(text)


def featurize_text_character_statistics_capital_ratio(text: str) -> float:
    text = re.sub("\\s+", "", text)
    if not text:
        return 0.0
    return sum(c.isupper() for c in text) / len(text)


def featurize_text_character_statistics(df, input_column=None, output_column_prefix=None, trained_parameters=None):
    """
    Extracts syntactic features from a string
    Args:
        df: input pyspark dataframe
        spark: spark session
        column: Input column. This column should contain text data
        output_prefix: The output consists of multiple columns, matching the different extracted features.
            This determines their prefix
    Returns:
        dataframe with columns containing features of the string in the input column
            word_count: number of words
            char_count: string length
            special_ratio: ratio of non alphanumeric characters to non-spaces in the string, 0 if empty string
            digit_ratio: ratio of digits characters to non-spaces in the string, 0 if empty string
            lower_ratio: ratio of lowercase characters to non-spaces in the string, 0 if empty string
            capital_ratio: ratio of uppercase characters to non-spaces in the string, 0 if empty string
    """
    from pyspark.sql.functions import udf
    from pyspark.sql.types import DoubleType

    expects_column(df, input_column, "Input column")
    output_column_prefix = output_column_prefix if output_column_prefix else input_column + "_"

    functions = {
        "word_count": featurize_text_character_statistics_word_count,
        "char_count": featurize_text_character_statistics_char_count,
        "special_ratio": featurize_text_character_statistics_special_ratio,
        "digit_ratio": featurize_text_character_statistics_digit_ratio,
        "lower_ratio": featurize_text_character_statistics_lower_ratio,
        "capital_ratio": featurize_text_character_statistics_capital_ratio,
    }

    # cast input to string
    temp_col = temp_col_name(df, *[output_column_prefix + func_name for func_name in functions.keys()])
    output_df = df.withColumn(temp_col, df[input_column].cast("string"))

    # add the features, one at a time
    for func_name, func in functions.items():
        ufunc = udf(func, returnType=DoubleType())
        output_df = output_df.withColumn(output_column_prefix + "_" + func_name, ufunc(output_df[temp_col]))

    output_df = output_df.drop(temp_col)

    return default_spark(output_df)


def featurize_text_vectorize(
    df,
    input_column=None,
    tokenizer=None,
    tokenizer_standard_parameters=None,
    tokenizer_custom_parameters=None,
    vectorizer=None,
    vectorizer_count_vectorizer_parameters=None,
    vectorizer_hashing_parameters=None,
    apply_idf=None,
    apply_idf_yes_parameters=None,
    apply_idf_no_parameters=None,
    output_format=None,
    output_column=None,
    trained_parameters=None,
):

    TOKENIZER_STANDARD = "Standard"
    TOKENIZER_CUSTOM = "Custom"

    VECTORIZER_COUNT_VECTORIZER = "Count Vectorizer"
    VECTORIZER_HASHING = "Hashing"

    APPLY_IDF_YES = "Yes"
    APPLY_IDF_NO = "No"

    OUTPUT_FORMAT_VECTOR = "Vector"
    OUTPUT_FORMAT_COLUMNS = "Columns"

    import re
    from pyspark.ml.feature import (
        Tokenizer,
        RegexTokenizer,
        HashingTF,
        CountVectorizer,
        CountVectorizerModel,
        IDF,
        IDFModel,
    )
    from pyspark.sql.functions import udf, lit
    from pyspark.sql.types import DoubleType, StringType

    expects_column(df, input_column, "Input column")
    expects_parameter_value_in_list("Tokenizer", tokenizer, [TOKENIZER_STANDARD, TOKENIZER_CUSTOM])
    expects_parameter_value_in_list("Vectorizer", vectorizer, [VECTORIZER_COUNT_VECTORIZER, VECTORIZER_HASHING])
    expects_parameter_value_in_list("Apply IDF", apply_idf, [APPLY_IDF_YES, APPLY_IDF_NO])
    expects_parameter_value_in_list("Output format", output_format, [OUTPUT_FORMAT_VECTOR, OUTPUT_FORMAT_COLUMNS])
    output_column = output_column if output_column else f"{input_column}_features"

    column_type = df.schema[input_column].dataType
    if not isinstance(column_type, StringType):
        raise OperatorSparkOperatorCustomerError(
            f'String column required. Please cast column to a string type first. Column "{input_column}" has type {column_type.simpleString()}.'
        )

    if tokenizer == TOKENIZER_CUSTOM:
        expects_parameter("Custom tokenizer parameters", tokenizer_custom_parameters)

    if vectorizer == VECTORIZER_COUNT_VECTORIZER:
        expects_parameter("Count vectorizer parameters", vectorizer_count_vectorizer_parameters)
    elif vectorizer == VECTORIZER_HASHING:
        expects_parameter("Hashing vectorizer parameters", vectorizer_hashing_parameters)

    if apply_idf == APPLY_IDF_YES:
        expects_parameter("Apply IDF parameters", apply_idf_yes_parameters)

    # raise error if vectorizer is not count_vectorizer
    if output_format == OUTPUT_FORMAT_COLUMNS and vectorizer != VECTORIZER_COUNT_VECTORIZER:
        raise OperatorSparkOperatorCustomerError(
            "To produce column output the vectorizer must be a Count Vectorizer. "
            f"Provided vectorizer is {vectorizer}"
        )

    trained_parameters = load_trained_parameters(
        trained_parameters,
        [
            tokenizer,
            tokenizer_standard_parameters,
            tokenizer_custom_parameters,
            vectorizer,
            vectorizer_count_vectorizer_parameters,
            vectorizer_hashing_parameters,
            apply_idf,
            apply_idf_yes_parameters,
            apply_idf_no_parameters,
        ],
    )

    # fill missing value with empty string
    df = df.fillna("", subset=[input_column])

    # tokenize
    token_col = temp_col_name(df, output_column)
    if tokenizer == TOKENIZER_STANDARD:
        tokenize = Tokenizer(inputCol=input_column, outputCol=token_col)
    else:  # custom
        min_token_length = parse_parameter(
            int, tokenizer_custom_parameters.get("minimum_token_length", 1), "minimum_token_length", 1
        )
        gaps = parse_parameter(bool, tokenizer_custom_parameters.get("gaps", True), "gaps", True)
        pattern = str(tokenizer_custom_parameters["pattern"])
        try:
            re.compile(pattern)
        except re.error:
            raise OperatorSparkOperatorCustomerError(
                f"Invalid regex pattern provided. Expected a legal regular expression but input received is {pattern}"
            )
        to_lower_case = parse_parameter(
            bool,
            tokenizer_custom_parameters.get("lowercase_before_tokenization", True),
            "lowercase_before_tokenization",
            True,
        )

        tokenize = RegexTokenizer(
            inputCol=input_column,
            outputCol=token_col,
            minTokenLength=min_token_length,
            gaps=gaps,
            pattern=pattern,
            toLowercase=to_lower_case,
        )
    with_token = tokenize.transform(df)

    # vectorize
    vector_col = temp_col_name(with_token, output_column)
    if vectorizer == VECTORIZER_HASHING:
        # check for legal inputs
        hashing_num_features = parse_parameter(
            int, vectorizer_hashing_parameters.get("number_of_features", 262144), "number_of_features", 262144
        )
        vectorize = HashingTF(inputCol=token_col, outputCol=vector_col, numFeatures=hashing_num_features)
    else:  # "Count vectorizer"
        min_term_freq = parse_parameter(
            float,
            vectorizer_count_vectorizer_parameters.get("minimum_term_frequency", 1.0),
            "minimum_term_frequency",
            1.0,
        )
        min_doc_freq = parse_parameter(
            float,
            vectorizer_count_vectorizer_parameters.get("minimum_document_frequency", 1.0),
            "minimum_document_frequency",
            1.0,
        )
        max_doc_freq = parse_parameter(
            float,
            vectorizer_count_vectorizer_parameters.get("maximum_document_frequency", 1.0),
            "maximum_document_frequency",
            1.0,
        )
        max_vocab_size = parse_parameter(
            int,
            vectorizer_count_vectorizer_parameters.get("maximum_vocabulary_size", 10000),
            "maximum_vocabulary_size",
            10000,
        )
        binary = parse_parameter(
            bool, vectorizer_count_vectorizer_parameters.get("binarize_count", False), "binarize_count", False
        )

        vectorize, vectorize_model_loaded = load_pyspark_model_from_trained_parameters(
            trained_parameters, CountVectorizerModel, "vectorizer_model"
        )

        if vectorize is None:
            count_vectorizer = CountVectorizer(
                inputCol=token_col,
                outputCol=vector_col,
                minTF=min_term_freq,
                minDF=min_doc_freq,
                maxDF=max_doc_freq,
                vocabSize=max_vocab_size,
                binary=binary,
            )
            vectorize = fit_and_save_model(trained_parameters, "vectorizer_model", count_vectorizer, with_token)

    with_vector = vectorize.transform(with_token).drop(token_col)

    # tf-idf
    if apply_idf == APPLY_IDF_YES:
        # check variables
        min_doc_freq = parse_parameter(
            int, apply_idf_yes_parameters.get("minimum_document_frequency", 1), "minimum_document_frequency", 1
        )

        idf_model, idf_model_loaded = load_pyspark_model_from_trained_parameters(
            trained_parameters, IDFModel, "idf_model"
        )

        if idf_model is None:
            idf = IDF(minDocFreq=min_doc_freq, inputCol=vector_col, outputCol=output_column,)
            idf_model = fit_and_save_model(trained_parameters, "idf_model", idf, with_vector)

        post_idf = idf_model.transform(with_vector)
    else:
        post_idf = with_vector.withColumn(output_column, with_vector[vector_col])

    # flatten output if requested
    if output_format == OUTPUT_FORMAT_COLUMNS:
        index_to_name = vectorize.vocabulary

        def indexing(vec, idx):
            try:
                return float(vec[int(idx)])
            except (IndexError, ValueError):
                return 0.0

        indexing_udf = udf(indexing, returnType=DoubleType())

        names = list(df.columns)
        for col_index, cur_name in enumerate(index_to_name):
            names.append(indexing_udf(output_column, lit(col_index)).alias(f"{output_column}_{cur_name}"))

        output_df = post_idf.select(escape_column_names(names))
    else:
        output_df = post_idf.drop(vector_col)

    return default_spark_with_trained_parameters(output_df, trained_parameters)


from enum import Enum

from pyspark.sql.types import BooleanType, DateType, DoubleType, LongType, StringType
from pyspark.sql import functions as f


class NonCastableDataHandlingMethod(Enum):
    REPLACE_WITH_NULL = "replace_null"
    REPLACE_WITH_NULL_AND_PUT_NON_CASTABLE_DATA_IN_NEW_COLUMN = "replace_null_with_new_col"
    REPLACE_WITH_FIXED_VALUE = "replace_value"
    REPLACE_WITH_FIXED_VALUE_AND_PUT_NON_CASTABLE_DATA_IN_NEW_COLUMN = "replace_value_with_new_col"
    DROP_NON_CASTABLE_ROW = "drop"

    @staticmethod
    def get_names():
        return [item.name for item in NonCastableDataHandlingMethod]

    @staticmethod
    def get_values():
        return [item.value for item in NonCastableDataHandlingMethod]


class MohaveDataType(Enum):
    BOOL = "bool"
    DATE = "date"
    FLOAT = "float"
    LONG = "long"
    STRING = "string"
    OBJECT = "object"

    @staticmethod
    def get_names():
        return [item.name for item in MohaveDataType]

    @staticmethod
    def get_values():
        return [item.value for item in MohaveDataType]


PYTHON_TYPE_MAPPING = {
    MohaveDataType.BOOL: bool,
    MohaveDataType.DATE: str,
    MohaveDataType.FLOAT: float,
    MohaveDataType.LONG: int,
    MohaveDataType.STRING: str,
}

MOHAVE_TO_SPARK_TYPE_MAPPING = {
    MohaveDataType.BOOL: BooleanType,
    MohaveDataType.DATE: DateType,
    MohaveDataType.FLOAT: DoubleType,
    MohaveDataType.LONG: LongType,
    MohaveDataType.STRING: StringType,
}

SPARK_TYPE_MAPPING_TO_SQL_TYPE = {
    BooleanType: "BOOLEAN",
    LongType: "BIGINT",
    DoubleType: "DOUBLE",
    StringType: "STRING",
    DateType: "DATE",
}

SPARK_TO_MOHAVE_TYPE_MAPPING = {value: key for (key, value) in MOHAVE_TO_SPARK_TYPE_MAPPING.items()}


def cast_single_column_type_helper(df, column_name_to_cast, column_name_to_add, mohave_data_type, date_formatting):
    if mohave_data_type == MohaveDataType.DATE:
        df = df.withColumn(column_name_to_add, f.to_date(df[column_name_to_cast], date_formatting))
    else:
        df = df.withColumn(
            column_name_to_add, df[column_name_to_cast].cast(MOHAVE_TO_SPARK_TYPE_MAPPING[mohave_data_type]())
        )
    return df


def cast_single_column_type(
    df, column, mohave_data_type, invalid_data_handling_method, replace_value=None, date_formatting="dd-MM-yyyy"
):
    """Cast single column to a new type

    Args:
        df (DataFrame): spark dataframe
        column (Column): target column for type casting
        mohave_data_type (Enum): Enum MohaveDataType
        invalid_data_handling_method (Enum): Enum NonCastableDataHandlingMethod
        replace_value (str): value to replace for invalid data when "replace_value" is specified
        date_formatting (str): format for date. Default format is "dd-MM-yyyy"

    Returns:
        df (DataFrame): casted spark dataframe
    """
    cast_to_date = f.to_date(df[column], date_formatting)
    cast_to_non_date = df[column].cast(MOHAVE_TO_SPARK_TYPE_MAPPING[mohave_data_type]())
    non_castable_column = f"{column}_typecast_error"
    temp_column = "temp_column"

    if invalid_data_handling_method == NonCastableDataHandlingMethod.REPLACE_WITH_NULL:
        # Replace non-castable data to None in the same column. pyspark's default behaviour
        # Original dataframe
        # +---+------+
        # | id | txt |
        # +---+---+--+
        # | 1 | foo  |
        # | 2 | bar  |
        # | 3 | 1    |
        # +---+------+
        # cast txt column to long
        # +---+------+
        # | id | txt |
        # +---+------+
        # | 1 | None |
        # | 2 | None |
        # | 3 | 1    |
        # +---+------+
        return df.withColumn(column, cast_to_date if (mohave_data_type == MohaveDataType.DATE) else cast_to_non_date)
    if invalid_data_handling_method == NonCastableDataHandlingMethod.DROP_NON_CASTABLE_ROW:
        # Drop non-castable row
        # Original dataframe
        # +---+------+
        # | id | txt |
        # +---+---+--+
        # | 1 | foo  |
        # | 2 | bar  |
        # | 3 | 1    |
        # +---+------+
        # cast txt column to long, _ non-castable row
        # +---+----+
        # | id|txt |
        # +---+----+
        # |  3|  1 |
        # +---+----+
        df = df.withColumn(column, cast_to_date if (mohave_data_type == MohaveDataType.DATE) else cast_to_non_date)
        return df.where(df[column].isNotNull())

    if (
        invalid_data_handling_method
        == NonCastableDataHandlingMethod.REPLACE_WITH_NULL_AND_PUT_NON_CASTABLE_DATA_IN_NEW_COLUMN
    ):
        # Replace non-castable data to None in the same column and put non-castable data to a new column
        # Original dataframe
        # +---+------+
        # | id | txt |
        # +---+------+
        # | 1 | foo  |
        # | 2 | bar  |
        # | 3 | 1    |
        # +---+------+
        # cast txt column to long
        # +---+----+------------------+
        # | id|txt |txt_typecast_error|
        # +---+----+------------------+
        # |  1|None|      foo         |
        # |  2|None|      bar         |
        # |  3|  1 |                  |
        # +---+----+------------------+
        df = df.withColumn(temp_column, cast_to_date if (mohave_data_type == MohaveDataType.DATE) else cast_to_non_date)
        df = df.withColumn(non_castable_column, f.when(df[temp_column].isNotNull(), "").otherwise(df[column]),)
    elif invalid_data_handling_method == NonCastableDataHandlingMethod.REPLACE_WITH_FIXED_VALUE:
        # Replace non-castable data to a value in the same column
        # Original dataframe
        # +---+------+
        # | id | txt |
        # +---+------+
        # | 1 | foo  |
        # | 2 | bar  |
        # | 3 | 1    |
        # +---+------+
        # cast txt column to long, replace non-castable value to 0
        # +---+-----+
        # | id| txt |
        # +---+-----+
        # |  1|  0  |
        # |  2|  0  |
        # |  3|  1  |
        # +---+----+
        value = _validate_and_cast_value(value=replace_value, mohave_data_type=mohave_data_type)

        df = df.withColumn(temp_column, cast_to_date if (mohave_data_type == MohaveDataType.DATE) else cast_to_non_date)

        replace_date_value = f.when(df[temp_column].isNotNull(), df[temp_column]).otherwise(
            f.to_date(f.lit(value), date_formatting)
        )
        replace_non_date_value = f.when(df[temp_column].isNotNull(), df[temp_column]).otherwise(value)

        df = df.withColumn(
            temp_column, replace_date_value if (mohave_data_type == MohaveDataType.DATE) else replace_non_date_value
        )
    elif (
        invalid_data_handling_method
        == NonCastableDataHandlingMethod.REPLACE_WITH_FIXED_VALUE_AND_PUT_NON_CASTABLE_DATA_IN_NEW_COLUMN
    ):
        # Replace non-castable data to a value in the same column and put non-castable data to a new column
        # Original dataframe
        # +---+------+
        # | id | txt |
        # +---+---+--+
        # | 1 | foo  |
        # | 2 | bar  |
        # | 3 | 1    |
        # +---+------+
        # cast txt column to long, replace non-castable value to 0
        # +---+----+------------------+
        # | id|txt |txt_typecast_error|
        # +---+----+------------------+
        # |  1|  0  |   foo           |
        # |  2|  0  |   bar           |
        # |  3|  1  |                 |
        # +---+----+------------------+
        value = _validate_and_cast_value(value=replace_value, mohave_data_type=mohave_data_type)

        df = df.withColumn(temp_column, cast_to_date if (mohave_data_type == MohaveDataType.DATE) else cast_to_non_date)
        df = df.withColumn(non_castable_column, f.when(df[temp_column].isNotNull(), "").otherwise(df[column]),)

        replace_date_value = f.when(df[temp_column].isNotNull(), df[temp_column]).otherwise(
            f.to_date(f.lit(value), date_formatting)
        )
        replace_non_date_value = f.when(df[temp_column].isNotNull(), df[temp_column]).otherwise(value)

        df = df.withColumn(
            temp_column, replace_date_value if (mohave_data_type == MohaveDataType.DATE) else replace_non_date_value
        )
    # drop temporary column
    df = df.withColumn(column, df[temp_column]).drop(temp_column)

    df_cols = df.columns
    if non_castable_column in df_cols:
        # Arrange columns so that non_castable_column col is next to casted column
        df_cols.remove(non_castable_column)
        column_index = df_cols.index(column)
        arranged_cols = df_cols[: column_index + 1] + [non_castable_column] + df_cols[column_index + 1 :]
        df = df.select(*arranged_cols)
    return df


def _validate_and_cast_value(value, mohave_data_type):
    if value is None:
        return value
    try:
        return PYTHON_TYPE_MAPPING[mohave_data_type](value)
    except ValueError as e:
        raise ValueError(
            f"Invalid value to replace non-castable data. "
            f"{mohave_data_type} is not in mohave supported date type: {MohaveDataType.get_values()}. "
            f"Please use a supported type",
            e,
        )


import os
import collections
import tempfile
import zipfile
import base64
import logging
from io import BytesIO
import numpy as np
import re

from pyspark.sql import Column


class OperatorSparkOperatorCustomerError(Exception):
    """Error type for Customer Errors in Spark Operators"""


def temp_col_name(df, *illegal_names):
    """Generates a temporary column name that is unused.
    """
    name = "temp_col"
    idx = 0
    name_set = set(list(df.columns) + list(illegal_names))
    while name in name_set:
        name = f"_temp_col_{idx}"
        idx += 1

    return name


def get_temp_col_if_not_set(df, col_name):
    """Extracts the column name from the parameters if it exists, otherwise generates a temporary column name.
    """
    if col_name:
        return col_name, False
    else:
        return temp_col_name(df), True


def replace_input_if_output_is_temp(df, input_column, output_column, output_is_temp):
    """Replaces the input column in the dataframe if the output was not set

    This is used with get_temp_col_if_not_set to enable the behavior where a 
    transformer will replace its input column if an output is not specified.
    """
    if output_is_temp:
        df = df.withColumn(input_column, df[output_column])
        df = df.drop(output_column)
        return df
    else:
        return df


def parse_parameter(typ, value, key, default=None, nullable=False):
    if value is None:
        if default is not None or nullable:
            return default
        else:
            raise OperatorSparkOperatorCustomerError(f"Missing required input: '{key}'")
    else:
        try:
            value = typ(value)
            if isinstance(value, (int, float, complex)) and not isinstance(value, bool):
                if np.isnan(value) or np.isinf(value):
                    raise OperatorSparkOperatorCustomerError(
                        f"Invalid value provided for '{key}'. Expected {typ.__name__} but received: {value}"
                    )
                else:
                    return value
            else:
                return value
        except (ValueError, TypeError):
            raise OperatorSparkOperatorCustomerError(
                f"Invalid value provided for '{key}'. Expected {typ.__name__} but received: {value}"
            )
        except OverflowError:
            raise OperatorSparkOperatorCustomerError(
                f"Overflow Error: Invalid value provided for '{key}'. Given value '{value}' exceeds the range of type "
                f"'{typ.__name__}' for this input. Insert a valid value for type '{typ.__name__}' and try your request "
                f"again."
            )


def expects_valid_column_name(value, key, nullable=False):
    if nullable and value is None:
        return

    if value is None or len(str(value).strip()) == 0:
        raise OperatorSparkOperatorCustomerError(f"Column name cannot be null, empty, or whitespace for parameter '{key}': {value}")


def expects_parameter(value, key, condition=None):
    if value is None:
        raise OperatorSparkOperatorCustomerError(f"Missing required input: '{key}'")
    elif condition is not None and not condition:
        raise OperatorSparkOperatorCustomerError(f"Invalid value provided for '{key}': {value}")


def expects_column(df, value, key):
    if not value or value not in df.columns:
        raise OperatorSparkOperatorCustomerError(f"Expected column in dataframe for '{key}' however received '{value}'")


def expects_parameter_value_in_list(key, value, items):
    if value not in items:
        raise OperatorSparkOperatorCustomerError(f"Illegal parameter value. {key} expected to be in {items}, but given {value}")


def encode_pyspark_model(model):
    with tempfile.TemporaryDirectory() as dirpath:
        dirpath = os.path.join(dirpath, "model")
        # Save the model
        model.save(dirpath)

        # Create the temporary zip-file.
        mem_zip = BytesIO()
        with zipfile.ZipFile(mem_zip, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as zf:
            # Zip the directory.
            for root, dirs, files in os.walk(dirpath):
                for file in files:
                    rel_dir = os.path.relpath(root, dirpath)
                    zf.write(os.path.join(root, file), os.path.join(rel_dir, file))

        zipped = mem_zip.getvalue()
        encoded = base64.b85encode(zipped)
        return str(encoded, "utf-8")


def decode_pyspark_model(model_factory, encoded):
    with tempfile.TemporaryDirectory() as dirpath:
        zip_bytes = base64.b85decode(encoded)
        mem_zip = BytesIO(zip_bytes)
        mem_zip.seek(0)

        with zipfile.ZipFile(mem_zip, "r") as zf:
            zf.extractall(dirpath)

        model = model_factory.load(dirpath)
        return model


def hash_parameters(value):
    # pylint: disable=W0702
    try:
        if isinstance(value, collections.Hashable):
            return hash(value)
        if isinstance(value, dict):
            return hash(frozenset([hash((hash_parameters(k), hash_parameters(v))) for k, v in value.items()]))
        if isinstance(value, list):
            return hash(frozenset([hash_parameters(v) for v in value]))
        raise RuntimeError("Object not supported for serialization")
    except:  # noqa: E722
        raise RuntimeError("Object not supported for serialization")


def load_trained_parameters(trained_parameters, operator_parameters):
    trained_parameters = trained_parameters if trained_parameters else {}
    parameters_hash = hash_parameters(operator_parameters)
    stored_hash = trained_parameters.get("_hash")
    if stored_hash != parameters_hash:
        trained_parameters = {"_hash": parameters_hash}
    return trained_parameters


def load_pyspark_model_from_trained_parameters(trained_parameters, model_factory, name):
    if trained_parameters is None or name not in trained_parameters:
        return None, False

    try:
        model = decode_pyspark_model(model_factory, trained_parameters[name])
        return model, True
    except Exception as e:
        logging.error(f"Could not decode PySpark model {name} from trained_parameters: {e}")
        del trained_parameters[name]
        return None, False


def fit_and_save_model(trained_parameters, name, algorithm, df):
    model = algorithm.fit(df)
    trained_parameters[name] = encode_pyspark_model(model)
    return model


def transform_using_trained_model(model, df, loaded):
    try:
        return model.transform(df)
    except Exception as e:
        if loaded:
            raise OperatorSparkOperatorCustomerError(
                f"Encountered error while using stored model. Please delete the operator and try again. {e}"
            )
        else:
            raise e


ESCAPE_CHAR_PATTERN = re.compile("[{}]+".format(re.escape(".`")))


def escape_column_name(col):
    """Escape column name so it works properly for Spark SQL"""

    # Do nothing for Column object, which should be already valid/quoted
    if isinstance(col, Column):
        return col

    column_name = col

    if ESCAPE_CHAR_PATTERN.search(column_name):
        column_name = f"`{column_name}`"

    return column_name


def rename_invalid_column(df, orig_col):
    """Rename a given column in a data frame to a new valid name

    Args:
        df: Spark dataframe
        orig_col: input column name

    Returns:
        a tuple of new dataframe with renamed column and new column name
    """
    temp_col = orig_col
    if ESCAPE_CHAR_PATTERN.search(orig_col):
        idx = 0
        temp_col = ESCAPE_CHAR_PATTERN.sub("_", orig_col)
        name_set = set(list(df.columns))
        while temp_col in name_set:
            temp_col = f"{temp_col}_{idx}"
            idx += 1
        df = df.withColumnRenamed(orig_col, temp_col)
    return df, temp_col


def escape_column_names(columns):
    return [escape_column_name(col) for col in columns]


def sanitize_df(df):
    """Sanitize dataframe with Spark safe column names and return column name mappings

    Args:
        df: input dataframe

    Returns:
        a tuple of
            sanitized_df: sanitized dataframe with all Spark safe columns
            sanitized_col_mapping: mapping from original col name to sanitized column name
            reversed_col_mapping: reverse mapping from sanitized column name to original col name
    """

    sanitized_col_mapping = {}
    sanitized_df = df

    for orig_col in df.columns:
        if ESCAPE_CHAR_PATTERN.search(orig_col):
            # create a temp column and store the column name mapping
            temp_col = f"{orig_col.replace('.', '_')}_{temp_col_name(sanitized_df)}"
            sanitized_col_mapping[orig_col] = temp_col

            sanitized_df = sanitized_df.withColumn(temp_col, sanitized_df[f"`{orig_col}`"])
            sanitized_df = sanitized_df.drop(orig_col)

    # create a reversed mapping from sanitized col names to original col names
    reversed_col_mapping = {sanitized_name: orig_name for orig_name, sanitized_name in sanitized_col_mapping.items()}

    return sanitized_df, sanitized_col_mapping, reversed_col_mapping


import re
from datetime import date

import numpy as np
import pandas as pd
from pyspark.sql.types import (
    BooleanType,
    IntegralType,
    FractionalType,
    StringType,
)



def type_inference(df):  # noqa: C901 # pylint: disable=R0912
    """Core type inference logic

    Args:
        df: spark dataframe

    Returns: dict a schema that maps from column name to mohave datatype

    """
    columns_to_infer = [col for (col, col_type) in df.dtypes if col_type == "string"]

    pandas_df = df.toPandas()
    report = {}
    for (columnName, _) in pandas_df.iteritems():
        if columnName in columns_to_infer:
            column = pandas_df[columnName].values
            report[columnName] = {
                "sum_string": len(column),
                "sum_numeric": sum_is_numeric(column),
                "sum_integer": sum_is_integer(column),
                "sum_boolean": sum_is_boolean(column),
                "sum_date": sum_is_date(column),
                "sum_null_like": sum_is_null_like(column),
                "sum_null": sum_is_null(column),
            }

    # Analyze
    numeric_threshold = 0.8
    integer_threshold = 0.8
    date_threshold = 0.8
    bool_threshold = 0.8

    column_types = {}

    for col, insights in report.items():
        # Convert all columns to floats to make thresholds easy to calculate.
        proposed = MohaveDataType.STRING.value
        if (insights["sum_numeric"] / insights["sum_string"]) > numeric_threshold:
            proposed = MohaveDataType.FLOAT.value
            if (insights["sum_integer"] / insights["sum_numeric"]) > integer_threshold:
                proposed = MohaveDataType.LONG.value
        elif (insights["sum_boolean"] / insights["sum_string"]) > bool_threshold:
            proposed = MohaveDataType.BOOL.value
        elif (insights["sum_date"] / insights["sum_string"]) > date_threshold:
            proposed = MohaveDataType.DATE.value
        column_types[col] = proposed

    for f in df.schema.fields:
        if f.name not in columns_to_infer:
            if isinstance(f.dataType, IntegralType):
                column_types[f.name] = MohaveDataType.LONG.value
            elif isinstance(f.dataType, FractionalType):
                column_types[f.name] = MohaveDataType.FLOAT.value
            elif isinstance(f.dataType, StringType):
                column_types[f.name] = MohaveDataType.STRING.value
            elif isinstance(f.dataType, BooleanType):
                column_types[f.name] = MohaveDataType.BOOL.value
            else:
                # unsupported types in mohave
                column_types[f.name] = MohaveDataType.OBJECT.value

    return column_types


def _is_numeric_single(x):
    try:
        x_float = float(x)
        return np.isfinite(x_float)
    except ValueError:
        return False
    except TypeError:  # if x = None
        return False


def sum_is_numeric(x):
    """count number of numeric element

    Args:
        x: numpy array

    Returns: int

    """
    castables = np.vectorize(_is_numeric_single)(x)
    return np.count_nonzero(castables)


def _is_integer_single(x):
    try:
        if not _is_numeric_single(x):
            return False
        return float(x) == int(x)
    except ValueError:
        return False
    except TypeError:  # if x = None
        return False


def sum_is_integer(x):
    castables = np.vectorize(_is_integer_single)(x)
    return np.count_nonzero(castables)


def _is_boolean_single(x):
    boolean_list = ["true", "false"]
    try:
        is_boolean = x.lower() in boolean_list
        return is_boolean
    except ValueError:
        return False
    except TypeError:  # if x = None
        return False
    except AttributeError:
        return False


def sum_is_boolean(x):
    castables = np.vectorize(_is_boolean_single)(x)
    return np.count_nonzero(castables)


def sum_is_null_like(x):  # noqa: C901
    def _is_empty_single(x):
        try:
            return bool(len(x) == 0)
        except TypeError:
            return False

    def _is_null_like_single(x):
        try:
            return bool(null_like_regex.match(x))
        except TypeError:
            return False

    def _is_whitespace_like_single(x):
        try:
            return bool(whitespace_regex.match(x))
        except TypeError:
            return False

    null_like_regex = re.compile(r"(?i)(null|none|nil|na|nan)")  # (?i) = case insensitive
    whitespace_regex = re.compile(r"^\s+$")  # only whitespace

    empty_checker = np.vectorize(_is_empty_single)(x)
    num_is_null_like = np.count_nonzero(empty_checker)

    null_like_checker = np.vectorize(_is_null_like_single)(x)
    num_is_null_like += np.count_nonzero(null_like_checker)

    whitespace_checker = np.vectorize(_is_whitespace_like_single)(x)
    num_is_null_like += np.count_nonzero(whitespace_checker)
    return num_is_null_like


def sum_is_null(x):
    return np.count_nonzero(pd.isnull(x))


def _is_date_single(x):
    try:
        return bool(date.fromisoformat(x))  # YYYY-MM-DD
    except ValueError:
        return False
    except TypeError:
        return False


def sum_is_date(x):
    return np.count_nonzero(np.vectorize(_is_date_single)(x))


def cast_df(df, schema):
    """Cast datafram from given schema

    Args:
        df: spark dataframe
        schema: schema to cast to. It map from df's col_name to mohave datatype

    Returns: casted dataframe

    """
    # col name to spark data type mapping
    col_to_spark_data_type_map = {}

    # get spark dataframe's actual datatype
    fields = df.schema.fields
    for f in fields:
        col_to_spark_data_type_map[f.name] = f.dataType
    cast_expr = []
    # iterate given schema and cast spark dataframe datatype
    for col_name in schema:
        mohave_data_type_from_schema = MohaveDataType(schema.get(col_name, MohaveDataType.OBJECT.value))
        if mohave_data_type_from_schema != MohaveDataType.OBJECT:
            spark_data_type_from_schema = MOHAVE_TO_SPARK_TYPE_MAPPING[mohave_data_type_from_schema]
            # Only cast column when the data type in schema doesn't match the actual data type
            if not isinstance(col_to_spark_data_type_map[col_name], spark_data_type_from_schema):
                # use spark-sql expression instead of spark.withColumn to improve performance
                expr = f"CAST (`{col_name}` as {SPARK_TYPE_MAPPING_TO_SQL_TYPE[spark_data_type_from_schema]})"
            else:
                # include column that has same dataType as it is
                expr = f"`{col_name}`"
        else:
            # include column that has same mohave object dataType as it is
            expr = f"`{col_name}`"
        cast_expr.append(expr)
    if len(cast_expr) != 0:
        df = df.selectExpr(*cast_expr)
    return df, schema


def validate_schema(df, schema):
    """Validate if every column is covered in the schema

    Args:
        schema ():
    """
    columns_in_df = df.columns
    columns_in_schema = schema.keys()

    if len(columns_in_df) != len(columns_in_schema):
        raise ValueError(
            f"Invalid schema column size. "
            f"Number of columns in schema should be equal as number of columns in dataframe. "
            f"schema columns size: {len(columns_in_schema)}, dataframe column size: {len(columns_in_df)}"
        )

    for col in columns_in_schema:
        if col not in columns_in_df:
            raise ValueError(
                f"Invalid column name in schema. "
                f"Column in schema does not exist in dataframe. "
                f"Non-existed columns: {col}"
            )


def s3_source(spark, mode, dataset_definition):
    """Represents a source that handles sampling, etc."""

    content_type = dataset_definition["s3ExecutionContext"]["s3ContentType"].upper()
    path = dataset_definition["s3ExecutionContext"]["s3Uri"].replace("s3://", "s3a://")

    try:
        if content_type == "CSV":
            has_header = dataset_definition["s3ExecutionContext"]["s3HasHeader"]
            field_delimiter = dataset_definition["s3ExecutionContext"].get("s3FieldDelimiter", ",")
            if not field_delimiter:
                field_delimiter = ","
            df = spark.read.csv(path=path, header=has_header, escape='"', quote='"', sep=field_delimiter)
        elif content_type == "PARQUET":
            df = spark.read.parquet(path)

        return default_spark(df)
    except Exception as e:
        raise RuntimeError("An error occurred while reading files from S3") from e


def infer_and_cast_type(df, spark, inference_data_sample_size=1000, trained_parameters=None):
    """Infer column types for spark dataframe and cast to inferred data type.

    Args:
        df: spark dataframe
        spark: spark session
        inference_data_sample_size: number of row data used for type inference
        trained_parameters: trained_parameters to determine if we need infer data types

    Returns: a dict of pyspark df with column data type casted and trained parameters

    """
    from pyspark.sql.utils import AnalysisException

    # if trained_parameters is none or doesn't contain schema key, then type inference is needed
    if trained_parameters is None or not trained_parameters.get("schema", None):
        # limit first 1000 rows to do type inference

        limit_df = df.limit(inference_data_sample_size)
        schema = type_inference(limit_df)
    else:
        schema = trained_parameters["schema"]
        try:
            validate_schema(df, schema)
        except ValueError as e:
            raise OperatorCustomerError(e)
    try:
        df, schema = cast_df(df, schema)
    except (AnalysisException, ValueError) as e:
        raise OperatorCustomerError(e)
    trained_parameters = {"schema": schema}
    return default_spark_with_trained_parameters(df, trained_parameters)


def concatenate_datasets(
    df1,
    df2,
    spark,
    concatenate_type,
    apply_dedupe=False,
    indicator_col_name=None,
    df1_indicator=None,
    df2_indicator=None,
):
    """ Concatenate two datasets into one.
    If the apply_dedupe option is true then we remove duplicates after merging two datasets.
    If the indicator_col_name is present then we append the indicator col, to the merged datasets,
    so that customer can identify each row belongs to which dataset.

    Args:
        df1: first dataframe
        df2: second dataframe
        spark: spark context
        concatenate_type: concatenate type either row_wise or column_wise
        apply_dedupe: boolean option to apply_dedupe
        indicator_col_name: name of new indicator col
        df1_indicator: indicator string to be used for first dataset
        df2_indicator: indicator string to be sued for second dataset

    Returns: concatenated dataframe, with optionally adding the indicator col or removing the duplicates

    """
    import pyspark.sql.functions as sf

    if concatenate_type == "column_wise":
        df1 = get_dataframe_with_sequence_ids(df1)
        df2 = get_dataframe_with_sequence_ids(df2)
        output_df = df1.join(df2, "_seq_id_", "outer").drop("_seq_id_")
    elif concatenate_type == "row_wise":
        if indicator_col_name:
            df1 = df1.withColumn(indicator_col_name, sf.lit(df1_indicator))
            df2 = df2.withColumn(indicator_col_name, sf.lit(df2_indicator))

        output_df = df1.unionByName(df2)

        if apply_dedupe:
            subset_cols = [col for col in output_df.columns if indicator_col_name != col]
            output_df = output_df.dropDuplicates(subset_cols)
    else:
        raise OperatorCustomerError(
            "Invalid value for Concatenate type. " "'concatenate_type' can either be 'row_wise' or 'column_wise' "
        )

    return default_spark(output_df)


def featurize_text(df, spark, **kwargs):

    return dispatch(
        "operator",
        [df],
        kwargs,
        {
            "Character statistics": (featurize_text_character_statistics, "character_statistics_parameters"),
            "Vectorize": (featurize_text_vectorize, "vectorize_parameters"),
        },
    )


op_1_output = s3_source(spark=spark, mode=mode, **{'dataset_definition': {'__typename': 'S3CreateDatasetDefinitionOutput', 'datasetSourceType': 'S3', 'name': 'part-00000-a8b31401-ec84-4d12-bbbb-817ef69f861e-c000.snappy.parquet', 'description': None, 's3ExecutionContext': {'__typename': 'S3ExecutionContext', 's3Uri': 's3://githubmachinelearningstack-rawdatabucket79e6ae92-dvgbsz21ce9v/data/part-00000-a8b31401-ec84-4d12-bbbb-817ef69f861e-c000.snappy.parquet', 's3ContentType': 'parquet', 's3HasHeader': True, 's3FieldDelimiter': ','}}})
op_2_output = infer_and_cast_type(op_1_output['default'], spark=spark, **{})
op_3_output = s3_source(spark=spark, mode=mode, **{'dataset_definition': {'__typename': 'S3CreateDatasetDefinitionOutput', 'datasetSourceType': 'S3', 'name': 'part-00001-a8b31401-ec84-4d12-bbbb-817ef69f861e-c000.snappy.parquet', 'description': None, 's3ExecutionContext': {'__typename': 'S3ExecutionContext', 's3Uri': 's3://githubmachinelearningstack-rawdatabucket79e6ae92-dvgbsz21ce9v/data/part-00001-a8b31401-ec84-4d12-bbbb-817ef69f861e-c000.snappy.parquet', 's3ContentType': 'parquet', 's3HasHeader': True, 's3FieldDelimiter': ','}}})
op_4_output = infer_and_cast_type(op_3_output['default'], spark=spark, **{})
op_5_output = concatenate_datasets(op_2_output['default'], op_4_output['default'], spark=spark, **{'concatenate_type': 'row_wise', 'apply_dedupe': False})
op_6_output = featurize_text(op_5_output['default'], spark=spark, **{'operator': 'Vectorize', 'vectorize_parameters': {'tokenizer': 'Standard', 'vectorizer': 'Count Vectorizer', 'apply_idf': 'Yes', 'output_format': 'Vector', 'tokenizer_standard_parameters': {}, 'vectorizer_count_vectorizer_parameters': {'minimum_term_frequency': 1, 'minimum_document_frequency': 1, 'maximum_document_frequency': 0.999, 'maximum_vocabulary_size': 262144, 'binarize_count': True}, 'apply_idf_yes_parameters': {'minimum_document_frequency': 5}, 'input_column': 'bodyText', 'output_column': 'tfIdfVector'}, 'character_statistics_parameters': {}})

#  Glossary: variable name to node_id
#
#  op_1_output: 01f04711-0ae4-493f-bf68-110a41188e9e
#  op_2_output: dc0e6751-775e-4075-8800-f53b5baf00e6
#  op_3_output: 9cc25a9b-4d3f-4ae1-b900-691e043550f7
#  op_4_output: cc4052af-6188-4103-989a-733949a0d532
#  op_5_output: 1c9fd857-3fd8-4ea9-b3b0-7500af4b0b5d
#  op_6_output: 3593e31a-c2c1-4dab-9b4c-ec86cd0e71a6