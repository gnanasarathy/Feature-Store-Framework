from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer
from pyspark.sql import DataFrame

def apply_label_encoding_inplace(df: DataFrame, columns: list[str]) -> DataFrame:
    """
    Applies label encoding in-place for multiple string columns in a PySpark DataFrame.

    Args:
        df (DataFrame): Input Spark DataFrame.
        columns (list[str]): List of column names to encode.

    Returns:
        DataFrame: DataFrame with the original columns replaced by label-encoded values.
    """
    indexers = [
        StringIndexer(inputCol=col, outputCol=f"{col}_indexed", handleInvalid="keep")
        for col in columns
    ]

    pipeline = Pipeline(stages=indexers)
    model = pipeline.fit(df)
    encoded_df = model.transform(df)

    # Drop original columns and rename the indexed ones
    for col in columns:
        encoded_df = encoded_df.drop(col)
        encoded_df = encoded_df.withColumnRenamed(f"{col}_indexed", col)

    return encoded_df