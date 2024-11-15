import polars as pl

class DataProcessor:
    @staticmethod
    def combine_df(new_runs_df: pl.DataFrame, old_runs_df: pl.DataFrame) -> pl.DataFrame:
        if old_runs_df.is_empty():
            all_runs_df = new_runs_df.clone()
        else:
            columns = new_runs_df.columns

            for col in columns:
                if col not in old_runs_df.columns:
                    old_runs_df = old_runs_df.with_columns(pl.lit(None).alias(col))

            old_runs_df = old_runs_df.select(columns)

            all_runs_df = (
                pl.concat((new_runs_df.pipe(DataProcessor.set_schema), old_runs_df.pipe(DataProcessor.set_schema)))
                .sort(["logged_at"], descending=True)
                .unique(["date", "company_name", "project", "run_id"], keep="first")
                .sort(["run_id", "project"])
                .sort(["date"], descending=True)
                .sort(["company_name"])
            )
        return all_runs_df

    @staticmethod
    def set_schema(df: pl.DataFrame) -> pl.DataFrame:
        """Dataframeのdata型をcastする"""
        try:
            new_runs_df = df.with_columns(
                pl.col("run_id").cast(pl.Utf8),
                pl.col("duration_hour").cast(pl.Float64),
                pl.col("gpu_count").cast(pl.Int64),
                pl.col("cpu_count").cast(pl.Int64),
                pl.col("average_gpu_utilization").cast(pl.Float64),
                pl.col("average_gpu_memory").cast(pl.Float64),
                pl.col("max_gpu_utilization").cast(pl.Float64),
                pl.col("max_gpu_memory").cast(pl.Float64),
            )
            return new_runs_df
        except:
            print("!!! Failed to cast data type !!!")
            return pl.DataFrame()
