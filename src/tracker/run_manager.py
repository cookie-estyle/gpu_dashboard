import wandb
import pytz
import re
import json
import time
import ast
import gc
import threading
from functools import wraps
from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError
import datetime as dt
import polars as pl
from fnmatch import fnmatch
from dataclasses import dataclass, field
from tqdm import tqdm
from easydict import EasyDict
from typing import List
from wandb_gql import gql

from src.tracker.config_parser import parse_configs
from src.utils.config import CONFIG

JAPAN_TIMEZONE = pytz.timezone("Asia/Tokyo")
LOGGED_AT = dt.datetime.now(JAPAN_TIMEZONE).replace(tzinfo=None)
JAPAN_UTC_OFFSET = 9

GQL_QUERY = """
query GetGpuInfoForProject($project: String!, $entity: String!, $first: Int!, $cursor: String!) {
    project(name: $project, entityName: $entity) {
        name
        runs(first: $first, after: $cursor) {
            edges {
                cursor
                node {
                    name
                    createdAt
                    updatedAt
                    heartbeatAt
                    state
                    tags
                    host
                    runInfo {
                        gpuCount
                        gpu
                    }
                    config
                }
            }
        }
    }
}
"""

@dataclass
class Run:
    run_path: str
    created_at: dt.datetime
    updated_at: dt.datetime
    state: str
    tags: list[str]
    host_name: str
    gpu_name: str
    gpu_count: int
    metrics_df: pl.DataFrame = pl.DataFrame()

@dataclass
class Project:
    project: str
    runs: list[Run] = field(default_factory=list)

def timeout(seconds):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            result = [TimeoutError('Function call timed out')]
            def target():
                try:
                    result[0] = func(*args, **kwargs)
                except Exception as e:
                    result[0] = e

            timer = threading.Timer(seconds, lambda: result.append(None))
            timer.start()
            try:
                target()
            finally:
                timer.cancel()

            if len(result) > 1:
                raise TimeoutError('Function call timed out')
            if isinstance(result[0], Exception):
                raise result[0]
            return result[0]
        return wrapper
    return decorator

class RunManager:
    def __init__(self, date_range: List, test_mode: bool = False):
        self.team_configs = parse_configs(CONFIG)
        self.start_date = dt.datetime.strptime(date_range[0], "%Y-%m-%d").date()
        self.end_date = dt.datetime.strptime(date_range[1], "%Y-%m-%d").date()
        self.api = wandb.Api(timeout=60)
        self.test_mode = test_mode
        self.total_valid_runs = 0
    
    def fetch_runs(self):
        self.__get_projects()
        self.__get_runs()
        self.__get_metrics()
        combined_df = self.__combined_run_df()
        return combined_df
    
    def __get_projects(self):
        for team_config in self.team_configs:
            if (self.end_date >= team_config.start_date) and (self.start_date <= team_config.end_date):
                projects = []
                ignore = team_config.ignore_project_pattern
                try:
                    for project in self.api.projects(team_config.team):
                        if ignore is None or not fnmatch(project.name, ignore):
                            projects.append(Project(project=project.name))
                    team_config.projects = projects
                except Exception as e:
                    print(f"Error fetching projects for {team_config.team}: {str(e)}")
                    team_config.projects = []
            else:
                team_config.projects = []
    
    def __get_runs(self):
        for team_config in self.team_configs:
            print(f"Get runs for {team_config.team} ...")
            for i, project in enumerate(team_config.projects):
                print(f"Processing project {i+1}/{len(team_config.projects)}: {project.project}")
                project.runs = self.__query_runs(
                    team=team_config.team,
                    project=project.project,
                    )
        print(f"\nTotal valid runs across all projects: {self.total_valid_runs}")
    
    def __get_metrics(self):
        print("Get metrics for each run ...")
        for team_config in self.team_configs:
            print(f"Processing team: {team_config.team}")
            for project in tqdm(team_config.projects, desc="Projects"):
                print(f"Processing project: {project.project}")
                if project.runs:
                    self.__process_project_runs(project)
                else:
                    print(f"Skipping project {project.project} as it has no runs.")
                gc.collect()
    
    def __combined_run_df(self):
        print("Create combined run DataFrame ...")
        combined_df = pl.DataFrame()
        for team_config in tqdm(self.team_configs, desc="Processing teams"):
            print(f"  Team: {team_config.team}")
            if not team_config.projects:
                print(f"  Skipping team {team_config.team} as it has no projects.")
                continue
            for project in team_config.projects:
                if not hasattr(project, 'runs') or not project.runs:
                    print(f"  Skipping project {project.project} as it has no runs.")
                    continue
                print(f"  Processing {len(project.runs)} runs for project {project.project}")
                for run in project.runs:
                    try:
                        new_run_df = self.__create_run_df(run)
                        if not new_run_df.is_empty():
                            combined_df = pl.concat([combined_df, new_run_df])
                        else:
                            print(f"Warning: Empty DataFrame created for run {run.run_path}")
                    except Exception as e:
                        print(f"Error processing run {run.run_path}: {str(e)}")
                        print(f"Run details: created_at={run.created_at}, updated_at={run.updated_at}, state={run.state}")
                gc.collect()
        
        if not combined_df.is_empty():
            print(f"Total runs processed: {len(combined_df)}")
            return combined_df
        else:
            print("Warning: No valid DataFrames were created.")
            return pl.DataFrame()
    
    def __query_runs(self, team: str, project: str) -> list[Run]:
        cursor = ""
        nodes = []
        total_processed = 0

        print(f"Starting to query runs for {team}/{project}")

        while True:
            try:
                results = self.api.client.execute(
                    gql(GQL_QUERY),
                    {
                        "entity": team,
                        "project": project,
                        "first": 1000,
                        "cursor": cursor,
                    },
                )
                _edges = results["project"]["runs"]["edges"]
                if not _edges:
                    return self.__process_nodes(nodes, team, project)
                new_nodes = [EasyDict(e["node"]) for e in _edges]
                nodes += new_nodes
                total_processed += len(new_nodes)
                print(f"Processed {len(new_nodes)} runs for {team}/{project}. Total processed: {total_processed}")
                cursor = _edges[-1]["cursor"]
            except Exception as e:
                print(f"Failed to execute query for {team}/{project}")
                print(f"Error details: {str(e)}")
                return self.__process_nodes(nodes, team, project)
    
    def __process_nodes(self, nodes: List[EasyDict], team: str, project: str) -> List[Run]:
        runs = []
        for node in nodes:
            createdAt = dt.datetime.fromisoformat(node.createdAt.rstrip('Z')) + dt.timedelta(hours=JAPAN_UTC_OFFSET)
            updatedAt = dt.datetime.fromisoformat(node.heartbeatAt.rstrip('Z')) + dt.timedelta(hours=JAPAN_UTC_OFFSET)

            if self.__is_run_valid(node, createdAt, updatedAt) or self.test_mode:
                run_path = "/".join((team, project, node.name))
                gpu_count = self.__set_gpucount(node, team)
                run = Run(
                    run_path=run_path,
                    updated_at=updatedAt,
                    created_at=createdAt,
                    state=node.state,
                    tags=node.tags,
                    host_name=node.host,
                    gpu_name=node.runInfo.gpu if node.runInfo else None,
                    gpu_count=gpu_count,
                )
                runs.append(run)
        self.total_valid_runs += len(runs)
        print(f"Total valid runs for {team}/{project}: {len(runs)}")
        return runs

    def __is_run_valid(self, node, createdAt, updatedAt) -> bool:
        # 必要な情報が含まれていないものはスキップ
        if not node.get("runInfo"):
            return False
        if not node.get("runInfo").get("gpu"):
            return False
        
        # 特定のtagをスキップ
        if not set(CONFIG.ignore_tags).isdisjoint([t.lower() for t in node.tags]):
            return False

        # 実行時間が短いものはスキップ
        if createdAt.timestamp() == updatedAt.timestamp():
            return False

        # 指定期間内にランが実行されていたかチェック
        if self.start_date and self.end_date:
            # ランの期間と指定期間に重なりがあるかチェック
            if updatedAt.date() < self.start_date or createdAt.date() > self.end_date:
                return False
        elif self.start_date:
            # ランの終了が start_date 以降かチェック
            if updatedAt.date() < self.start_date:
                return False
        elif self.end_date:
            # ランの開始が end_date 未満かチェック
            if createdAt.date() >= self.end_date:
                return False

        return True

    def __process_project_runs(self, project):
        max_workers = min(CONFIG.max_workers, len(project.runs))
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(self.__create_metrics_df_with_retry, run.run_path): run for run in project.runs}
            completed = 0
            for future in tqdm(as_completed(futures), total=len(futures), desc=f"Processing {project.project}"):
                run = futures[future]
                try:
                    metrics_df = future.result(timeout=300)
                    run.metrics_df = metrics_df
                except Exception as e:
                    print(f"Error retrieving metrics for run {run.run_path}: {str(e)}")
                    run.metrics_df = pl.DataFrame()
                completed += 1
                gc.collect()
        
        print(f"Completed processing {len(project.runs)} runs for {project.project}")
    
    def __create_metrics_df_with_retry(self, run_path: str, max_retries=3, initial_timeout=5):
        for attempt in range(max_retries):
            try:
                @timeout(initial_timeout * (attempt + 1))
                def attempt_create_metrics_df():
                    return self.__create_metrics_df(run_path)

                return attempt_create_metrics_df()

            except TimeoutError:
                if attempt < max_retries - 1:
                    print(f"Timeout occurred for run {run_path}. Retrying (attempt {attempt + 1}/{max_retries})...")
                    time.sleep(5)
                else:
                    print(f"Failed to retrieve metrics for run {run_path} after {max_retries} attempts")
                    return pl.DataFrame()
            except Exception as e:
                if attempt < max_retries - 1:
                    print(f"Error processing run {run_path} (attempt {attempt + 1}/{max_retries}): {str(e)}")
                    time.sleep(5)
                else:
                    print(f"Failed to process run {run_path} after {max_retries} attempts: {str(e)}")
                    return pl.DataFrame()

    def __create_metrics_df(self, run_path: str) -> pl.DataFrame:
        try:
            run = self.api.run(path=run_path)
            metrics_df = pl.from_dataframe(run.history(stream="events", samples=100))
            if len(metrics_df) <= 1:
                return pl.DataFrame()

            metrics_df_with_datetime = self.__add_datetime_and_filter(metrics_df)
            if metrics_df_with_datetime.is_empty():
                return pl.DataFrame()

            metrics_df_small_width = self.__extract_relevant_columns(metrics_df_with_datetime)
            del metrics_df_with_datetime  # 不要になったDataFrameを削除

            if metrics_df_small_width.width == 2:
                return pl.DataFrame()

            daily_metrics_df = self.__process_daily_metrics(metrics_df_small_width, metrics_df)
            del metrics_df_small_width  # 不要になったDataFrameを削除
            del metrics_df  # 元のmetrics_dfも削除

            return daily_metrics_df

        except Exception as e:
            print(f"Error processing run {run_path}: {str(e)}")
            return pl.DataFrame()
    
    def __add_datetime_and_filter(self, metrics_df: pl.DataFrame) -> pl.DataFrame:
        return (
            metrics_df
            .with_columns(
                pl.col("_timestamp")
                .map_elements(lambda x: dt.datetime.fromtimestamp(x))
                .alias("datetime")
            )
            .filter(
                (pl.col("datetime").dt.date() >= self.start_date) &
                (pl.col("datetime").dt.date() < self.end_date + dt.timedelta(days=1))
            )
        )

    def __extract_relevant_columns(self, df: pl.DataFrame) -> pl.DataFrame:
        gpu_ptn = "^system\.gpu\.\d+\.gpu$"
        memory_ptn = "^system\.gpu\.\d+\.memory$"
        return df.select(
            "datetime",
            "_timestamp",
            pl.col('^' + gpu_ptn),
            pl.col('^' + memory_ptn)
        )

    def __process_daily_metrics(self, df: pl.DataFrame, original_df: pl.DataFrame) -> pl.DataFrame:
        gpu_ptn = "^system\.gpu\.\d+\.gpu$"
        memory_ptn = "^system\.gpu\.\d+\.memory$"
        
        return (
            df
            .with_columns(pl.col("datetime").dt.date().alias("date"))
            .melt(
                id_vars=["date", "datetime", "_timestamp"],
                value_vars=[c for c in original_df.columns if re.findall(gpu_ptn, c)] +
                           [c for c in original_df.columns if re.findall(memory_ptn, c)],
                variable_name="gpu",
                value_name="value",
            )
            .with_columns(pl.col("gpu").map_elements(lambda x: x.split(".")[-1]))
            .group_by(["date", "gpu"])
            .agg(
                pl.col("value").mean().alias("average"),
                pl.col("value").max().alias("max"),
                pl.col("_timestamp")
                .map_elements(lambda x: (max(x) - min(x)) / 60**2)
                .alias("metrics_hours"),
            )
            .pivot(index="date", columns="gpu", values=["average", "max"])
            .rename({f"{prefix}_gpu_gpu": f"{prefix}_gpu_utilization" for prefix in ("average", "max")})
            .select(
                pl.col("date").cast(pl.Date),
                pl.col("average_gpu_utilization").cast(pl.Float64),
                pl.col("max_gpu_utilization").cast(pl.Float64),
                pl.col("average_gpu_memory").cast(pl.Float64),
                pl.col("max_gpu_memory").cast(pl.Float64),
            )
        )
    
    def __create_run_df(self, run: Run) -> pl.DataFrame:
        duration_df = self.__calculate_daily_duration(run.created_at, run.updated_at)
        
        metrics_columns = [
            pl.lit(None).cast(pl.Float64).alias(col) for col in 
            ["average_gpu_utilization", "max_gpu_utilization", "average_gpu_memory", "max_gpu_memory"]
        ]
        
        new_run_df = (
            duration_df.with_columns(metrics_columns) if run.metrics_df.is_empty()
            else duration_df.join(run.metrics_df, on=["date"], how="left")
        )
        
        company_name, project, run_id = run.run_path.split('/')
        
        return new_run_df.with_columns([
            pl.lit(company_name).alias("company_name"),
            pl.lit(project).alias("project"),
            pl.lit(run_id).alias("run_id"),
            pl.lit(json.dumps(run.tags)).alias("tags"),
            pl.lit(run.created_at).cast(pl.Datetime).alias("created_at"),
            pl.lit(run.updated_at).cast(pl.Datetime).alias("updated_at"),
            pl.lit(run.state).cast(pl.String).alias("state"),
            pl.lit(run.gpu_count).cast(pl.Int64).alias("gpu_count"),
            pl.lit(run.host_name).cast(pl.String).alias("host_name"),
            pl.lit(LOGGED_AT).cast(pl.Datetime).alias("logged_at"),
        ]).select([
            "date", "company_name", "project", "run_id", "tags",
            "created_at", "updated_at", "state", "duration_hour", "gpu_count",
            "average_gpu_utilization", "average_gpu_memory",
            "max_gpu_utilization", "max_gpu_memory", "host_name", "logged_at"
        ])

    def __calculate_daily_duration(self, start: dt.datetime, end: dt.datetime) -> pl.DataFrame:
        minutes_range = (
            pl.datetime_range(start, end, interval="1m", eager=True)
            .dt.strftime("%Y-%m-%d %H:00")
            .str.strptime(pl.Datetime, "%Y-%m-%d %H:%M")
        )
        df = (
            pl.DataFrame()
            .with_columns(
                minutes_range.alias("datetime_mins"),
            )
            .with_columns(pl.col("datetime_mins").dt.strftime("%Y-%m-%d").alias("date"))
            .group_by("date")
            .agg(
                pl.col("datetime_mins").count().truediv(60).alias("duration_hour")
            )
            .with_columns(
                pl.col("date").str.strptime(pl.Datetime, "%Y-%m-%d").cast(pl.Date),
            )
            .filter((pl.col("date") >= self.start_date) & (pl.col("date") <= self.end_date))
            .select(
                pl.col("date").cast(pl.Date),
                pl.col("duration_hour").cast(pl.Float64),
            )
        )
        return df
    
    def __set_gpucount(self, node: EasyDict, team: str):
        gpu_count = 0  # デフォルト値

        if team in ["nii-geniac", "elyza-geniac", "kotoba-geniac"]:
            if isinstance(node.config, str):
                try:
                    # configが文字列の場合、JSONとしてパースを試みる
                    config_dict = json.loads(node.config)
                except json.JSONDecodeError as e:
                    print(f"Warning: Unable to parse config JSON for {node.name}: {str(e)}")
                    config_dict = {}
            elif isinstance(node.config, dict):
                # configが既に辞書の場合
                config_dict = node.config
            else:
                print(f"Warning: Unexpected config type for {node.name}: {type(node.config)}")
                config_dict = {}

            if team == "kotoba-geniac":
                # kotoba-geniacの場合、num_nodes * num_gpusで計算
                num_nodes = config_dict.get("num_nodes", 0)
                num_gpus = config_dict.get("num_gpus", 0)
                
                # num_nodesとnum_gpusが辞書の場合、'value'キーの値を取得
                if isinstance(num_nodes, dict):
                    num_nodes = num_nodes.get('value', 0)
                if isinstance(num_gpus, dict):
                    num_gpus = num_gpus.get('value', 0)
                
                # 整数に変換
                try:
                    num_nodes = int(num_nodes)
                    num_gpus = int(num_gpus)
                    gpu_count = num_nodes * num_gpus
                except (ValueError, TypeError):
                    print(f"Warning: Unable to calculate gpu_count for {node.name}. num_nodes: {num_nodes}, num_gpus: {num_gpus}")
                    gpu_count = 0
            else:
                # nii-geniacとelyza-geniacの場合、従来のworld_size処理
                world_size = config_dict.get("world_size")
                if world_size is not None:
                    if isinstance(world_size, dict):
                        gpu_count = world_size.get('value', 0)
                    elif isinstance(world_size, str):
                        if world_size.startswith("{") and world_size.endswith("}"):
                            try:
                                world_size_dict = ast.literal_eval(world_size)
                                gpu_count = world_size_dict.get('value', 0)
                            except (ValueError, SyntaxError):
                                gpu_count = 0
                        else:
                            try:
                                gpu_count = int(world_size)
                            except ValueError:
                                print(f"Warning: Unable to convert world_size to int for {node.name}: {world_size}")
                                gpu_count = 0
                    else:
                        try:
                            gpu_count = int(world_size)
                        except (ValueError, TypeError):
                            print(f"Warning: Unable to convert world_size to int for {node.name}: {world_size}")
                            gpu_count = 0

        # gpu_countが0の場合（取得できなかった場合）、node.runInfo.gpuCountを使用
        if gpu_count == 0:
            gpu_count = node.runInfo.gpuCount if node.runInfo else 0

        return gpu_count

if __name__ == "__main__":
    date_range = ["2024-02-15", "2024-10-11"]
    rm = RunManager(date_range, True)
    df = rm.fetch_runs()
    df.write_csv("dev/new_runs_df.csv")