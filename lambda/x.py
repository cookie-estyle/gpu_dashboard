import os
import sys
import argparse
import datetime as dt
import json

from easydict import EasyDict
import polars as pl
from tqdm import tqdm
import wandb
import yaml

from y import pipeline, update_artifacts


def handler(event: dict[str, str], context: object) -> None:
    ### Read yaml
    with open("config.yaml") as y:
        config = EasyDict(yaml.safe_load(y))
    ### Test mode
    print(f"Test mode: {config.testmode}")

    ### Set WANDB envirionment
    WANDB_API_KEY = event.get("WANDB_API_KEY")
    if WANDB_API_KEY is not None:
        del os.environ["WANDB_API_KEY"]
        os.environ["WANDB_API_KEY"] = WANDB_API_KEY
    environ = config.environ
    os.environ["WANDB_CACHE_DIR"] = environ.WANDB_CACHE_DIR
    os.environ["WANDB_DATA_DIR"] = environ.WANDB_DATA_DIR
    os.environ["WANDB_DIR"] = environ.WANDB_DIR
    # Check
    print(f"Default entity: {wandb.api.default_entity}")

    ### Set target date
    target_date: dt.date
    target_date_str = event.get("target_date")
    if target_date_str is None:
        target_date = dt.date.today()
    else:
        try:
            target_date = dt.datetime.strptime(target_date_str, "%Y-%m-%d").date()
        except:
            print(body := "!!! Invalid date format !!!")
            return {"statusCode": 200, "body": json.dumps(body)}
    # Check
    print(f"Target date is {target_date}")

    ### Get new runs
    df_list = []
    company_config: EasyDict
    for company_config in tqdm(config.companies):
        print(f"Processing {company_config.company_name} ...")
        company_runs_df: pl.DataFrame = pipeline(
            company_name=company_config.company_name,
            gpu_schedule=company_config.schedule,
            target_date=target_date,
            logged_at=dt.datetime.now(),
            testmode=config.testmode,
        )
        if company_runs_df.is_empty():
            continue
        elif (config.testmode) & (len(df_list) == 2):
            continue
        else:
            df_list.append(company_runs_df)
    if df_list:
        print(f"{len(df_list)} runs found.")
        new_runs_df = pl.concat(df_list)
    else:
        print(body := "!!! No runs found !!!")
        return {"statusCode": 200, "body": body}

    ## Update artifacts
    result: dict = update_artifacts(
        new_runs_df=new_runs_df, path_to_dashboard=config.path_to_dashboard
    )

    return {"statusCode": 200, "body": json.dumps(result)}


if __name__ == "__main__":
    ### Parse
    parser = argparse.ArgumentParser(description="推論実行ファイル")
    parser.add_argument("--api", type=str, required=True)  # 「--」無しだと必須の引数
    parser.add_argument("--target-date", type=str)  # 「--」付きだとオプション引数
    args = parser.parse_args()
    ### Run
    event = {"WANDB_API_KEY": args.api, "target_date": args.target_date}
    handler(event=event, context=None)
