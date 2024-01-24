import datetime

from func import (
    get_new_runs,
    update_data_src,
    remove_latest_tags,
    update_companies_table,
    update_overall_table,
)
from utils import set_date


def handler(event, context):
    # 対象とする日付を定義。今日を取得する場合はNone
    target_date = None
    # for day in range(6, 7):
    # target_date = datetime.date(2023, 12, day)
    target_date, processed_at = set_date(target_date=target_date)
    new_runs_df = get_new_runs(target_date=target_date, processed_at=processed_at)
    all_runs_df = update_data_src(df=new_runs_df, target_date=target_date)
    remove_latest_tags()
    update_companies_table(df=all_runs_df, target_date=target_date)
    update_overall_table(df=all_runs_df, target_date=target_date)
    return


if __name__ == "__main__":
    handler(event=None, context=None)
