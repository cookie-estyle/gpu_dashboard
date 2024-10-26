from dataclasses import dataclass
import datetime as dt
from typing import List, Set
import pytz
from easydict import EasyDict
import wandb
import yaml

@dataclass
class CompanySchedule:
    company: str
    start_date: dt.date

@dataclass
class UpdateError:
    title: str
    text: str

class Config:
    def __init__(self, config_path: str):
        with open(config_path, "r") as f:
            self.data = EasyDict(yaml.safe_load(f))
        self.LOCAL_TZ = pytz.timezone("Asia/Tokyo")
        self.TARGET_DATE = dt.datetime.now(self.LOCAL_TZ).date() + dt.timedelta(days=-1)
        self.TARGET_DATE_STR = self.TARGET_DATE.strftime("%Y-%m-%d")

class DashboardChecker:
    def __init__(self, config: Config):
        self.config = config
        self.api = wandb.Api()

    def check_dashboard(self) -> None:
        """ダッシュボードの健全性をチェックする"""
        companies = self.get_in_progress_companies()
        errors = []

        if companies:
            runs = self.get_runs()
            errors.extend(self.check_runs(companies, runs))
            errors.extend(self.check_artifacts(companies, runs))

        else:
            errors.append(UpdateError(title="No active companies", text="There are no companies currently active."))
        
        if self.config.data.enable_alert:
            self.send_alert(errors)
        else:
            for error in errors:
                print(error)

    def get_company_schedule(self) -> List[CompanySchedule]:
        """企業名と開始日を取得する"""
        return [
            CompanySchedule(
                company=company.company,
                start_date=min(dt.datetime.strptime(s.date, "%Y-%m-%d").date() for s in company.schedule)
            )
            for company in self.config.data.companies
        ]

    def get_in_progress_companies(self) -> Set[str]:
        """進行中の企業を取得する"""
        today = self.config.TARGET_DATE
        companies = set()
        
        for company in self.config.data.companies:
            start_date = min(dt.datetime.strptime(s['date'], "%Y-%m-%d").date() for s in company.schedule)
            end_date = max(dt.datetime.strptime(s['date'], "%Y-%m-%d").date() for s in company.schedule)
            
            # 開始日以降かつ終了日の前日まで進行中とみなす
            if start_date <= today < end_date:
                companies.add(company.company)
        
        # 進行中の企業がある場合のみ "overall" を追加
        if companies:
            companies.add("overall")
        return companies

    def get_runs(self) -> object:
        """runを取得する"""
        project_path = f"{self.config.data.dashboard.entity}/{self.config.data.dashboard.project}"
        return self.api.runs(path=project_path)

    def check_runs(self, companies: Set[str], runs: object) -> List[UpdateError]:
        """runをチェックし、エラーがあれば返す"""
        errors = []
        companies_found = set()
        tag_for_latest = self.config.data.dashboard.tag_for_latest
        
        for run in runs:
            if tag_for_latest in run.tags:
                company_tags = [r for r in run.tags if r != tag_for_latest]
                if len(company_tags) == 1 and company_tags[0] in companies:
                    companies_found.add(company_tags[0])
                    self.check_target_date(run.name, errors)
        
        missing_companies = companies - companies_found
        if missing_companies:
            errors.append(UpdateError(title="Missing latest runs", text=f"Companies without latest runs: {missing_companies}"))
        
        extra_companies = companies_found - companies
        if extra_companies:
            errors.append(UpdateError(title="Unexpected latest runs", text=f"Unexpected companies with latest runs: {extra_companies}"))
        
        return errors

    def check_tags(self, another_tags: List[str], all_tags: List[str], companies_found: List[str], errors: List[UpdateError]) -> None:
        if len(another_tags) != 1:
            errors.append(UpdateError(title="Error of number of tags", text=str(all_tags)))
        else:
            companies_found.append(another_tags[0])

    def check_target_date(self, run_name: str, errors: List[UpdateError]) -> None:
        target_date_str_found = run_name.split("_")[-1]
        if target_date_str_found != self.config.TARGET_DATE_STR:
            errors.append(UpdateError(title="Error of target date", text=f"{self.config.TARGET_DATE_STR}, {run_name}"))

    def check_companies(self, companies: Set[str], companies_found: List[str], errors: List[UpdateError]) -> None:
        if len(companies) != len(companies_found):
            errors.append(UpdateError(title="Error of number of latest runs", text=""))
        if companies != set(companies_found):
            errors.append(UpdateError(title="Error of companies", text=str(companies_found)))

    def check_artifacts(self, companies: Set[str], runs: object) -> List[UpdateError]:
        """アーティファクトをチェックし、エラーがあれば返す"""
        errors = []
        tag_for_latest = self.config.data.dashboard.tag_for_latest

        for run in runs:
            if tag_for_latest in run.tags:
                company_tags = [r for r in run.tags if r != tag_for_latest]
                if len(company_tags) == 1 and company_tags[0] in companies and company_tags[0] != "overall":
                    company = company_tags[0]
                    try:
                        # runに関連付けられたアーティファクトを取得
                        artifact_name = f"run-{run.id}-company_daily_gpu_usage:v0"
                        artifact = self.api.artifact(f"{self.config.data.dashboard.entity}/{self.config.data.dashboard.project}/{artifact_name}")
                        
                        # テーブル名は 'company_daily_gpu_usage' と仮定
                        table = artifact.get('company_daily_gpu_usage')
                        df = table.get_dataframe()

                        # GPU稼働率(%)のチェック
                        if 'GPU稼働率(%)' in df.columns:
                            gpu_usage = df['GPU稼働率(%)'].mean()
                            if gpu_usage <= 10:
                                errors.append(UpdateError(
                                    title=f"Low GPU Usage for {company}",
                                    text=f"Average GPU usage is {gpu_usage:.2f}%, which is below 10%"
                                ))

                    except Exception as e:
                        errors.append(UpdateError(
                            title=f"Error checking artifacts for {company}",
                            text=str(e)
                        ))

        return errors
    
    def send_alert(self, errors: List[UpdateError]) -> None:
        """wandbでアラートを送信する"""
        with wandb.init(
            entity=self.config.data.dashboard.entity,
            project=self.config.data.dashboard.project,
            name="Health Alert",
        ) as run:
            alert_title = f"Dashboard health check for {self.config.TARGET_DATE_STR}"
            if errors:
                msg = f"Target Date: {self.config.TARGET_DATE_STR}\n\n" + "\n".join(f"{error.title}: {error.text}" for error in errors)
            else:
                msg = f"Target Date: {self.config.TARGET_DATE_STR}\n\nNo errors found. All active companies are reporting as expected."
            
            print(msg)
            wandb.alert(title=alert_title, text=msg)

if __name__ == "__main__":
    config = Config("config.yaml")
    checker = DashboardChecker(config)
    checker.check_dashboard()