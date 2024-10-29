from dataclasses import dataclass
import datetime as dt
from typing import List, Set, Any
import pytz
from easydict import EasyDict
import wandb
import yaml

@dataclass
class CompanySchedule:
    company: str
    start_date: dt.date
    end_date: dt.date

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
        
        self.handle_errors(errors)

    def get_company_schedule(self) -> List[CompanySchedule]:
        """企業名と開始日、終了日を取得する"""
        return [
            CompanySchedule(
                company=company.company,
                start_date=self.parse_date(min(s.date for s in company.schedule)),
                end_date=self.parse_date(max(s.date for s in company.schedule))
            )
            for company in self.config.data.companies
        ]

    def get_in_progress_companies(self) -> Set[str]:
        """進行中の企業を取得する"""
        companies = {
            company.company for company in self.get_company_schedule()
            if company.start_date <= self.config.TARGET_DATE < company.end_date
        }
        if companies:
            companies.add("overall")
        return companies

    def get_runs(self) -> List[Any]:
        """runを取得する"""
        project_path = f"{self.config.data.dashboard.entity}/{self.config.data.dashboard.project}"
        return list(self.api.runs(path=project_path))

    def check_runs(self, companies: Set[str], runs: List[Any]) -> List[UpdateError]:
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
        
        self.check_missing_companies(companies, companies_found, errors)
        self.check_extra_companies(companies, companies_found, errors)
        
        return errors

    def check_target_date(self, run_name: str, errors: List[UpdateError]) -> None:
        target_date_str_found = run_name.split("_")[-1]
        if target_date_str_found != self.config.TARGET_DATE_STR:
            errors.append(UpdateError(title="Error of target date", text=f"Expected: {self.config.TARGET_DATE_STR}, Found: {run_name}"))

    def check_missing_companies(self, expected: Set[str], found: Set[str], errors: List[UpdateError]) -> None:
        missing = expected - found
        if missing:
            errors.append(UpdateError(title="Missing latest runs", text=f"Companies without latest runs: {missing}"))

    def check_extra_companies(self, expected: Set[str], found: Set[str], errors: List[UpdateError]) -> None:
        extra = found - expected
        if extra:
            errors.append(UpdateError(title="Unexpected latest runs", text=f"Unexpected companies with latest runs: {extra}"))

    def check_artifacts(self, companies: Set[str], runs: List[Any]) -> List[UpdateError]:
        """アーティファクトをチェックし、エラーがあれば返す"""
        errors = []
        tag_for_latest = self.config.data.dashboard.tag_for_latest

        for run in runs:
            if tag_for_latest in run.tags:
                company_tags = [r for r in run.tags if r != tag_for_latest]
                if len(company_tags) == 1 and company_tags[0] in companies and company_tags[0] != "overall":
                    company = company_tags[0]
                    errors.extend(self.check_company_artifact(company, run))

        return errors

    def check_company_artifact(self, company: str, run: Any) -> List[UpdateError]:
        errors = []
        try:
            artifact_name = f"run-{run.id}-company_daily_gpu_usage:v0"
            artifact = self.api.artifact(f"{self.config.data.dashboard.entity}/{self.config.data.dashboard.project}/{artifact_name}")
            
            table = artifact.get('company_daily_gpu_usage')
            df = table.get_dataframe()

            if 'GPU稼働率(%)' in df.columns:
                latest_gpu_usage = df['GPU稼働率(%)'].iloc[0]
                if latest_gpu_usage <= 10:
                    errors.append(UpdateError(
                        title=f"Low GPU Usage for {company}",
                        text=f"Latest GPU usage is {latest_gpu_usage:.2f}%, which is below 10%"
                    ))
        except Exception as e:
            errors.append(UpdateError(
                title=f"Error checking artifacts for {company}",
                text=str(e)
            ))
        return errors
    
    def handle_errors(self, errors: List[UpdateError]) -> None:
        if self.config.data.enable_alert:
            self.send_alert(errors)
        else:
            for error in errors:
                print(f"{error.title}: {error.text}")

    def send_alert(self, errors: List[UpdateError]) -> None:
        """wandbでアラートを送信する"""
        with wandb.init(
            entity=self.config.data.dashboard.entity,
            project=self.config.data.dashboard.project,
            name="Health Alert",
        ) as run:
            alert_title = f"Dashboard health check for {self.config.TARGET_DATE_STR}"
            msg = self.format_alert_message(errors)
            
            print(msg)
            wandb.alert(title=alert_title, text=msg)

    def format_alert_message(self, errors: List[UpdateError]) -> str:
        header = f"Target Date: {self.config.TARGET_DATE_STR}\n\n"
        if errors:
            return header + "\n".join(f"{error.title}: {error.text}" for error in errors)
        else:
            return header + "No errors found. All active companies are reporting as expected."

    @staticmethod
    def parse_date(date_str: str) -> dt.date:
        return dt.datetime.strptime(date_str, "%Y-%m-%d").date()

if __name__ == "__main__":
    config = Config("config.yaml")
    checker = DashboardChecker(config)
    checker.check_dashboard()