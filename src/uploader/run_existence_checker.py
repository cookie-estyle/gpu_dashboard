import wandb
from tqdm import tqdm
import polars as pl
from src.utils.config import CONFIG

class RunExistenceChecker:
    def __init__(self):
        self.config = CONFIG
        self.api = wandb.Api()

    def get_artifact(self, entity, project, artifact_name):
        return self.api.artifact(f"{entity}/{project}/{artifact_name}:latest")

    def load_dataframe(self, artifact):
        df = pl.read_csv(artifact.file())
        if 'run_exists' not in df.columns:
            df = df.with_columns(pl.lit('exists').alias('run_exists'))
        return df

    def check_run_existence(self, row):
        run_path = f"{row['company_name']}/{row['project']}/{row['run_id']}"
        try:
            self.api.run(run_path)
            return 'exists'
        except wandb.errors.CommError:
            return 'deleted'
        except Exception as e:
            print(f"Unexpected error occurred for run {run_path}: {str(e)}")
            return 'error'

    def update_run_existence(self, df):
        deleted_runs = []
        total_rows = df.shape[0]
        with tqdm(total=total_rows, desc="Checking runs") as pbar:
            def update_progress(row):
                pbar.update(1)
                new_status = self.check_run_existence(row)
                if new_status == 'deleted' and row['run_exists'] != 'deleted':
                    deleted_runs.append(row)
                return new_status

            df = df.with_columns(
                pl.struct(df.columns)
                .map_elements(update_progress)
                .alias('run_exists')
            )
        return df, deleted_runs

    def save_and_upload_artifact(self, df, csv_path, artifact_name, run):
        df.write_csv(csv_path)
        new_artifact = wandb.Artifact(name=artifact_name, type="dataset")
        new_artifact.add_file(csv_path)
        run.log_artifact(new_artifact)

    def run_existence_check(self):
        artifact_name = self.config.dataset.artifact_name
        csv_path = f"{self.config.wandb_dir}/{artifact_name}.csv"
        deleted_runs_artifact_name = self.config.dataset.deleted_runs_artifact_name

        with wandb.init(
            entity=self.config.dashboard.entity,
            project=self.config.dashboard.project,
            name="run_existence_check", 
            job_type="data_update") as run:
            try:
                artifact = self.get_artifact(self.config.dataset.entity, self.config.dataset.project, artifact_name)
                df = self.load_dataframe(artifact)
                df, deleted_runs = self.update_run_existence(df)
                self.save_and_upload_artifact(df, csv_path, artifact_name, run)
                
                # 削除が検知されたrunの情報を保存し、アップロード
                deleted_runs_df = pl.DataFrame(deleted_runs) if deleted_runs else pl.DataFrame({"run_id": [], "company_name": [], "project": [], "created_at": []})
                deleted_runs_csv_path = f"{self.config.wandb_dir}/{deleted_runs_artifact_name}.csv"
                deleted_runs_df.write_csv(deleted_runs_csv_path)
                
                deleted_runs_artifact = wandb.Artifact(name=deleted_runs_artifact_name, type="dataset")
                deleted_runs_artifact.add_file(deleted_runs_csv_path)
                run.log_artifact(deleted_runs_artifact)

                print(f"Process completed. Updated CSV files have been uploaded as new artifacts.")
            except Exception as e:
                print(f"An error occurred: {str(e)}")
                wandb.log({"error": str(e)})

if __name__ == "__main__":
    checker = RunExistenceChecker()
    checker.run_existence_check()