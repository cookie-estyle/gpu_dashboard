import wandb
from tqdm import tqdm
import polars as pl
from src.utils.config import CONFIG

def get_artifact(api, entity, project, artifact_name):
    return api.artifact(f"{entity}/{project}/{artifact_name}:latest")

def load_dataframe(artifact):
    df = pl.read_csv(artifact.file())
    if 'run_exists' not in df.columns:
        df = df.with_columns(pl.lit('exists').alias('run_exists'))
    return df

def check_run_existence(api, row):
    run_path = f"{row['company_name']}/{row['project']}/{row['run_id']}"
    try:
        api.run(run_path)
        return 'exists'
    except wandb.errors.CommError:
        return 'deleted'
    except Exception as e:
        print(f"Unexpected error occurred for run {run_path}: {str(e)}")
        return 'error'

def update_run_existence(df, api):
    deleted_runs = []
    total_rows = df.shape[0]
    with tqdm(total=total_rows, desc="Checking runs") as pbar:
        def update_progress(row):
            pbar.update(1)
            new_status = check_run_existence(api, row)
            if new_status == 'deleted' and row['run_exists'] != 'deleted':
                deleted_runs.append(row)
            return new_status

        df = df.with_columns(
            pl.struct(df.columns)
            .map_elements(update_progress)
            .alias('run_exists')
        )
    return df, deleted_runs

def save_and_upload_artifact(df, csv_path, artifact_name, run):
    df.write_csv(csv_path)
    new_artifact = wandb.Artifact(name=artifact_name, type="dataset")
    new_artifact.add_file(csv_path)
    run.log_artifact(new_artifact)

def run_existence_check():
    artifact_name = CONFIG.dataset.artifact_name
    csv_path = f"{CONFIG.wandb_dir}/{artifact_name}.csv"
    deleted_runs_artifact_name = CONFIG.dataset.deleted_runs_artifact_name

    with wandb.init(
        entity=CONFIG.dashboard.entity,
        project=CONFIG.dashboard.project,
        name="run_existence_check", 
        job_type="data_update") as run:
        try:
            api = wandb.Api()
            artifact = get_artifact(api, CONFIG.dataset.entity, CONFIG.dataset.project, artifact_name)
            df = load_dataframe(artifact)
            df, deleted_runs = update_run_existence(df, api)
            save_and_upload_artifact(df, csv_path, artifact_name, run)
            
            # 削除が検知されたrunの情報を保存し、アップロード
            if deleted_runs:
                deleted_runs_df = pl.DataFrame(deleted_runs)
                deleted_runs_csv_path = f"{CONFIG.wandb_dir}/{deleted_runs_artifact_name}.csv"
                deleted_runs_df.write_csv(deleted_runs_csv_path)
                
                deleted_runs_artifact = wandb.Artifact(name=deleted_runs_artifact_name, type="dataset")
                deleted_runs_artifact.add_file(deleted_runs_csv_path)
                run.log_artifact(deleted_runs_artifact)

            print(f"Process completed. Updated CSV files have been uploaded as new artifacts.")
        except Exception as e:
            print(f"An error occurred: {str(e)}")
            wandb.log({"error": str(e)})

if __name__ == "__main__":
    run_existence_check()