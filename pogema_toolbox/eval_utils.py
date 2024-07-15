import shutil
import tempfile

from pathlib import Path
import wandb


def initialize_wandb(evaluation_config, eval_dir, disable_wandb, project_name):
    mode = 'disabled' if disable_wandb else 'online'
    wandb.init(project=project_name, anonymous="allow", config=evaluation_config,
               mode=mode, job_type=eval_dir.stem, group='eval')


def save_evaluation_results(eval_dir):
    zip_path = f"{eval_dir}.zip"
    shutil.make_archive(str(eval_dir), 'zip', eval_dir)
    wandb.save(str(zip_path), )


def create_and_push_summary_archive(folder_names, base_path, project_name, archive_name='eval_summary'):
    """Create a zip archive and push it to wandb and return the download link."""

    with tempfile.TemporaryDirectory() as tempdir:
        for folder in folder_names:
            src = base_path / folder
            dst = Path(tempdir) / folder
            shutil.copytree(src, dst)

        archive_path = shutil.make_archive(str(base_path / archive_name), 'zip', tempdir)

        with wandb.init(project=project_name) as run:
            artifact = wandb.Artifact(
                name=archive_name,
                type='archive',
                description="Summary archive",
                metadata=dict(folder_names=folder_names)
            )
            artifact.add_file(archive_path)
            run.log_artifact(artifact)
