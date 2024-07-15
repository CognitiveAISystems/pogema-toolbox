from pogema_toolbox.evaluator import evaluation
from pogema import BatchAStarAgent

from pogema_toolbox.eval_utils import initialize_wandb, save_evaluation_results, create_and_push_summary_archive

from pathlib import Path
import wandb

import yaml

from pogema_toolbox.create_env import create_env_base, Environment
from pogema_toolbox.registry import ToolboxRegistry

PROJECT_NAME = 'pogema-toolbox'
BASE_PATH = Path('config_examples')


def main(disable_wandb=True):
    ToolboxRegistry.setup_logger(level='INFO')
    ToolboxRegistry.register_env('Pogema-v0', create_env_base, Environment)
    ToolboxRegistry.register_algorithm('A*', BatchAStarAgent)

    folder_names = [
        'logo',
    ]

    for folder in folder_names:
        config_path = BASE_PATH / folder / f"{Path(folder).name}.yaml"
        eval_dir = BASE_PATH / folder

        ToolboxRegistry.info(f'Starting: {folder}')

        with open(config_path) as f:
            evaluation_config = yaml.safe_load(f)
        if folder == 'eval-fast':
            disable_wandb = True

        initialize_wandb(evaluation_config, eval_dir, disable_wandb, PROJECT_NAME)
        evaluation(evaluation_config, eval_dir=eval_dir)
        save_evaluation_results(eval_dir)
        wandb.finish()

    if not disable_wandb and len(folder_names) > 1:
        ToolboxRegistry.warning(
            create_and_push_summary_archive(folder_names=folder_names, base_path=BASE_PATH, project_name=PROJECT_NAME))


if __name__ == '__main__':
    main()
