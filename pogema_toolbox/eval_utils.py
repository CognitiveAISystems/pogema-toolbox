from pogema import pogema_v0

from pogema_toolbox.registry import ToolboxRegistry
import shutil
import tempfile
import yaml

from pathlib import Path
import wandb


def seeded_configs_to_scenarios_converter(env_configs):
    scenarios = {}

    for idx, cfg in enumerate(env_configs):
        env: pogema_v0 = ToolboxRegistry.create_env(env_name=cfg['name'], **cfg)
        env.reset()
        if env.grid_config.on_target == 'restart':
            targets_xy = env.get_lifelong_targets_xy(ignore_borders=True)
        else:
            targets_xy = env.get_targets_xy(ignore_borders=True)
        agents_xy = env.get_agents_xy(ignore_borders=True)
        if hasattr(agents_xy, 'tolist'):
            agents_xy = agents_xy.tolist()
        else:
            agents_xy = [[int(x), int(y)] for x, y in agents_xy]
            
        if hasattr(targets_xy, 'tolist'):
            targets_xy = targets_xy.tolist()
        elif env.grid_config.on_target == 'restart':
            targets_xy = [[[int(x), int(y)] for x, y in agent_targets] for agent_targets in targets_xy]
        else:
            targets_xy = [[int(x), int(y)] for x, y in targets_xy]
            
        scenario = {'agents_xy': agents_xy,
                    'targets_xy': targets_xy,
                    'map_name': env.grid_config.map_name,
                    'seed': cfg.get('seed', env.grid_config.seed)}
        scenario_name = f'Scenario-{str(idx).zfill(len(str(len(env_configs))))}'

        scenarios[scenario_name] = scenario

    return scenarios

def scenarios_to_yaml(scenarios):
    class FlowStyleDumper(yaml.Dumper):
        def represent_sequence(self, tag, sequence, flow_style=None):
            if isinstance(sequence, list) and all(isinstance(i, list) for i in sequence):
                flow_style = True  # Use flow style for lists of lists
            return super().represent_sequence(tag, sequence, flow_style)

    yaml_str = yaml.dump(scenarios, Dumper=FlowStyleDumper, default_flow_style=None, width=256)
    return yaml_str


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
