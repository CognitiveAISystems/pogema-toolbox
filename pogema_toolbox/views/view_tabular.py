import numpy as np
from numpy import random
from tabulate import tabulate

from pogema_toolbox.registry import ToolboxRegistry
from pogema_toolbox.views.view_utils import View, eval_logs_to_pandas, drop_na
from typing import Literal


class TabularView(View):
    print_results: bool = False
    type: Literal['tabular'] = 'tabular'
    table_format: str = 'psql'
    show_std: bool = True
    skip_zero_std: bool = True


class Aggregator:
    def __init__(self, estimator=np.mean, error_method="ci", error_level=1.96, round_digits=2):
        self.estimator = estimator
        self.error_method = error_method
        self.error_level = error_level
        self.round_digits = round_digits

    def __call__(self, vals):
        estimate = vals.agg(self.estimator) if not callable(self.estimator) else self.estimator(vals)
        interval_str = ""
        round_fmt = f"{{:.{self.round_digits}f}}"

        if self.error_method is None or len(vals) <= 1:
            interval_str = ""
        elif self.error_method == "sd":
            half_interval = vals.std() * self.error_level
            interval_str = f"± {round_fmt.format(half_interval)}"
        elif self.error_method == "ci":
            boots = self.bootstrap(vals, func=self.estimator)
            ci_lower, ci_upper = self._percentile_interval(boots, self.error_level)
            half_interval = (ci_upper - ci_lower) / 2
            interval_str = f"± {round_fmt.format(half_interval)}"
        else:
            raise KeyError(f"Unknown error_method {self.error_method}")

        return f"{round_fmt.format(estimate)} {interval_str}".strip()

    @staticmethod
    def bootstrap(data, n_boot=1000, func=np.mean):
        bs_samples = np.random.choice(data, replace=True, size=(n_boot, len(data)))
        bs_estimates = np.apply_along_axis(func, 1, bs_samples)
        return bs_estimates

    @staticmethod
    def _percentile_interval(data, percentile):
        lower_percentile = (100 - percentile) / 2
        upper_percentile = 100 - lower_percentile
        ci_lower, ci_upper = np.percentile(data, [lower_percentile, upper_percentile])
        return ci_lower, ci_upper


def preprocess_table(eval_configs, view):
    df = eval_logs_to_pandas(eval_configs)
    drop_na(df)

    metrics_keys = [key for key in eval_configs[0].get('metrics', []) if key in df.columns]
    drop_keys = list(view.drop_keys) if hasattr(view, 'drop_keys') else []
    group_by = [col for col in df.columns if col not in metrics_keys + drop_keys]

    # Define the error method and error level from the view configuration
    error_method = getattr(view, 'error_method', 'ci')  # Default to confidence interval
    error_level = getattr(view, 'error_level', 95)  # Default to 95% CI
    aggregator = Aggregator(error_method=error_method, error_level=error_level, round_digits=view.round_digits)

    # Define aggregation operations using the new Aggregator class
    agg_ops = {key: aggregator for key in metrics_keys}
    df_agg = df.groupby(by=group_by, as_index=False).agg(agg_ops).round(view.round_digits)

    # Drop specified keys, handle sorting, and renaming if required
    df_agg.drop(columns=drop_keys, errors='ignore', inplace=True)
    if hasattr(view, 'sort_by') and view.sort_by:
        df_agg.sort_values(by=view.sort_by, inplace=True)
    if hasattr(view, 'rename_fields') and view.rename_fields:
        df_agg.rename(columns=view.rename_fields, inplace=True)

    return df_agg


def process_table_view(results, view: TabularView):
    df = preprocess_table(results, view)

    table = tabulate(df, headers='keys', tablefmt=view.table_format)
    if view.print_results:
        ToolboxRegistry.info('\n' + table)


def generate_mock_data(num_results=25, ):
    results = []
    for i in range(num_results):
        result = {
            "metrics": {
                "avg_throughput": random.uniform(0.01, 0.25),  # Random float between 0.01 and 0.25
                "avg_num_agents_in_obs": random.uniform(1.4, 1.7),  # Random float between 1.4 and 1.7
                "runtime": random.uniform(29, 55)  # Random float between 29 and 55
            },
            "env_grid_search": {
                "map_name": f"mazes-seed-{i}-10x10",
                "num_agents": random.randint(10, 16),
                "seed": random.randint(0, 3)  # Random int between 0 and 1
            },
            "algorithm": "MATS-LP"
        }
        results.append(result)
    return results


def main():
    results = generate_mock_data()
    process_table_view(results, TabularView(round_digits=1, print_results=True, drop_keys=['seed', 'map_name']))
    process_table_view(results, TabularView(round_digits=2, print_results=True, drop_keys=[]))

    process_table_view(results, TabularView(round_digits=3, print_results=True, drop_keys=['seed', 'map_name', 'num_agents']))


if __name__ == '__main__':
    main()
