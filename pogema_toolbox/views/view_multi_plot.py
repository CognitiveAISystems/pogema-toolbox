from typing import Tuple
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from pogema_toolbox.views.view_plot import PlotView, prepare_plt, prepare_plot_fields
from pogema_toolbox.views.view_utils import eval_logs_to_pandas, drop_na

from typing import Literal


class MultiPlotView(PlotView):
    type: Literal['multi-plot'] = 'multi-plot'
    over: str = None
    num_cols: int = 3
    share_x: bool = False
    share_y: bool = False
    remove_individual_titles: bool = True
    legend_bbox_to_anchor: Tuple[float, float] = (0.5, -0.05)
    legend_loc: str = 'lower center'
    legend_columns: int = 5
    width: float = 2.0
    height: float = 2.0


def process_multi_plot_view(results, view: MultiPlotView, save_path=None):
    df = eval_logs_to_pandas(results)
    df = drop_na(df)
    if view.hue_order is None:
        view.hue_order = sorted(df['algorithm'].unique())

    if view.sort_by:
        df.sort_values(by=['map_name', 'algorithm'], inplace=True)

    if view.rename_fields:
        df = df.rename(columns=view.rename_fields)

    over_keys = sorted(df[view.over].unique())
    num_cols = view.num_cols
    num_rows = len(over_keys) // num_cols + (1 if len(over_keys) % num_cols else 0)

    prepare_plt(view)
    fig, axs = plt.subplots(num_rows, num_cols, figsize=(view.width * num_cols, view.height * num_rows),
                            sharex=view.share_x, sharey=view.share_y)

    # Adjust for when axs is not a 2D array
    if num_rows == 1 or num_cols == 1:
        axs = np.array(axs).reshape(num_rows, num_cols)

    x, y, hue = prepare_plot_fields(view)
    if view.ticks:
        plt.setp(axs, xticks=view.ticks)

    for idx, over in enumerate(over_keys):
        ax = axs[idx // num_cols, idx % num_cols]
        g = sns.lineplot(x=x, y=y, data=df[df[view.over] == over], errorbar=view.error_bar, hue=hue, ax=ax,
                         style=hue if view.line_types else None, markers=view.markers, palette=view.palette,
                         linewidth=view.line_width, hue_order=view.hue_order, style_order=view.hue_order)
        ax.set_title(over if not view.remove_individual_titles else '')

        if view.remove_individual_titles:
            legend = g.get_legend()
            if legend is not None:  # Check if the legend exists before removing
                legend.remove()

        if view.use_log_scale_x:
            ax.set_xscale('log', base=2)
            from matplotlib.ticker import ScalarFormatter
            ax.xaxis.set_major_formatter(ScalarFormatter())

        g.grid()

    # Remove unused axes
    for idx in range(len(over_keys), num_rows * num_cols):
        fig.delaxes(axs.flatten()[idx])

    if view.tight_layout:
        plt.tight_layout()

    # Handle legend outside the loop to prevent duplication
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, bbox_to_anchor=view.legend_bbox_to_anchor, loc=view.legend_loc,
                   ncol=view.legend_columns, fancybox=True, shadow=False)

    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    plt.close(fig)
