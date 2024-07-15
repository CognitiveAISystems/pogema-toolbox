import seaborn as sns

import numpy as np
from matplotlib import pyplot as plt
from typing import Optional

from pogema_toolbox.registry import ToolboxRegistry
from pogema_toolbox.views.view_utils import View, eval_logs_to_pandas, drop_na
from typing import Tuple

from typing_extensions import Literal


def custom_palette():
    q = list(sns.color_palette("deep"))
    q[1], q[9] = q[9], q[1]
    q[5], q[9] = q[9], q[5]
    q[6], q[9] = q[9], q[6]
    return q


class PlotView(View):
    type: Literal['plot'] = 'plot'
    name: str = None
    x: str = None
    y: str = None
    by: str = 'algorithm'
    width: float = 2.6
    height: float = 2.8
    remove_title: bool = False
    line_width: float = 2.0

    error_bar: Tuple[str, int] = ('ci', 95)

    plt_style: str = 'seaborn-v0_8-colorblind'
    figure_dpi: int = 300
    font_size: int = 8
    legend_font_size: int = 7
    legend_loc: Literal[
        'best', 'upper right', 'upper left', 'lower left', 'lower right', 'right',
        'center left', 'center right', 'lower center', 'upper center', 'center'] = 'best'
    figure_face_color: str = '#FFFFFF'
    use_log_scale_x: bool = False
    use_log_scale_y: bool = False
    markers: bool = True
    line_types: bool = True
    extension: Literal['svg', 'png', 'pdf'] = 'pdf'
    palette: list = custom_palette()
    hue_order: list = None

    tight_layout: bool = True
    ticks: Optional[list] = None
    remove_legend_title: bool = True


def process_plot_view(results, view: PlotView, save_path=None):
    df = eval_logs_to_pandas(results)
    df = drop_na(df)
    if view.hue_order is None:
        view.hue_order = sorted(df['algorithm'].unique())

    if view.sort_by:
        df.sort_values(by=['map_name', 'algorithm'], inplace=True)

    if view.rename_fields:
        df = df.rename(columns=view.rename_fields)

    prepare_plt(view)
    x, y, hue = prepare_plot_fields(view)

    fig, ax = plt.subplots()
    if x not in df.keys():
        ToolboxRegistry.warning(f"Could not interpret value {x} for parameter 'x'. Skipping this plot.")
        return
    if y not in df.keys():
        ToolboxRegistry.warning(f"Could not interpret value {y} for parameter 'y'. Skipping this plot.")
        return

    sns.lineplot(x=x, y=y, data=df, errorbar=view.error_bar, hue=hue, hue_order=view.hue_order,
                 style_order=view.hue_order, linewidth=view.line_width,
                 style=hue if view.line_types else None, markers=view.markers,
                 palette=view.palette[:len(view.hue_order)], ax=ax,
                 )
    if not view.remove_title:
        ax.set_title(view.name)

    if view.remove_legend_title:
        ax.legend().set_title('')

    if view.use_log_scale_x:
        ax.set_xscale('log', base=2)
        from matplotlib.ticker import ScalarFormatter
        ax.xaxis.set_major_formatter(ScalarFormatter())

    if view.use_log_scale_y:
        ax.set_yscale('log', base=2)
        from matplotlib.ticker import ScalarFormatter
        ax.yaxis.set_major_formatter(ScalarFormatter())

    plt.tight_layout()
    plt.grid()

    if view.ticks:
        ax.set_xticks(np.array(view.ticks))

    plt.plot()

    if save_path:
        plt.savefig(save_path)

    plt.close()


def prepare_plt(view: PlotView):
    plt.style.use(view.plt_style)
    plt.rcParams['figure.figsize'] = (view.width, view.height)
    plt.rcParams['figure.dpi'] = view.figure_dpi
    plt.rcParams['font.size'] = view.font_size
    plt.rcParams['legend.fontsize'] = view.legend_font_size
    plt.rcParams['legend.loc'] = view.legend_loc
    plt.rcParams['figure.facecolor'] = view.figure_face_color

    if view.name:
        plt.title(view.name)


def prepare_plot_fields(view):
    x = view.x if view.x not in view.rename_fields else view.rename_fields[view.x]
    y = view.y if view.y not in view.rename_fields else view.rename_fields[view.y]
    hue = view.by if view.by not in view.rename_fields else view.rename_fields[view.by]
    return x, y, hue
