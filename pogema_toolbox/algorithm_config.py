from typing_extensions import Literal
from typing import Optional

from pydantic import BaseModel


class AlgoBase(BaseModel):
    name: str = None
    num_process: int = 3
    device: str = 'cuda'
    parallel_backend: Literal['dask', 'sequential', 'balanced_dask', 'balanced_multiprocessing', 'multiprocessing'] = 'balanced_dask'
    run_episode_func: str = 'default'

    seed: Optional[int] = 0
    preprocessing: Optional[str] = None
