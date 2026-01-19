from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, List, Union

from hydra.core.config_store import ConfigStore
from hydra.core.hydra_config import HydraConfig

from gmfm.config.sweep import get_slurm_config, get_sweep
from gmfm.utils.tools import epoch_time, unique_id


@dataclass
class Network:
    arch: str = "unet"
    size: str = "m"
    emb_features: list[int] = field(default_factory=lambda: [512, 512])
    residual: bool = False


@dataclass
class Optimizer:
    lr: float = 5e-4
    iters: int = 20_000
    scheduler: str = 'cos'
    warm_up: bool = False
    optimizer: str = "adam"
    ema_decay: float | None = None
    grad_clip: float | None = None
    pbar_delay: int | None = None


@dataclass
class Data:
    normalize: bool = False
    norm_method: str = '-11'
    sub_t: int = 1
    sub_x: int = 1
    n_t: int = -1
    n_x: int = -1
    n_samples: int = -1
    alpha: float = 1.0
    has_mu: bool = False


@dataclass
class Sample:
    bs_o: int = 256
    bs_n: int = 256
    materialize: bool = True
    resize: bool = False
    replace: bool = False


@dataclass
class Integrate:
    boundary: str | None = None  # clip or periodize, [-1,1]
    n_steps: int = 100


@dataclass
class Loss:
    run: bool = True
    sigma: float = 0.0
    bandwidths: list[float] = field(
        default_factory=lambda: [1.0, 0.5, 0.1, 0.05, 0.01])
    b_min: float = -1
    b_max: float = -1
    n_bands: int = 100
    n_functions: int = 10_000
    reg_kin: float = 0.0
    reg_smt: float = 0.0
    reg_traj: float = 0.0
    relative: bool = True
    stride: int = 1
    dt: str = 'cubic'
    nt_interp: None | int = None
    normalize: None | str = None
    resample: bool = False
    omega_rho: str = 'gauss'  # gauss, orf
    # method of approximation
    # 'cubic': C1 cubic splines (aka local splines)
    # 'cubic2': C2 cubic splines. Can also pass kwarg bc_type, same as scipy.interpolate.CubicSpline
    # 'catmull-rom': C1 cubic centripetal “tension” splines
    # 'cardinal': C1 cubic general tension splines. If used, can also pass keyword parameter c in float[0,1] to specify tension
    # 'akima': C1 cubic splines that appear smooth and natural


@dataclass
class Test:
    n_samples: int = 256
    t_samples: int = 128
    save_samples: bool = False
    n_plot: int = 16
    save_trajectories: bool = False
    plot: bool = True
    metrics: bool = True
    test_idx: list[int] = field(
        default_factory=lambda: [0])


@dataclass
class Config:

    dataset: str = 'wave'
    dump: bool = True

    retest: str | None = None

    net: Network = field(default_factory=Network)

    optimizer: Optimizer = field(default_factory=Optimizer)
    data: Data = field(default_factory=Data)
    sample: Sample = field(default_factory=Sample)
    loss: Loss = field(default_factory=Loss)
    integrate: Integrate = field(default_factory=Integrate)
    test: Test = field(default_factory=Test)

    # misc
    name: str = field(
        default_factory=lambda: f"{unique_id(4)}_{epoch_time(2)}")
    x64: bool = False  # whether to use 64 bit precision in jax

    platform: Union[str, None] = None  # gpu or cpu, None will let jax default

    seed: int = 1

    # hydra config configuration
    hydra: Any = field(default_factory=lambda: hydra_config)
    defaults: List[Any] = field(default_factory=lambda: defaults)


##########################
## hydra settings stuff ##
##########################
defaults = [
    # https://hydra.cc/docs/tutorials/structured_config/defaults/
    # "_self_",
    {"override hydra/launcher": "submitit_slurm"},
]


hydra_config = {
    # sets the out dir from config.problem and id
    "run": {"dir": "results/${dataset}/single/${name}"},
    "sweep": {"dir": "results/${dataset}/multi/${name}"},
    "sweeper": {"params": {**get_sweep()}},
    # https://hydra.cc/docs/1.2/plugins/submitit_launcher/
    "launcher": {**get_slurm_config()},
    # "job": {"env_set": {"XLA_PYTHON_CLIENT_PREALLOCATE": "false", "XLA_FLAGS": "--xla_slow_operation_alarm=false", "TF_CPP_MIN_LOG_LEVEL": "3"}},
    "job_logging": {"root": {"level": "WARN"}},
}
cs = ConfigStore.instance()
cs.store(name="default", node=Config)

wave_cfg = Config(
    dataset="wave",
    data=Data(sub_x=4, sub_t=1, n_samples=1024),
    sample=Sample(bs_n=128, bs_o=5_000),
    loss=Loss(n_functions=50_000, relative=True,
              bandwidths=[5.0, 1.0, 0.5, 0.1, 0.05])

)
cs.store(name="wave", node=wave_cfg)


adv_cfg = Config(
    dataset="adv",
    data=Data(sub_x=2, sub_t=1, n_samples=1024,
              normalize=True, norm_method='-11'),
    sample=Sample(bs_n=256, bs_o=-1, replace=False),
    loss=Loss(n_functions=200_000, relative=True,
              bandwidths=[8.0])

)
cs.store(name="adv", node=adv_cfg)


lz9_cfg = Config(
    dataset="lz9",
    net=Network(arch='mlp'),
    optimizer=Optimizer(pbar_delay=20),
    data=Data(n_samples=25_000, n_t=64, normalize=True, norm_method='-11'),
    sample=Sample(bs_n=10_000, bs_o=-1),
    loss=Loss(n_functions=50_000, relative=True,
              bandwidths=[2.0, 1.0, 0.5, 0.1, 0.05]),
    test=Test(n_samples=20_000)

)
cs.store(name="lz9", node=lz9_cfg)


turb_cfg = Config(
    dataset="turb",
    data=Data(sub_x=1, sub_t=1, n_samples=4096,
              normalize=True, norm_method='-11'),
    sample=Sample(bs_n=256, bs_o=-1, replace=False),
    loss=Loss(n_functions=200_000, relative=True,
              bandwidths=[16.0])

)
cs.store(name="turb", node=turb_cfg)


vtwo_cfg = Config(
    dataset="vbump",
    net=Network(arch='mlp'),
    optimizer=Optimizer(pbar_delay=50),
    data=Data(normalize=True, norm_method='-11', sub_t=1, has_mu=True),
    sample=Sample(bs_n=-1, bs_o=-1),
    loss=Loss(n_functions=50_000, relative=True, b_min=0.005, b_max=1.0),
    test=Test(n_samples=-1, test_idx=[1, 8]),
    integrate=Integrate(boundary='period')

)
cs.store(name="vbump", node=vtwo_cfg)

vtwo_cfg = Config(
    dataset="vtwo",
    net=Network(arch='mlp'),
    optimizer=Optimizer(pbar_delay=50),
    data=Data(normalize=True, norm_method='-11', sub_t=1, has_mu=True),
    sample=Sample(bs_n=-1, bs_o=-1),
    loss=Loss(n_functions=100_000, relative=True, bandwidths=[
              0.5, 0.1], sigma=5e-2, reg_kin=1e-2),
    test=Test(n_samples=-1, test_idx=[1, 8]),
    integrate=Integrate(boundary='period')

)
cs.store(name="vtwo", node=vtwo_cfg)


vtwo_cfg = Config(
    dataset="v6",
    net=Network(arch='mlp'),
    optimizer=Optimizer(pbar_delay=50),
    data=Data(normalize=True, norm_method='-11',
              sub_t=1, has_mu=True, n_samples=25_000),
    sample=Sample(bs_n=-1, bs_o=-1),
    loss=Loss(n_functions=100_000, relative=True, b_min=0.01,
              b_max=0.5, sigma=5e-2, reg_kin=1e-2),
    test=Test(n_samples=-1, test_idx=[2]),
    integrate=Integrate(boundary='period')
)
cs.store(name="v6", node=vtwo_cfg)


def get_outpath() -> Path:

    # save files to
    OUTDIR = HydraConfig.get().runtime.output_dir
    OUTDIR_PATH = Path(OUTDIR)

    return OUTDIR_PATH
