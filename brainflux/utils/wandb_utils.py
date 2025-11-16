# %%

import wandb

from brainflux.utils import wandb_cfg, rouge_one_cfg
from brainflux.utils.config import ExperimentConfig
from brainflux.utils.console import ConsoleManager


# atexit.register(wandb.finish if rouge_one_cfg.do_logging else lambda: None)

ConsoleManager.console_print(
    "Initializing Weights & Biases integration...", style="cyan"
)


def wandb_logging_wrapper(func):
    def wrapper(*args, **kwargs):
        if not rouge_one_cfg.do_logging:
            return None
        return func(*args, **kwargs)

    return wrapper


def sign_in_wandb():
    try:
        wandb.login(key=wandb_cfg.api_key, verify=True)
        ConsoleManager.console_print(
            "Logged in to Weights & Biases using WANDB_API_KEY from environment."
        )
        return
    except Exception as e:
        ConsoleManager.console_error_print(f"Failed to login with WANDB_API_KEY: {e}")


def init_wandb_run(cfg: ExperimentConfig):

    sign_in_wandb()

    if not rouge_one_cfg.do_logging:
        ConsoleManager.console_print(
            "ROGUE_ONE_DO_LOGGING is disabled. Skipping Weights & Biases run initialization.",
            style="yellow",
        )
        return None

    ConsoleManager.console_print(
        f"Initializing Weights & Biases run for {cfg.experiment_name}", style="cyan"
    )
    run = wandb.init(
        project=wandb_cfg.project_name,
        entity=wandb_cfg.entity,
        name=cfg.experiment_name,
        config=cfg.to_dict(),
    )
    ConsoleManager.console_print(
        f"Weights & Biases run initialized: {run.name} ({run.id})", style="green"
    )

    return run


if __name__ == "__main__":
    from math import cos

    from pathlib import Path
    import io

    wandb_run = init_wandb_run(
        ExperimentConfig.from_yaml(
            Path("/workspaces/BrainFlux/tasks/suppression_rato/config.yml")
        )
    )

    for i in range(100):
        wandb.log({"test_metric": cos(i / 10)}, step=i)

# %%
