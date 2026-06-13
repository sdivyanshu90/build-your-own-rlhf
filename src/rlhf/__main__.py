"""
rlhf.__main__ — Typer CLI entrypoint (``python -m rlhf``).

Exposes the three training stages and evaluation as subcommands, each of which
loads an :class:`~rlhf.config.schema.RLHFConfig` from a YAML file. This module is
intentionally thin: it parses arguments and delegates to the stage scripts.
"""

from __future__ import annotations

import typer

from rlhf import __version__

app = typer.Typer(help="RLHF-PPO pipeline command-line interface.", no_args_is_help=True)


@app.command()
def version() -> None:
    """Print the package version."""
    typer.echo(__version__)


@app.command()
def sft(config: str = typer.Option(..., help="Path to the RLHF YAML config.")) -> None:
    """Run supervised fine-tuning (stage 0)."""
    from scripts.train_sft import main as run

    run(config)


@app.command()
def reward(config: str = typer.Option(..., help="Path to the RLHF YAML config.")) -> None:
    """Train the reward model (stage 1)."""
    from scripts.train_reward_model import main as run

    run(config)


@app.command()
def ppo(config: str = typer.Option(..., help="Path to the RLHF YAML config.")) -> None:
    """Run PPO policy optimization (stage 2)."""
    from scripts.train_ppo import main as run

    run(config)


if __name__ == "__main__":  # pragma: no cover - CLI entrypoint
    app()
