"""Logic to make units configurable."""

import re
from dataclasses import dataclass
from pathlib import Path
from typing import TypeVar

import gin
import gin.config
from termcolor import colored

T = TypeVar("T")


def configurable(cls: T) -> T:
    """Combine gin.configurable and dataclass for configuration.

    Returns:
        A gin configurable dataclass.

    """
    decorated_cls = dataclass(cls)  # Apply dataclass first
    return gin.configurable(decorated_cls)  # Then make it gin configurable


def parse_gin_config(config_path: str | Path) -> None:
    """Parse a gin config file and display all parameters in scope with color coding.

    Args:
        config_path: Path to the gin config file

    """
    if isinstance(config_path, str):
        config_path = Path(config_path)

    if not config_path.exists():
        print(colored(f"Error: Config file not found: {config_path}", "red"))
        return

    # Parse the config file
    gin.parse_config_file(config_path)

    # Get explicitly set parameters from the file
    with config_path.open("r") as f:
        content = f.read()

    # Get all bound parameters (explicit + defaults) from Gin's current scope
    all_bindings = gin.config_str()

    print(colored("╔═══════════════════════════════════════════════", "white"))
    print(colored("║ Gin configuration", "white", attrs=["bold"]))
    print(colored("╠═══════════════════════════════════════════════", "white"))

    # Display explicitly set parameters (from file)
    param_pattern = re.compile(
        r"([A-Za-z0-9_./]+(?:\.[A-Za-z0-9_]+)?)\s*=\s*(.+?)(?:\s*#.*)?$",
        re.MULTILINE,
    )

    print(colored("║ Explicitly Set Parameters:", "yellow", attrs=["bold"]))
    explicitly_set = set()

    for match in param_pattern.finditer(content):
        param_path, value = match.groups()
        explicitly_set.add(param_path.strip())

        # Clean up value
        value = value.strip().rstrip(";").strip()

        # Color the parameter and value
        colored_param = colored(f"║ {param_path}", "green")
        colored_value = colored(value, "cyan")

        print(f"{colored_param} = {colored_value}")

    # Parse all bindings to find those not explicitly set
    implicit_pattern = re.compile(
        r"([A-Za-z0-9_./]+(?:\.[A-Za-z0-9_]+)?)\s*=\s*(.+?)$",
        re.MULTILINE,
    )

    print(colored("║", "white"))
    print(colored("║ Default/Inherited Parameters:", "yellow", attrs=["bold"]))

    for match in implicit_pattern.finditer(all_bindings):
        param_path, value = match.groups()
        param_path = param_path.strip()

        # Skip if already shown as explicitly set
        if param_path in explicitly_set:
            continue

        # Clean up value
        value = value.strip().rstrip(";").strip()

        # Color differently to show these are defaults
        colored_param = colored(f"║ {param_path}", "blue")
        colored_value = colored(value, "magenta")

        print(f"{colored_param} = {colored_value}")

    print(colored("╚═══════════════════════════════════════════════", "white"))

    # Also display the operative configuration (what Gin will actually use)
    print(colored("\nOperative Configuration:", "white", attrs=["bold"]))
    print(colored(gin.operative_config_str(), "white"))
