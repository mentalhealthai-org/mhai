"""Configuration for pytest."""

from pathlib import Path

from dotenv import load_dotenv

ENV_PATH = Path(__file__).parent / '.envs' / '.env'
load_dotenv(dotenv_path=ENV_PATH)
