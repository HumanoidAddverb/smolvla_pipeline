from dataclasses import dataclass
from pathlib import Path

from omegaconf import OmegaConf


@dataclass(frozen=True)
class Configs:
	PKG: Path = Path(__file__).parents[0]
	ROOT: Path = Path(__file__).parents[1]

	def _load(self, file):
		return OmegaConf.load(file)

	@classmethod
	def sac(cls):
		return cls()._load(f'{cls.PKG}/configs/sac.yaml')