import os
from pathlib import Path
from typing import Literal
from dataclasses import dataclass

from dotenv import load_dotenv
from openai.types.shared.reasoning import Reasoning

from rogueone.utils.singelton import Singleton

p = Path(__file__).resolve().parent.parent.parent
load_dotenv(p / ".env")
load_dotenv(p / ".dev.env", override=True)
load_dotenv(p / ".secret.env")


@dataclass(frozen=True)
class ExperimentConfig:
    experiment_name: str
    experiment_description: str

    knowledge_db_path: Path
    knowledge_db_collection_name: str

    output_dir: Path

    modality: Literal["tabular", "time_series"]
    task_type: Literal["classification", "regression"]

    feature_file_path: Path
    label_file_path: Path

    context_and_goal: str

    def __post_init__(self):
        if isinstance(self.knowledge_db_path, str):
            object.__setattr__(self, "knowledge_db_path", Path(self.knowledge_db_path))
        if isinstance(self.feature_file_path, str):
            object.__setattr__(self, "feature_file_path", Path(self.feature_file_path))
        if isinstance(self.label_file_path, str):
            object.__setattr__(self, "label_file_path", Path(self.label_file_path))
        if isinstance(self.output_dir, str):
            object.__setattr__(self, "output_dir", Path(self.output_dir))

        # assert (
        #     self.knowledge_db_path.exists()
        # ), f"Knowledge DB path {self.knowledge_db_path} does not exist."
        assert (
            self.feature_file_path.exists()
        ), f"Feature file path {self.feature_file_path} does not exist."
        assert (
            self.label_file_path.exists()
        ), f"Label file path {self.label_file_path} does not exist."

        if not self.output_dir.exists():
            self.output_dir.mkdir(parents=True)

    @classmethod
    def from_dict(cls, config_dict: dict) -> "ExperimentConfig":
        return cls(
            experiment_name=config_dict.get("experiment_name", "default_experiment"),
            experiment_description=config_dict.get(
                "experiment_description", "No description provided."
            ),
            knowledge_db_path=Path(config_dict["knowledge_db_path"]),
            knowledge_db_collection_name=config_dict["knowledge_db_collection_name"],
            feature_file_path=Path(config_dict["feature_file_path"]),
            label_file_path=Path(config_dict["label_file_path"]),
            context_and_goal=config_dict["context_and_goal"],
            output_dir=Path(config_dict.get("output_dir", "./output")),
            modality=config_dict.get("modality", "tabular"),
            task_type=config_dict.get("task_type", "classification"),
        )

    @classmethod
    def from_yaml(cls, yaml_path: Path | str) -> "ExperimentConfig":
        import yaml

        if isinstance(yaml_path, str):
            yaml_path = Path(yaml_path)

        assert yaml_path.exists(), f"YAML file {yaml_path} does not exist."

        with open(yaml_path, "r") as file:
            config_dict = yaml.safe_load(file)
        return cls.from_dict(config_dict)

    def to_dict(self) -> dict:
        return {
            "experiment_name": self.experiment_name,
            "experiment_description": self.experiment_description,
            "knowledge_db_path": str(self.knowledge_db_path),
            "knowledge_db_collection_name": self.knowledge_db_collection_name,
            "feature_file_path": str(self.feature_file_path),
            "label_file_path": str(self.label_file_path),
            "context_and_goal": self.context_and_goal,
            "output_dir": str(self.output_dir),
            "modality": self.modality,
            "task_type": self.task_type,
        }


class LLMConfig(metaclass=Singleton):
    def __init__(self):
        self.model: Literal["gpt-oss-120b", "gpt-oss-20b"] = os.getenv("LLM_MODEL")
        self.temperature: float = float(os.getenv("LLM_TEMPERATURE", 0.7))
        self.max_retries: int = int(os.getenv("LLM_MAX_RETRIES", 3))
        self.api_key: str = os.getenv("LLM_API_KEY", "")
        self.port: str = os.getenv("LLM_PORT", "")
        self.endpoint: str = f"http://localhost:{self.port}/v1"


class EmbeddingConfig(metaclass=Singleton):
    def __init__(self):
        self.model: Literal["Qwen/Qwen3-Embedding-4B"] = os.getenv("EMBEDD_MODEL")
        self.chunk_size: int = int(os.getenv("EMBEDD_CHUNK_SIZE", 1))
        self.api_key: str = os.getenv("EMBEDD_API_KEY", "")
        self.port: str = os.getenv("EMBEDD_PORT", "")
        self.endpoint: str = f"http://localhost:{self.port}/v1"


class ConsoleConfig(metaclass=Singleton):
    def __init__(self):
        self.disable: bool = os.getenv("CONSOLE_DISABLE", "0") in ["1", "true", "True"]


class rogueoneConfig(metaclass=Singleton):
    def __init__(self):
        self.num_iterations: int = int(os.getenv("ROGUE_ONE_NUM_ITERATIONS", 3))
        self.train_size: float = float(os.getenv("ROGUE_ONE_TRAIN_SIZE", 0.8))
        self.test_size: float = 1 - self.train_size
        self.do_logging: bool = os.getenv("ROGUE_ONE_DO_LOGGING", "1") in [
            "1",
            "true",
            "True",
        ]
        self.k_folds: int = int(os.getenv("ROGUE_ONE_K_FOLDS", 5))
        self.do_balancing = os.getenv("ROGUE_ONE_DO_BALANCING", "1") in [
            "1",
            "true",
            "True",
        ]


class WandBConfig(metaclass=Singleton):
    def __init__(self):
        self.project_name: str = os.getenv("WANDB_PROJECT_NAME", "brainflux_project")
        self.entity: str = os.getenv("WANDB_ENTITY", "brainflux_entity")
        self.api_key: str = os.getenv("WANDB_API_KEY", "")


class AgentConfig:
    def __init__(
        self,
        temperature: float = 0.7,
        reasoning_effort: Literal["minimal", "low", "medium", "high"] | None = "medium",
        max_iterations: int = 100,
    ):
        self.temperature = temperature
        self.reasoning = Reasoning(effort=reasoning_effort)
        self.max_iterations = max_iterations


class ExtractorAgentConfig(AgentConfig, metaclass=Singleton):
    def __init__(self):
        super().__init__(
            temperature=os.getenv("EXTRACTOR_AGENT_TEMPERATURE", 0.7),
            reasoning_effort=os.getenv("EXTRACTOR_AGENT_REASONING_EFFORT", None),
            max_iterations=int(os.getenv("EXTRACTOR_AGENT_MAX_ITERATIONS", 100)),
        )


class MedicalKnowledgeAgentConfig(AgentConfig, metaclass=Singleton):
    def __init__(self):
        super().__init__(
            temperature=os.getenv("MEDICAL_KNOWLEDGE_AGENT_TEMPERATURE", 0.7),
            reasoning_effort=os.getenv(
                "MEDICAL_KNOWLEDGE_AGENT_REASONING_EFFORT", None
            ),
            max_iterations=int(
                os.getenv("MEDICAL_KNOWLEDGE_AGENT_MAX_ITERATIONS", 100)
            ),
        )


class TestAgentConfig(AgentConfig, metaclass=Singleton):
    def __init__(self):
        super().__init__(
            temperature=os.getenv("TEST_AGENT_TEMPERATURE", 0.7),
            reasoning_effort=os.getenv("TEST_AGENT_REASONING_EFFORT", None),
            max_iterations=int(os.getenv("TEST_AGENT_MAX_ITERATIONS", 100)),
        )


class ScientistAgentConfig(AgentConfig, metaclass=Singleton):
    def __init__(self):
        super().__init__(
            temperature=os.getenv("ORCHESTRATOR_AGENT_TEMPERATURE", 0.7),
            reasoning_effort=os.getenv("ORCHESTRATOR_AGENT_REASONING_EFFORT", None),
            max_iterations=int(os.getenv("ORCHESTRATOR_AGENT_MAX_ITERATIONS", 100)),
        )
