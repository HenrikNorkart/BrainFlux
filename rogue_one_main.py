import asyncio
from dataclasses import dataclass

import pandas as pd
from pathlib import Path
from rich.live import Live
from agents import set_tracing_disabled


set_tracing_disabled(True)

# from weave.integrations.openai_agents.openai_agents import WeaveTracingProcessor

# import weave

# set_trace_processors([WeaveTracingProcessor()])

from rogueone.utils import rouge_one_cfg
from rogueone.utils.config import ExperimentConfig
from rogueone.utils.train_test_splitter import TrainTestSplitter
from rogueone.utils.console import ConsoleManager
from rogueone.llm.agents import (
    ExtractorAgent,
    ScientistAgent,
    TesterAgent,
)
from rogueone.dataclasses import AttributeExplanation
from rogueone.utils.wandb_utils import init_wandb_run, wandb_logging_wrapper
from rogueone.utils.run_vllm import start_vllm_servers


@dataclass
class TestResult:
    hypothesis: str
    score: float
    attributes_used: list[
        AttributeExplanation
    ]  # List of attribute names used in the test


class RogueOneAgentNetwork:

    def __init__(self, cfg: ExperimentConfig):

        self.cfg = cfg

        ConsoleManager.console_rule("Setting up Weights & Biases Logging")
        init_wandb_run(cfg)

        self.df = pd.read_csv(self.cfg.feature_file_path)

        try:
            # If time series data with multiple entries per entity
            self.df_entities_attributes = pd.DataFrame(
                self.df["id"].unique(), columns=["id"]
            )
        except:
            self.df_entities_attributes = pd.DataFrame({"id": list(self.df.index)})

        self.df_test_pool = None

        self.extractor_agent = ExtractorAgent(self.cfg)
        self.scientist_agent = ScientistAgent(self.cfg)
        self.tester_agent = TesterAgent(self.cfg, include_report=True)

        self.splitter = TrainTestSplitter(
            cfg=self.cfg,
            test_size=rouge_one_cfg.test_size,
            # random_state=42,
        )

        self.df_attributes_explanations = pd.DataFrame(
            columns=["Attribute", "Description", "Pandas Command", "Status", "Added"]
        )

        # self.attributes_explanations: dict[str, AttributeExplanation] = {}
        self.focus_history = []

    async def update_attribute_pool(self, attribute_explanations: pd.DataFrame):
        for _, attr in attribute_explanations.iterrows():
            mask = self.df_attributes_explanations["Attribute"] == attr["Attribute"]
            if mask.any():
                # Update status using .loc to avoid chained assignment warnings/errors
                self.df_attributes_explanations.loc[mask, "Status"] = attr["Status"]

                # if attr["Status"] == "Pruned":
                #     self.df_attributes_explanations.drop(
                #         self.df_attributes_explanations[mask].index,
                #         inplace=True,
                #     )

            else:
                # Create a new row DataFrame from the Series and align columns with the target dataframe
                new_row = pd.DataFrame([attr.to_dict()])
                # Ensure all expected columns exist in the new row (missing columns -> NA)

                new_row = new_row[self.df_attributes_explanations.columns]
                self.df_attributes_explanations = pd.concat(
                    [self.df_attributes_explanations, new_row],
                    ignore_index=True,
                )

        # De-fragment the dataframe
        self.df_attributes_explanations = self.df_attributes_explanations.copy()
        self.df_test_pool = (
            self.df_test_pool.copy() if self.df_test_pool is not None else None
        )

    async def forward_pass(self, index: int):
        ######### Setting Focus of Investigation #########
        ConsoleManager.console_rule("Setting Focus of Investigation")
        focus = await self.scientist_agent.determine_focus(
            self.df,
            self.df_test_pool,
            self.df_attributes_explanations,
            self.focus_history,
            index=index,
            max_index=rouge_one_cfg.num_iterations,
        )
        self.focus_history.append(focus)
        ConsoleManager.console_print(f"Focus of Investigation: [bold]{focus}[/bold]\n")

        ######### Extracting Attributes #########
        num_known_attributes = len(self.df_entities_attributes.columns) - 1
        ConsoleManager.console_rule("Extracting Attributes")
        ConsoleManager.progress_bar_update(description="Extracting attributes")
        self.df_entities_attributes, _attribute_explanations = (
            await self.extractor_agent.generate_patient_attributes(
                self.df, self.df_entities_attributes, focus, step=index
            )
        )

        ######## Logging and Updating Attribute Explanations #########
        ConsoleManager.console_rule("Extraction Summary")
        ConsoleManager.print_dataframe_as_table(
            _attribute_explanations,
            title="Extraction Summary",
        )

        num_new_attributes = (
            len(self.df_entities_attributes.columns) - 1 - num_known_attributes
        )
        if num_new_attributes == 0:
            ConsoleManager.console_print(
                "[red]No new attributes were extracted. Ending iteration.[/red]\n"
            )
            return

        await self.update_attribute_pool(_attribute_explanations)

        ConsoleManager.console_print(f"Extracted {num_new_attributes} attributes.\n")

        ######### Splitting Patient Data #########
        ConsoleManager.console_rule("Splitting Data")
        ConsoleManager.progress_bar_update(description="Splitting data")

        match self.cfg.modality:
            case "tabular":
                df_entities_folds = await self.splitter.k_fold_split(
                    self.df_entities_attributes, k=rouge_one_cfg.k_folds
                )
            case "time_series":
                df_entities_folds = await self.splitter.test_train_split(
                    self.df_entities_attributes
                )
            case _:
                raise ValueError(f"Unsupported modality: {self.cfg.modality}")

        ######### Generating and Testing Hypotheses #########
        ConsoleManager.console_rule("Generating and Testing Hypotheses")
        ConsoleManager.progress_bar_update(description="Testing hypotheses")
        test_results, _attribute_explanations = await self.tester_agent.test_hypotheses(
            df_entities_folds,
            self.df_attributes_explanations,
            step=index,
        )
        await self.update_attribute_pool(_attribute_explanations)
        test_results = test_results.as_dict.copy()
        test_results["id"] = "Trial " + str(index)

        if self.df_test_pool is None:
            self.df_test_pool = pd.DataFrame([test_results])
        else:
            self.df_test_pool = pd.concat(
                [self.df_test_pool, pd.DataFrame([test_results])],
                ignore_index=True,
            )

        report = test_results.pop("report")

        await self.dump_data(index)
        await self.save_report(report, index)
        ConsoleManager.print_dict_as_table(test_results)
        ConsoleManager.console_print(report)

    async def dump_data(self, step: int):
        dst_dir = self.cfg.output_dir / "intermediate_steps"
        if not dst_dir.exists():
            dst_dir.mkdir(parents=True)

        self.df_entities_attributes.to_csv(
            dst_dir / f"df_entities_attributes_trail_{step}.csv",
            index=False,
        )

    async def save_report(self, report: str | None, index: int):
        if report is None:
            return
        dir = self.cfg.output_dir / "tester_reports"
        if not dir.exists():
            dir.mkdir(parents=True)
        dst_path = dir / f"tester_report_iteration_{index}.md"
        with open(dst_path, "w") as f:
            f.write(report)

    async def save_data(self):
        dst_path = self.cfg.output_dir

        df_test_pool_path = dst_path / "df_test_pool_final.csv"
        df_entities_attributes_path = dst_path / "df_entities_attributes_final.csv"
        df_attributes_explanations_path = (
            dst_path / "df_attributes_explanations_final.csv"
        )

        self.df_test_pool.to_csv(df_test_pool_path, index=False)
        self.df_entities_attributes.to_csv(df_entities_attributes_path, index=False)
        self.df_attributes_explanations.to_csv(
            df_attributes_explanations_path, index=False
        )

        @wandb_logging_wrapper
        def log_wandb_metrics():
            import wandb

            artifact = wandb.Artifact(name="df_test_pool_final", type="dataset")
            artifact.add_file(str(df_test_pool_path))
            wandb.log_artifact(artifact)

            artifact = wandb.Artifact(
                name="df_attributes_explanations_final", type="dataset"
            )
            artifact.add_file(str(df_attributes_explanations_path))
            wandb.log_artifact(artifact)

        log_wandb_metrics()

    async def main(self):
        ConsoleManager.reset_progress_bar()
        with Live(
            ConsoleManager().progress_bar,
            console=ConsoleManager().console,
            vertical_overflow="ellipsis",
        ) as live:

            ConsoleManager.progress_bar_add_task(
                description="Rogue One Agent Network Progress",
                total=rouge_one_cfg.num_iterations,
            )

            for i in range(rouge_one_cfg.num_iterations):
                ConsoleManager.progress_bar_update(completed=i)
                await self.forward_pass(i)

                # break  # For now, only one iteration
            ConsoleManager.console_rule("Final Results")
            await self.save_data()
            ConsoleManager.print_dataframe_as_table(
                self.df_test_pool.iloc[:, :5], title="Final Test Pool (First 5 Columns)"
            )

        @wandb_logging_wrapper
        def log_final_results():
            import wandb

            wandb.finish()

        log_final_results()


if __name__ == "__main__":

    paths = [
        ### Custom tasks
        # Path("/workspace/tasks/aphasia/config.yml"),
        # Path("/workspace/tasks/suppression_rato/config.yml"),
        #
        #
        ### Time Series Classification
        # Path("/workspace/tasks/time_series/EthanolConcentration/config.yml"),
        # Path("/workspace/tasks/time_series/NATOPS/config.yml"),
        # Path("/workspace/tasks/time_series/FaceDetection/config.yml"),
        # Path("/workspace/tasks/time_series/ArticularyWordRecognition/config.yml"),
        # Path("/workspace/tasks/time_series/BasicMotions/config.yml"),
        #
        #
        ### Tabular Classification
        Path("/workspace/tasks/classification/balance-scale/config.yml"),
        Path("/workspace/tasks/classification/covtype/config.yml"),
        Path("/workspace/tasks/classification/pc1/config.yml"),
        Path("/workspace/tasks/classification/myocardial/config.yml"),
        Path("/workspace/tasks/classification/tic-tac-toe/config.yml"),
        Path("/workspace/tasks/classification/junglechess/config.yml"),
        Path("/workspace/tasks/classification/communities/config.yml"),  # Skipped?
        Path("/workspace/tasks/classification/eucalyptus/config.yml"),
        Path("/workspace/tasks/classification/blood/config.yml"),
        Path("/workspace/tasks/classification/car/config.yml"),
        Path("/workspace/tasks/classification/arrhythmia/config.yml"),
        Path("/workspace/tasks/classification/bank/config.yml"),
        Path("/workspace/tasks/classification/breast-w/config.yml"),
        Path("/workspace/tasks/classification/diabetes/config.yml"),
        Path("/workspace/tasks/classification/cmc/config.yml"),
        Path("/workspace/tasks/classification/adult/config.yml"),
        Path("/workspace/tasks/classification/heart/config.yml"),
        Path("/workspace/tasks/classification/vehicle/config.yml"),
        Path("/workspace/tasks/classification/credit-g/config.yml"),
        # #
        # #
        # ### Tabular Regression
        Path("/workspace/tasks/regression/forest-fires/config.yml"),
        Path("/workspace/tasks/regression/airfoil_self_noice/config.yml"),
        Path("/workspace/tasks/regression/wine/config.yml"),
        Path("/workspace/tasks/regression/plasma_retinol/config.yml"),
        Path("/workspace/tasks/regression/housing/config.yml"),
        Path("/workspace/tasks/regression/insurance/config.yml"),
        Path("/workspace/tasks/regression/crab/config.yml"),
        Path("/workspace/tasks/regression/diamonds/config.yml"),
        Path("/workspace/tasks/regression/bike/config.yml"),
    ]

    async def run_experiments():
        for p in paths:
            try:
                cfg = ExperimentConfig.from_yaml(p)
                await RogueOneAgentNetwork(cfg).main()
            except Exception as e:
                ConsoleManager.console_error_print(
                    f"Error running experiment for config {p}: {e}"
                )

    start_vllm_servers()
    asyncio.run(run_experiments())
