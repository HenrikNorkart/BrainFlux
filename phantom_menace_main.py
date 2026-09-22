from dataclasses import dataclass
import asyncio
import os

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
from rogueone.utils.console import ConsoleManager
from rogueone.llm.agents import (
    ExtractorAgent,
    ScientistAgent,
)
from rogueone.dataclasses import AttributeExplanation
from rogueone.utils.wandb_utils import init_wandb_run, wandb_logging_wrapper
from rogueone.utils.run_vllm import start_vllm_servers
from phantom_menace.agents.tester import TesterAgent


@dataclass
class TestResult:
    hypothesis: str
    score: float
    attributes_used: list[
        AttributeExplanation
    ]  # List of attribute names used in the test


class RogueOneAgentNetwork:

    def __init__(self, cfg: ExperimentConfig, precision_min=1.00, tm_frac=0.1,
                 objective_seed: int = 20260917):
        self.cfg = cfg
        self.precision_min = precision_min
        self.tm_frac = tm_frac
        # Pre-reg section 7 requires seeds 20260916/17/18 with PASS at 2 of 3.
        # This was a hardcoded default inside TesterAgent, unreachable from
        # config, env or CLI, so the three-seed design was not expressible at
        # all -- which is why bf-newobj is n=1.
        self.objective_seed = objective_seed

        ConsoleManager.console_rule("Setting up Weights & Biases Logging")
        init_wandb_run(
            cfg,
            precision_min=precision_min,
            tm_frac=tm_frac,
            objective_seed=objective_seed,
        )

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
        self.tester_agent = TesterAgent(
            self.cfg,
            precision_min=precision_min,
            tm_frac=tm_frac,
            objective_seed=objective_seed,
        )

        self.df_attributes_explanations = pd.DataFrame(
            columns=["Attribute", "Description", "Pandas Command", "Status", "Added"]
        )

        self.focus_history = []

    def update_attribute_pool(self, attribute_explanations: pd.DataFrame):
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

        self.update_attribute_pool(_attribute_explanations)

        ConsoleManager.console_print(f"Extracted {num_new_attributes} attributes.\n")

        ######### Splitting Patient Data #########
        # ConsoleManager.console_rule("Splitting Data")
        # ConsoleManager.progress_bar_update(description="Splitting data")

        # match self.cfg.modality:
        #     case "tabular":
        #         df_entities_folds = self.splitter.k_fold_split(
        #             self.df_entities_attributes, k=rouge_one_cfg.k_folds
        #         )
        #     case "time_series":
        #         df_entities_folds = self.splitter.test_train_split(
        #             self.df_entities_attributes
        #         )
        #     case _:
        #         raise ValueError(f"Unsupported modality: {self.cfg.modality}")

        ######### Generating and Testing Hypotheses #########
        ConsoleManager.console_rule("Generating and Testing Hypotheses")
        ConsoleManager.progress_bar_update(description="Testing hypotheses")

        self.df_entities_attributes.to_csv("/workspace/dummy.csv", index=False)

        test_results, _attribute_explanations = await self.tester_agent.test_hypotheses(
            [self.df_entities_attributes],
            self.df_attributes_explanations,
            step=index,
        )
        self.update_attribute_pool(_attribute_explanations)
        test_results = test_results.as_dict.copy()
        test_results["id"] = "Trial " + str(index)
        # AC3: without this a pooled three-seed run cannot be split by seed.
        test_results["objective_seed"] = self.objective_seed

        if self.df_test_pool is None:
            self.df_test_pool = pd.DataFrame([test_results])
        else:
            self.df_test_pool = pd.concat(
                [self.df_test_pool, pd.DataFrame([test_results])],
                ignore_index=True,
            )

        report = test_results.pop("report")

        self.dump_data(index)
        self.save_report(report, index)
        ConsoleManager.print_dict_as_table(test_results)
        # ConsoleManager.console_print(report)

    def dump_data(self, step: int):
        dst_dir = self.cfg.output_dir / "intermediate_steps"
        if not dst_dir.exists():
            dst_dir.mkdir(parents=True)

        self.df_entities_attributes.to_csv(
            dst_dir / f"df_entities_attributes_trail_{step}.csv",
            index=False,
        )

    def save_report(self, report: str | None, index: int):
        if report is None:
            return
        dir = self.cfg.output_dir / "tester_reports"
        if not dir.exists():
            dir.mkdir(parents=True)
        dst_path = dir / f"tester_report_iteration_{index}.md"
        with open(dst_path, "w") as f:
            f.write(report)

    def save_data(self):
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

            for i in range(1, rouge_one_cfg.num_iterations + 1):

                try:
                    ConsoleManager.progress_bar_update(completed=i)
                    await self.forward_pass(i)
                except Exception as e:
                    ConsoleManager.console_print(
                        f"Error during iteration {i}: {e}", style="bold red"
                    )
                    continue  # Skip to the next iteration on error
                # break  # For now, only one iteration
            ConsoleManager.console_rule("Final Results")
            self.save_data()
            # ConsoleManager.print_dataframe_as_table(
            #     self.df_test_pool.iloc[:, :5], title="Final Test Pool (First 5 Columns)"
            # )

        @wandb_logging_wrapper
        def log_final_results():
            import wandb

            wandb.finish()

        log_final_results()


if __name__ == "__main__":
    start_vllm_servers()

    # The pre-registration (tmp/t2_prereg.md, sections 3 and 4) fixes ONE arm:
    # precision floor 0.95, top-p 0.1. The previous nested sweep ran
    # [0.90, 0.95, 0.99, 1.00] x [0.1, 0.3, 0.5, 0.7] = 16 cells, which re-opened
    # exactly the garden of forking paths the ICTAI result died of, and was most of
    # the reason bf-newobj never finished: 16 cells against an objective costing
    # ~7000 classifier fits per tester pass.
    #
    # Read from the environment rather than hardcoded, so the arm is explicit and
    # a secondary analysis can be run deliberately instead of by editing literals.
    # NOTE: if a floor sweep is ever wanted again as a SECONDARY analysis, 0.98 --
    # the manuscript's own operating point -- must be in it. It was absent from the
    # old grid entirely, which A6 flagged.
    PRECISION_MIN = float(os.getenv("ROGUE_ONE_PRECISION_MIN", "0.95"))
    TM_FRAC = float(os.getenv("ROGUE_ONE_TM_FRAC", "0.1"))
    # Pre-reg section 7: 20260916 / 20260917 / 20260918, PASS at 2 of 3.
    # One run carries one seed; the three are separate runs.
    OBJECTIVE_SEED = int(os.getenv("ROGUE_ONE_OBJECTIVE_SEED", "20260917"))

    ConsoleManager.console_print(
        f"PRE-REGISTERED ARM: precision_min={PRECISION_MIN}, tm_frac={TM_FRAC}, "
        f"objective_seed={OBJECTIVE_SEED} "
        f"(single configuration, not a sweep -- see tmp/t2_prereg.md). "
        f"Both values are also written to the WandB run config.",
        style="bold magenta",
    )

    async def run_experiments():
        # This is a retry-on-exception guard that stops at the first success. It is
        # NOT the pre-registered three-seed replication -- the old `for _ in range(3)`
        # spelling was read that way at least once. Replication is a separate concern
        # and is driven by objective_seed (F3/#150), not by this loop.
        MAX_LAUNCH_ATTEMPTS = 3
        last_error: Exception | None = None

        for attempt in range(1, MAX_LAUNCH_ATTEMPTS + 1):
            try:
                p = Path("/workspace/phantom_menace/config.yml")
                cfg = ExperimentConfig.from_yaml(p)

                await RogueOneAgentNetwork(
                    cfg,
                    precision_min=PRECISION_MIN,
                    tm_frac=TM_FRAC,
                    objective_seed=OBJECTIVE_SEED,
                ).main()
                return
            except Exception as e:  # noqa: BLE001 - retried, then re-raised below
                last_error = e
                ConsoleManager.console_print(
                    f"Launch attempt {attempt}/{MAX_LAUNCH_ATTEMPTS} failed: {e}",
                    style="bold red",
                )

        # Previously the loop fell through silently, so a run that never started
        # looked the same as one that finished.
        raise RuntimeError(
            f"All {MAX_LAUNCH_ATTEMPTS} launch attempts failed; last error: {last_error}"
        )

    asyncio.run(run_experiments())
