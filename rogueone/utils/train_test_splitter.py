# %%
from pathlib import Path
import pandas as pd

from rogueone.utils.console import ConsoleManager
from rogueone.utils.config import ExperimentConfig


class TrainTestSplitter:

    def __init__(
        self,
        cfg: ExperimentConfig,
        test_size: float = 0.2,
        random_state: int = 42,
    ):
        self.cfg = cfg
        self.test_size = test_size
        self.random_state = random_state

        labels_file_path = cfg.label_file_path

        assert labels_file_path.exists(), f"File {labels_file_path} does not exist."

        self.labels = pd.read_csv(labels_file_path)

    async def test_train_split(
        self, df: pd.DataFrame
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Split the DataFrame into train and test sets.

        Args:
            df (pd.DataFrame): The input DataFrame to split.

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: The train and test DataFrames.
        """

        local_df = df.copy()

        # Normalize self.labels to a DataFrame with column "id"
        if isinstance(self.labels, pd.DataFrame):
            tm_target = self.labels.copy()
            if "id" not in tm_target.columns and "patient_id" in tm_target.columns:
                tm_target = tm_target.rename(columns={"patient_id": "id"})
        else:
            tm_target = pd.DataFrame(self.labels, columns=["id", "target"])

        # Merge labels into local_df on the "id" column (or on the first column if "id" is absent)
        if "id" in local_df.columns:
            local_df = local_df.merge(
                tm_target.drop_duplicates(subset=["id"]),
                on="id",
                how="left",
            )
        else:
            id_col = local_df.columns[0]
            local_df = local_df.merge(
                tm_target.drop_duplicates(subset=["id"]).rename(columns={"id": id_col}),
                on=id_col,
                how="left",
            )

        if "is_train" in local_df.columns and "is_test" in local_df.columns:
            df_train = local_df[local_df["is_train"] == True].reset_index(drop=True)
            df_test = local_df[local_df["is_test"] == True].reset_index(drop=True)

        else:
            # Shuffle deterministically
            local_df = local_df.sample(
                frac=1, random_state=self.random_state
            ).reset_index(drop=True)

            sep_index = int(len(local_df) * (1 - self.test_size))

            df_train = local_df.iloc[:sep_index].reset_index(drop=True)
            df_test = local_df.iloc[sep_index:].reset_index(drop=True)

            df_train["is_train"] = True
            df_train["is_test"] = False

            df_test["is_test"] = True
            df_test["is_train"] = False

        test_group_counts = df_test.groupby("target").size().reset_index(name="count")
        train_group_counts = df_train.groupby("target").size().reset_index(name="count")

        merged_counts = pd.merge(
            train_group_counts.rename(columns={"count": "train_count"}),
            test_group_counts.rename(columns={"count": "test_count"}),
            on="target",
            how="outer",
        ).fillna(0)

        merged_counts["train_count"] = merged_counts["train_count"].astype(int)
        merged_counts["test_count"] = merged_counts["test_count"].astype(int)
        merged_counts["total_count"] = (
            merged_counts["train_count"] + merged_counts["test_count"]
        )

        ConsoleManager.print_dataframe_as_table(
            merged_counts, title="Target counts (train/test)", style="cyan"
        )

        return df_train, df_test

    async def k_fold_split(self, df: pd.DataFrame, k: int) -> list[pd.DataFrame]:
        """Perform k-fold cross-validation split.

        Args:
            df (pd.DataFrame): The input DataFrame to split.
            k (int): The number of folds.

        Returns:
            List[pd.DataFrame]: A list of k DataFrames. One for each fold.
        """

        local_df = df.copy()

        # Normalize labels to a DataFrame with a column named "id"
        if isinstance(self.labels, pd.DataFrame):
            labels_df = self.labels.copy()
            if "id" not in labels_df.columns and "patient_id" in labels_df.columns:
                labels_df = labels_df.rename(columns={"patient_id": "id"})
        else:
            labels_df = pd.DataFrame(self.labels, columns=["id", "target"])

        # Merge labels into the working DataFrame
        if "id" in local_df.columns:
            labels_to_merge = labels_df.drop_duplicates(subset=["id"])
            merged_on = "id"
        else:
            merged_on = local_df.columns[0]
            labels_to_merge = labels_df.drop_duplicates(subset=["id"]).rename(
                columns={"id": merged_on}
            )

        local_df = local_df.merge(labels_to_merge, on=merged_on, how="left")

        # Shuffle deterministically
        local_df = local_df.sample(frac=1, random_state=self.random_state).reset_index(
            drop=True
        )
        sep_length = len(local_df) // k

        folds = [local_df.iloc[i * sep_length : (i + 1) * sep_length] for i in range(k)]
        if len(local_df) % k != 0:
            folds[-1] = pd.concat(
                [folds[-1], local_df.iloc[k * sep_length :]], ignore_index=True
            )

        # Finalize folds (reset indexes)
        folds = [fold.reset_index(drop=True) for fold in folds]

        # Print distribution of targets per fold

        stats = {i: {} for i in range(k)}

        match self.cfg.task_type:
            case "classification":
                for i, fold in enumerate(folds):
                    if not fold.empty and "target" in fold.columns:
                        vc = fold.groupby("target").size().reset_index(name="count")
                        stats[i]["Fold"] = f"Fold {i+1}"
                        for _, row in vc.iterrows():
                            stats[i][row["target"]] = row["count"]

            case "regression":
                for i, fold in enumerate(folds):
                    if not fold.empty and "target" in fold.columns:
                        stats[i]["Fold"] = f"Fold {i+1}"
                        stats[i]["Mean"] = f'{fold["target"].mean():.2f}'
                        stats[i]["StdDev"] = f'{fold["target"].std():.2f}'
                        stats[i]["Min"] = f'{fold["target"].min():.2f}'
                        stats[i]["Max"] = f'{fold["target"].max():.2f}'
            case _:
                raise ValueError(f"Unsupported task type: {self.cfg.task_type}")

        if not stats:
            return

        try:
            stats_df = pd.DataFrame.from_dict(stats, orient="index").fillna(0)

            ConsoleManager.print_dataframe_as_table(
                stats_df, title="Target distribution per fold", style="cyan"
            )
        except Exception as e:
            ConsoleManager.console_print(
                f"[red]Error displaying fold statistics: {e}[/red]"
            )
        return folds


if __name__ == "__main__":
    splitter = TrainTestSplitter(
        Path("/workspaces/BrainFlux/tasks/vehicle/labels.csv"),
        test_size=0.7,
        random_state=42,
    )

    df = pd.read_csv("/workspaces/BrainFlux/tasks/vehicle/data.csv")

    train_df, test_df = splitter.test_train_split(df)
