# %%
import pandas as pd
import numpy as np

import os

def load_all_llm_answers_from_json(
    answers_save_path: str,
    prefix_replace="auto-eval",
    sub_folders=[""],
) -> list[dict[pd.DataFrame]]:
    # reload all the scored answers from json files
    all_llm_answers = {}
    for sub_folder in sub_folders:
        answers_save_path_sub = f"{answers_save_path}{sub_folder}"
        if not os.path.exists(answers_save_path_sub):
            continue
        for output_file in os.listdir(f"{answers_save_path_sub}/"):
            if output_file.endswith(".json"):
                outputs_df = pd.read_json(
                    f"{answers_save_path_sub}/{output_file}", orient="index"
                )
                model = output_file.replace(prefix_replace, "").replace(".json", "")
                all_llm_answers.setdefault(model, pd.DataFrame())
                all_llm_answers[model] = pd.concat([all_llm_answers[model], outputs_df])
    print("test statistics, all answers", all_llm_answers)
    return all_llm_answers


def calculate_llm_stats(all_llm_answers: dict, bootstrap_n=10000) -> dict:
    all_llm_stats = {}
    for model, outputs in all_llm_answers.items():
        print(f"Calculating stats for {model}")
        mean_score = outputs["score"].mean()
        std_dev_score = outputs["score"].std()
        # do a n(10,000) bootstrap to get the 95% CI
        bootstrap_scores = []
        for _ in range(bootstrap_n):
            bootstrap_scores.append(
                outputs["score"].sample(frac=1, replace=True).mean()
            )
        ci_lower = np.percentile(bootstrap_scores, 2.5)
        ci_upper = np.percentile(bootstrap_scores, 97.5)
        # caculate z-interval 95%
        z = 1.96
        z_interval_error = z * (std_dev_score / np.sqrt(len(outputs)))
        all_llm_stats[model] = {
            "mean_score": mean_score,
            "std_dev_score": std_dev_score,
            "z_interval_error": z_interval_error,
            "ci_lower": ci_lower,
            "ci_upper": ci_upper,
            "output_count": len(outputs),
        }
    return all_llm_stats


def get_llm_stats(
    all_llm_answers: dict, stats_save_path: str, file_suffix="", bootstrap_n=10000
) -> pd.DataFrame:
    all_llm_stats = calculate_llm_stats(all_llm_answers, bootstrap_n)
    stats_df = (
        pd.DataFrame(all_llm_stats)
        .transpose()
        .sort_values("mean_score", ascending=False)
    )
    stats_df.index.name = "model"
    os.makedirs(stats_save_path, exist_ok=True)
    stats_df.to_csv(f"./{stats_save_path}/final_stats{file_suffix}.csv")
    return stats_df


