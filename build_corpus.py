"""
Build a one-document-per-occupation text corpus from the O*NET 31.0 database,
joined on O*NET-SOC Code. Output: data/corpus.csv (soc_code, title, description, document).

Importance threshold: skills/knowledge elements are only included if their
Scale ID == "IM" (Importance, 1-5) value is >= IM_THRESHOLD. 3.5 sits between
O*NET's "Important" (3) and "Very Important" (4) anchors. Below this, occupations
end up listing nearly every element and the text stops discriminating between them.

Per-occupation field caps (tasks, software examples, alternate titles) compress the
document-length spread: without them, broad occupations like Software Developers
(300+ technologies) end up so long that TF-IDF's L2 normalization dilutes every
individual term match, and they lose to narrower, shorter occupations under cosine
similarity even when they're the better match. Capping keeps the highest-signal
entries (Core tasks over Supplemental, Hot/In-Demand technologies over long-tail ones)
and drops the rest.

The Title + alternate-titles block is repeated 3x in the document: occupation identity
is the strongest signal for matching a resume to a job, and it would otherwise be
drowned out by thousands of characters of task/skill/software text.
"""

import pandas as pd

RAW = "data/raw/db_31_0_csv"
IM_THRESHOLD = 3.5
MAX_TASKS = 30
MAX_SOFTWARE = 80
MAX_ALT_TITLES = 40
TITLE_BLOCK_REPEATS = 3


def read_csv(name):
    return pd.read_csv(f"{RAW}/{name}", encoding="utf-8")


def joined_by_soc(df, col, dedupe=True):
    """Group a dataframe's `col` values by O*NET-SOC Code, returning {soc: [values]}."""
    if dedupe:
        df = df.drop_duplicates(subset=["O*NET-SOC Code", col])
    grouped = df.groupby("O*NET-SOC Code")[col].apply(list)
    return grouped.to_dict()


def capped_by_soc(df, col, cap):
    """Like joined_by_soc, but truncate each occupation's list to the first `cap`
    entries in the dataframe's current row order. Callers sort/prioritize first."""
    df = df.drop_duplicates(subset=["O*NET-SOC Code", col])
    grouped = df.groupby("O*NET-SOC Code")[col].apply(lambda s: list(s)[:cap])
    return grouped.to_dict()


def importance_filtered(df, threshold=IM_THRESHOLD):
    return df[(df["Scale ID"] == "IM") & (df["Data Value"] >= threshold)]


def prioritized_tasks(tasks_df):
    """Core tasks before Supplemental before untyped; within a tier, higher
    Incumbents Responding (more validated by survey respondents) first."""
    df = tasks_df.dropna(subset=["Task"]).copy()
    type_rank = {"Core": 0, "Supplemental": 1}
    df["_type_rank"] = df["Task Type"].map(type_rank).fillna(2)
    return df.sort_values(
        ["_type_rank", "Incumbents Responding"],
        ascending=[True, False],
        kind="stable",
    )


def prioritized_software(software_df):
    """Hot Technology / In Demand examples before long-tail ones."""
    df = software_df.copy()
    df["_flagged"] = ~((df["Hot Technology"] == "Y") | (df["In Demand"] == "Y"))
    return df.sort_values("_flagged", kind="stable")


def main():
    occupations = read_csv("occupation_data.csv")
    tasks = read_csv("task_statements.csv")
    essential_skills = read_csv("essential_skills.csv")
    transferable_skills = read_csv("transferable_skills.csv")
    software_skills = read_csv("software_skills.csv")
    knowledge = read_csv("knowledge.csv")
    job_titles = read_csv("job_titles.csv")

    tasks_by_soc = capped_by_soc(prioritized_tasks(tasks), "Task", MAX_TASKS)
    alt_titles_by_soc = capped_by_soc(job_titles, "Job Title", MAX_ALT_TITLES)
    software_by_soc = capped_by_soc(prioritized_software(software_skills), "Workplace Example", MAX_SOFTWARE)

    essential_by_soc = joined_by_soc(importance_filtered(essential_skills), "Element Name")
    transferable_by_soc = joined_by_soc(importance_filtered(transferable_skills), "Element Name")
    knowledge_by_soc = joined_by_soc(importance_filtered(knowledge), "Element Name")

    rows = []
    for _, occ in occupations.iterrows():
        soc = occ["O*NET-SOC Code"]
        title = occ["Title"]
        description = occ["Description"]

        alt_titles = alt_titles_by_soc.get(soc, [])
        occ_tasks = tasks_by_soc.get(soc, [])
        occ_essential = essential_by_soc.get(soc, [])
        occ_transferable = transferable_by_soc.get(soc, [])
        occ_knowledge = knowledge_by_soc.get(soc, [])
        occ_software = software_by_soc.get(soc, [])

        title_block = title
        if alt_titles:
            title_block += " Also known as: " + "; ".join(alt_titles) + "."
        title_block_repeated = " ".join([title_block] * TITLE_BLOCK_REPEATS)

        parts = [
            title_block_repeated,
            description,
            "Tasks: " + " ".join(occ_tasks) if occ_tasks else "",
            "Essential skills: " + ", ".join(occ_essential) + "." if occ_essential else "",
            "Transferable skills: " + ", ".join(occ_transferable) + "." if occ_transferable else "",
            "Knowledge areas: " + ", ".join(occ_knowledge) + "." if occ_knowledge else "",
            "Technologies and software: " + ", ".join(occ_software) + "." if occ_software else "",
        ]
        document = " ".join(p for p in parts if p)

        rows.append({
            "soc_code": soc,
            "title": title,
            "description": description,
            "document": document,
        })

    corpus = pd.DataFrame(rows, columns=["soc_code", "title", "description", "document"])
    corpus.to_csv("data/corpus.csv", index=False, encoding="utf-8")

    lengths = corpus["document"].str.len()
    print(f"Wrote data/corpus.csv with {len(corpus)} rows")
    print(f"Document length (chars): min={lengths.min()}, median={lengths.median()}, max={lengths.max()}")


if __name__ == "__main__":
    main()
