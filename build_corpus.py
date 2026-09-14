"""
Build a one-document-per-occupation text corpus from the O*NET 31.0 database,
joined on O*NET-SOC Code. Output: data/corpus.csv (soc_code, title, description, document).

Importance threshold: skills/knowledge elements are only included if their
Scale ID == "IM" (Importance, 1-5) value is >= IM_THRESHOLD. 3.5 sits between
O*NET's "Important" (3) and "Very Important" (4) anchors. Below this, occupations
end up listing nearly every element and the text stops discriminating between them.
"""

import pandas as pd

RAW = "data/raw/db_31_0_csv"
IM_THRESHOLD = 3.5


def read_csv(name):
    return pd.read_csv(f"{RAW}/{name}", encoding="utf-8")


def joined_by_soc(df, col, dedupe=True):
    """Group a dataframe's `col` values by O*NET-SOC Code, returning {soc: [values]}."""
    if dedupe:
        df = df.drop_duplicates(subset=["O*NET-SOC Code", col])
    grouped = df.groupby("O*NET-SOC Code")[col].apply(list)
    return grouped.to_dict()


def importance_filtered(df, threshold=IM_THRESHOLD):
    return df[(df["Scale ID"] == "IM") & (df["Data Value"] >= threshold)]


def main():
    occupations = read_csv("occupation_data.csv")
    tasks = read_csv("task_statements.csv")
    essential_skills = read_csv("essential_skills.csv")
    transferable_skills = read_csv("transferable_skills.csv")
    software_skills = read_csv("software_skills.csv")
    knowledge = read_csv("knowledge.csv")
    job_titles = read_csv("job_titles.csv")

    tasks_by_soc = joined_by_soc(tasks, "Task")
    alt_titles_by_soc = joined_by_soc(job_titles, "Job Title")
    software_by_soc = joined_by_soc(software_skills, "Workplace Example")

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

        parts = [
            title,
            description,
            "Also known as: " + "; ".join(alt_titles) + "." if alt_titles else "",
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
    print(f"Wrote data/corpus.csv with {len(corpus)} rows")


if __name__ == "__main__":
    main()
