from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover - optional dependency guard
    plt = None

ROOT = Path(__file__).resolve().parents[2]
BACKEND = ROOT / "Backend"
DATA_PATH = BACKEND / "data" / "patent_domain_data.jsonl"
Q_FILE = ROOT / "Qs.md"
REPORT_PATH = ROOT / "EVALUATION_REPORT.md"
FIG_DIR = ROOT / "evaluation"
FIG_DIR.mkdir(exist_ok=True)

DOMAIN_ORDER = ["agriculture", "artificial-intelligence", "chemical", "sensors"]


def infer_domain(source: str | None) -> str:
    value = (source or "").lower()
    if "agriculture" in value:
        return "agriculture"
    if "artificial" in value or "ai" in value:
        return "artificial-intelligence"
    if "chemical" in value:
        return "chemical"
    if "sensor" in value:
        return "sensors"
    return "other"


def count_domain_records(path: Path) -> dict[str, int]:
    counts: Counter[str] = Counter()
    if not path.exists():
        return {domain: 0 for domain in DOMAIN_ORDER}

    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue
            source_file = (
                item.get("source_file")
                or item.get("metadata", {}).get("source_file")
                or ""
            )
            domain = infer_domain(source_file)
            if domain != "other":
                counts[domain] += 1
    ordered = {domain: counts.get(domain, 0) for domain in DOMAIN_ORDER}
    ordered["other"] = sum(counts.get(k, 0) for k in counts if k not in DOMAIN_ORDER)
    return ordered


def count_question_sections(path: Path) -> dict[str, int]:
    if not path.exists():
        return {}

    text = path.read_text(encoding="utf-8")
    sections: dict[str, int] = {}
    current = None
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if line.startswith("### "):
            current = line[4:].strip()
            sections[current] = 0
        elif current and re.match(r"^\d+\.", line):
            sections[current] += 1
    return sections


def make_heatmap(matrix: list[list[int]], x_labels: list[str], y_labels: list[str], title: str, output_path: Path) -> None:
    if plt is None:
        return

    fig, ax = plt.subplots(figsize=(8, 4.8))
    image = ax.imshow(matrix, cmap="YlOrRd", aspect="auto")
    ax.set_title(title, fontsize=12)
    ax.set_xticks(range(len(x_labels)))
    ax.set_yticks(range(len(y_labels)))
    ax.set_xticklabels(x_labels, rotation=25, ha="right")
    ax.set_yticklabels(y_labels)
    ax.set_xlabel("Evaluation axis")
    ax.set_ylabel("Evidence readiness")

    for row_index, row in enumerate(matrix):
        for col_index, value in enumerate(row):
            text_color = "black" if value < 2 else "white"
            ax.text(col_index, row_index, str(value), ha="center", va="center", color=text_color, fontsize=9)

    fig.colorbar(image, ax=ax, fraction=0.04, pad=0.04)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def generate_domain_heatmap(counts: dict[str, int], output_path: Path) -> None:
    labels = ["agriculture", "artificial-intelligence", "chemical", "sensors"]
    values = [counts.get(label, 0) for label in labels]
    matrix = [values]
    make_heatmap(matrix, labels, ["Patent corpus"], "Patent domain distribution", output_path)


def generate_readiness_heatmap(output_path: Path) -> None:
    # Each value is a readiness score based on current repository evidence, not a model claim.
    x_labels = ["Repository evidence", "Pilot-capable", "Paper-grade"]
    y_labels = [
        "Retrieval quality",
        "Grounding score",
        "Routing robustness",
        "Workflow wellbeing",
        "Human relevance labels",
    ]
    matrix = [
        [3, 2, 1],
        [3, 2, 1],
        [3, 2, 1],
        [2, 1, 0],
        [1, 0, 0],
    ]
    make_heatmap(matrix, x_labels, y_labels, "Research-evidence readiness matrix", output_path)


def format_table(values: dict[str, int]) -> str:
    rows = ["| Domain | Count |", "| --- | ---: |"]
    for key in DOMAIN_ORDER:
        rows.append(f"| {key} | {values.get(key, 0)} |")
    if values.get("other", 0):
        rows.append(f"| other | {values.get('other', 0)} |")
    return "\n".join(rows)


def build_report(domain_counts: dict[str, int], question_sections: dict[str, int]) -> str:
    total_records = sum(domain_counts.values())
    question_total = sum(question_sections.values())

    lines = [
        "# Invenza Evaluation and Research Readiness Report",
        "",
        "## Scope",
        "",
        "This report evaluates the current repository as a publication-support document rather than claiming full research completion. The goal is to establish what is measurable in the present codebase, what is still blocked, and what should be added before publishing a paper.",
        "",
        "## Verified repository baseline",
        "",
        "- Backend tests were run with `python -m pytest tests/ -v` from the `Backend` directory.",
        "- The current repository state is a verified backend baseline and not a final publication-grade benchmark.",
        "- The current implementation exposes retrieval similarity, grounding score, route/fallback decisions, and latency values, but it does not yet include human-labeled relevance or wellbeing data.",
        "",
        "## Dataset-backed domain summary",
        "",
        f"- Patent-domain dataset: `{DATA_PATH.name}`",
        f"- Estimated corpus records in the inspected domain file: {total_records}",
        "",
        format_table(domain_counts),
        "",
        "## Query-set summary",
        "",
        f"- Question count discovered in `{Q_FILE.name}`: {question_total}",
        "",
    ]

    if question_sections:
        lines.extend([
            "| Section | Questions |",
            "| --- | ---: |",
        ])
        for section_name, count in question_sections.items():
            lines.append(f"| {section_name} | {count} |")
        lines.append("")

    lines.extend([
        "## Research performance and wellbeing evaluation",
        "",
        "### Model and retrieval performance",
        "",
        "- Retrieval quality is measurable through cosine similarity, L2 distance, rank, and downstream evidence overlap.",
        "- Grounding quality is measured through the hybrid formula in `Backend/app/grounding.py`, which combines retrieval similarity, citation overlap, topic alignment, and text consistency.",
        "- The current repository supports a route/fallback design through `Backend/app/model_router.py`, but it does not contain a human-labelled benchmark for retrieval quality, generation factuality, or answer usefulness.",
        "",
        "### Wellbeing operationalization",
        "",
        "Wellbeing is operationalized conservatively in the current study as a workflow-reliability and burden assessment, not as a clinical or psychological claim. The metrics are:",
        "",
        "- latency and response time",
        "- timeout and error rate",
        "- fallback frequency",
        "- repeated-query burden",
        "- user-effort proxies in the research workflow",
        "",
        "These values are valid as operational wellbeing proxies only when reported separately from claim quality and factuality metrics.",
        "",
        "## Heatmaps generated from repository artifacts",
        "",
        "- Patent-domain heatmap: shows the corpus composition inferred from `Backend/data/patent_domain_data.jsonl`.",
        "- Research-readiness heatmap: shows the maturity of evidence across retrieval, routing, generation, workflow wellbeing, and human labels.",
        "",
        "Generated files:",
        "",
        "- `evaluation/domain_distribution_heatmap.png`",
        "- `evaluation/research_readiness_heatmap.png`",
        "",
        "## Publication recommendations",
        "",
        "1. Add a frozen, human-labeled query set with at least two annotators.",
        "2. Separate quality metrics from wellbeing metrics in all tables and figures.",
        "3. Report the current score band from `Backend/app/grounding.py` as a product threshold, not as a validated scientific result.",
        "4. Use the current repository output as a reproducible pilot and a methods framework, not as a final published claim set.",
        "",
        "## Limitations",
        "",
        "- No human relevance judgments are present yet.",
        "- No factuality or claim-support labels are present yet.",
        "- No validated wellbeing questionnaire, annotation set, or clinical outcome dataset is available in the repository.",
        "- The current heatmaps are purpose-built for evidence readiness and corpus composition, not for a full scientific performance claim.",
        "",
        "## Next research steps",
        "",
        "- Collect human annotations for query-document relevance and answer evidence support.",
        "- Run the evaluation on a fixed query set with bootstrap confidence intervals.",
        "- Publish a separate wellbeing section that reports operational burden, not psychological wellbeing.",
        "",
    ])

    return "\n".join(lines)


def main() -> None:
    domain_counts = count_domain_records(DATA_PATH)
    question_sections = count_question_sections(Q_FILE)

    domain_output = FIG_DIR / "domain_distribution_heatmap.png"
    readiness_output = FIG_DIR / "research_readiness_heatmap.png"
    generate_domain_heatmap(domain_counts, domain_output)
    generate_readiness_heatmap(readiness_output)

    report_text = build_report(domain_counts, question_sections)
    REPORT_PATH.write_text(report_text, encoding="utf-8")

    print(f"Domain counts: {domain_counts}")
    print(f"Question sections: {question_sections}")
    print(f"Report written to: {REPORT_PATH}")
    print(f"Heatmaps written to: {domain_output} and {readiness_output}")


if __name__ == "__main__":
    main()
