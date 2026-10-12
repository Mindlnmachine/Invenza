# Invenza Evaluation and Research Readiness Report

## Scope

This report evaluates the current repository as a publication-support document rather than claiming full research completion. The goal is to establish what is measurable in the present codebase, what is still blocked, and what should be added before publishing a paper.

## Verified repository baseline

- Backend tests were run with `python -m pytest tests/ -v` from the `Backend` directory.
- The current repository state is a verified backend baseline and not a final publication-grade benchmark.
- The current implementation exposes retrieval similarity, grounding score, route/fallback decisions, and latency values, but it does not yet include human-labeled relevance or wellbeing data.

## Dataset-backed domain summary

- Patent-domain dataset: `patent_domain_data.jsonl`
- Estimated corpus records in the inspected domain file: 87497

| Domain | Count |
| --- | ---: |
| agriculture | 21840 |
| artificial-intelligence | 21940 |
| chemical | 21775 |
| sensors | 21942 |

## Query-set summary

- Question count discovered in `Qs.md`: 26

| Section | Questions |
| --- | ---: |
| Discover Business Ideas | 10 |
| Improve an Idea | 8 |
| Evaluate an Idea | 8 |

## Research performance and wellbeing evaluation

### Model and retrieval performance

- Retrieval quality is measurable through cosine similarity, L2 distance, rank, and downstream evidence overlap.
- Grounding quality is measured through the hybrid formula in `Backend/app/grounding.py`, which combines retrieval similarity, citation overlap, topic alignment, and text consistency.
- The current repository supports a route/fallback design through `Backend/app/model_router.py`, but it does not contain a human-labelled benchmark for retrieval quality, generation factuality, or answer usefulness.

### Wellbeing operationalization

Wellbeing is operationalized conservatively in the current study as a workflow-reliability and burden assessment, not as a clinical or psychological claim. The metrics are:

- latency and response time
- timeout and error rate
- fallback frequency
- repeated-query burden
- user-effort proxies in the research workflow

These values are valid as operational wellbeing proxies only when reported separately from claim quality and factuality metrics.

## Heatmaps generated from repository artifacts

- Patent-domain heatmap: shows the corpus composition inferred from `Backend/data/patent_domain_data.jsonl`.
- Research-readiness heatmap: shows the maturity of evidence across retrieval, routing, generation, workflow wellbeing, and human labels.

Generated files:

- `evaluation/domain_distribution_heatmap.png`
- `evaluation/research_readiness_heatmap.png`

## Publication recommendations

1. Add a frozen, human-labeled query set with at least two annotators.
2. Separate quality metrics from wellbeing metrics in all tables and figures.
3. Report the current score band from `Backend/app/grounding.py` as a product threshold, not as a validated scientific result.
4. Use the current repository output as a reproducible pilot and a methods framework, not as a final published claim set.

## Limitations

- No human relevance judgments are present yet.
- No factuality or claim-support labels are present yet.
- No validated wellbeing questionnaire, annotation set, or clinical outcome dataset is available in the repository.
- The current heatmaps are purpose-built for evidence readiness and corpus composition, not for a full scientific performance claim.

## Next research steps

- Collect human annotations for query-document relevance and answer evidence support.
- Run the evaluation on a fixed query set with bootstrap confidence intervals.
- Publish a separate wellbeing section that reports operational burden, not psychological wellbeing.
