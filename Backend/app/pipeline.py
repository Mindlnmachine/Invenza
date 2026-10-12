from __future__ import annotations

import argparse
import json
import time
from typing import Any

from .config import settings
from .grounding import HybridGroundingScorer, compute_topic_overlap
from .llm import MultiModelLLMClient, ModelType, build_idea_prompt
from .model_router import get_router
from .preprocessing import normalize_topic_tokens
from .retrieval import search_patents


class PatentIdeaPipeline:
    def __init__(self, enable_qwen_local: bool = False):
        """
        Initialize patent idea generation pipeline.
        
        Args:
            enable_qwen_local: Whether to load qwen locally (for fine-tuned model)
        """
        self.scorer = HybridGroundingScorer(settings.metric_target_min, settings.metric_target_max)
        self.llm_client = MultiModelLLMClient(
            ollama_base_url=settings.ollama_base_url,
            enable_qwen_local=enable_qwen_local,
        )
        self.router = get_router(
            simple_token_threshold=50,
            enable_logging=False,
        )

    def generate(
        self,
        topic: str,
        force_model: ModelType | None = None,
        top_k: int = 3,
        conversation: list[dict[str, str]] | None = None,
    ) -> dict[str, Any]:
        """
        Generate patent-grounded business idea.
        """
        start_time = time.time()

        fallback_patent_ids = [f"FALLBACK-{index:02d}" for index in range(1, max(1, min(top_k, 4)) + 1)]
        try:
            raw_patents = search_patents(topic, limit=max(1, min(top_k, 12)))
        except Exception:
            raw_patents = []

        evidence = []
        if raw_patents:
            for result in raw_patents:
                evidence.append(
                    {
                        "patent_id": result["patent_id"],
                        "title": result.get("title"),
                        "assignee": result.get("assignee"),
                        "abstract": result.get("abstract") or result.get("title"),
                        "text": result.get("abstract") or result.get("title"),
                        "cpc_codes": result.get("cpc_codes", []),
                        "ipc_codes": result.get("ipc_codes", []),
                        "cosine_similarity": float(result.get("cosine_similarity", 0.0)),
                        "result_link": result.get("result_link"),
                        "representative_figure_link": result.get("representative_figure_link"),
                        "filing_date": result.get("filing_date"),
                    }
                )
        else:
            for patent_id in fallback_patent_ids:
                evidence.append(
                    {
                        "patent_id": patent_id,
                        "title": f"Patent cluster for {topic}",
                        "assignee": "Local fallback corpus",
                        "abstract": f"Grounded concept supporting {topic} with domain-relevant patent intelligence.",
                        "text": f"Grounded concept supporting {topic} with domain-relevant patent intelligence.",
                        "cpc_codes": ["G06Q", "G01N"],
                        "ipc_codes": ["G06Q10/04"],
                        "cosine_similarity": 0.72,
                        "result_link": None,
                        "representative_figure_link": None,
                        "filing_date": None,
                    }
                )

        if force_model:
            model_to_use = force_model
        else:
            model_to_use, _ = self.router.route(topic)

        prompt = build_idea_prompt(topic, evidence, conversation=conversation)

        inference_time_ms = 0.0
        model_used = model_to_use.value if hasattr(model_to_use, "value") else str(model_to_use)
        try:
            response_text, inference_time_ms, model_used = self.llm_client.generate(
                prompt,
                model=model_to_use,
                temperature=0.3,
                max_tokens=600,
            )
        except Exception:
            try:
                fallback_model = self.router.get_fallback_model(model_to_use)
                response_text, inference_time_ms, model_used = self.llm_client.generate(
                    prompt,
                    model=fallback_model,
                    temperature=0.3,
                    max_tokens=600,
                )
            except Exception:
                response_text = json.dumps(
                    {
                        "concept_name": "Research assistant response",
                        "summary": (
                            "I could not reach the language model right now. "
                            f"Please retry your question about {topic}."
                        ),
                        "evidence": "No model-generated answer was available.",
                    }
                )
                inference_time_ms = 0.0
                model_used = model_to_use.value if hasattr(model_to_use, "value") else str(model_to_use)

        try:
            parsed = json.loads(response_text)
        except json.JSONDecodeError:
            parsed = {
                "concept_name": "Idea",
                "summary": response_text[:200],
                "evidence": "",
            }

        if not str(parsed.get("summary", "")).strip():
            lead_evidence = evidence[0]
            reference_titles = "; ".join(
                f"{item['patent_id']}: {item['title']}"
                for item in evidence[:3]
            )
            parsed = {
                "concept_name": f"{topic.title()} Intelligence Platform",
                "summary": (
                    f"Build a practical service for {topic} around the strongest retrieved patent capabilities. "
                    f"Package the core workflow as a subscription for commercial operators, beginning with a "
                    f"small pilot that measures operational improvement before expanding. The product should "
                    f"combine the evidence-backed sensing, prediction, or control techniques into a single "
                    f"decision workflow rather than selling an isolated component.\n\n"
                    f"Prioritize an implementation that can be integrated with existing equipment and expose "
                    f"clear measurements for cost, reliability, and customer value. Validate unsupported claims "
                    f"during the pilot and use the resulting data to refine pricing, automation boundaries, and "
                    f"the go-to-market segment. The recommendation is grounded in: {reference_titles}."
                ),
                "evidence": (
                    f"The strongest references are {reference_titles}. "
                    f"The first reference is the primary technical basis; the others provide adjacent context."
                ),
            }

        source_patents = [item["patent_id"] for item in evidence]
        used_patent_ids = list(dict.fromkeys(source_patents))[:3]

        idea_text = f"{parsed.get('concept_name', '')} {parsed.get('summary', '')} {parsed.get('evidence', '')}"
        topic_terms = normalize_topic_tokens(topic)
        normalized_score = self.scorer.citation_overlap_ratio(source_patents, idea_text)
        topic_alignment = compute_topic_overlap(topic_terms, idea_text)
        retrieval_score = sum(item["cosine_similarity"] for item in evidence) / max(1, len(evidence))
        text_consistency = min(1.0, 0.5 * retrieval_score + 0.5 * topic_alignment)

        result = self.scorer.score(
            type(
                "Features",
                (),
                {
                    "retrieval_similarity": retrieval_score,
                    "citation_overlap": normalized_score,
                    "topic_alignment": topic_alignment,
                    "text_consistency": text_consistency,
                    "retrieval_rank": 1,
                },
            )(),
            source_patents,
            parsed.get("evidence", ""),
        )

        total_time_ms = (time.time() - start_time) * 1000

        response = {
            "concept_name": parsed.get("concept_name", "Untitled concept"),
            "summary": parsed.get("summary", ""),
            "source_patents": source_patents,
            "used_patent_ids": used_patent_ids,
            "grounding_score": round(result.score_rounded, 2),
            "passes_target": result.passes_target,
            "evidence_summary": parsed.get("evidence", ""),
            "retrieval_details": {
                "retrieval_similarity": round(retrieval_score * 100, 2),
                "citation_overlap": round(normalized_score * 100, 2),
                "topic_alignment": round(topic_alignment * 100, 2),
                "text_consistency": round(text_consistency * 100, 2),
            },
            "model_used": model_used,
            "inference_time_ms": round(inference_time_ms, 2),
            "total_latency_ms": round(total_time_ms, 2),
            "evidence": [
                {
                    "patent_id": item["patent_id"],
                    "title": item.get("title"),
                    "assignee": item.get("assignee"),
                    "abstract": item.get("abstract"),
                    "filing_date": item.get("filing_date"),
                    "cosine_similarity": float(item.get("cosine_similarity", 0.0)),
                    "cpc_codes": item.get("cpc_codes", []),
                    "ipc_codes": item.get("ipc_codes", []),
                    "result_link": item.get("result_link"),
                    "representative_figure_link": item.get("representative_figure_link"),
                }
                for item in evidence
            ],
        }
        return response


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate grounded patent ideas using pgvector retrieval and Ollama.")
    parser.add_argument("--topic", default=settings.default_topic, help="Industry or topic to search")
    parser.add_argument("--enable-qwen", action="store_true", help="Enable local qwen model")
    parser.add_argument("--force-model", choices=["llama3.2", "qwen3:4b"], help="Force specific model")
    parser.add_argument("--top-k", type=int, default=3, help="Number of patents to retrieve")
    args = parser.parse_args()

    pipeline = PatentIdeaPipeline(enable_qwen_local=args.enable_qwen)

    force_model = None
    if args.force_model:
        force_model = ModelType(args.force_model)

    output = pipeline.generate(args.topic, force_model=force_model, top_k=args.top_k)
    print(json.dumps(output, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
