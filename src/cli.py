"""
LLM Forge CLI -- production command-line interface for the benchmarking engine.

Commands:
  benchmark  --model <name> [--url URL] [--api-key KEY] [--category CAT]
             [--difficulty 1-3] [--workers N] [--output FILE]
                             Run full benchmark against a local or remote model.

  compare    --models a,b,c [--url URL] [--api-key KEY] [--category CAT]
                             Run the same battery against multiple models and
                             print a side-by-side comparison table.

  list-prompts [--category CAT] [--difficulty N]
                             List available prompts in the library.

  health     --url URL [--api-key KEY]
                             Check if the backend is reachable.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from core.client import LLMClient
from core.evaluator import BenchmarkRunner, compare_models, print_comparison_table
from core.prompt_library import (
    get_all, get_by_category, get_by_difficulty, get_categories, summary,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("LLMForge")

# ─── Helpers ───────────────────────────────────────────────────────────────────
def _build_client(url: str, api_key: str | None = None) -> LLMClient:
    return LLMClient(base_url=url, api_key=api_key)


def _select_prompts(category: str | None = None, difficulty: int | None = None):
    if category and difficulty:
        return [p for p in get_by_category(category) if p.difficulty == difficulty]
    if category:
        return get_by_category(category)
    if difficulty:
        return get_by_difficulty(difficulty)
    return get_all()

# ─── Command handlers ─────────────────────────────────────────────────────────
def cmd_health(args: argparse.Namespace) -> None:
    client = _build_client(args.url, getattr(args, "api_key", None))
    ok = client.is_healthy()
    status = "HEALTHY" if ok else "UNREACHABLE"
    print(f"\n  Backend {args.url}: [{status}]\n")
    if ok:
        models = client.list_models()
        if models:
            print(f"  Available models ({len(models)}):")
            for m in models:
                print(f"    - {m}")
    sys.exit(0 if ok else 1)


def cmd_benchmark(args: argparse.Namespace) -> None:
    client  = _build_client(args.url, getattr(args, "api_key", None))
    prompts = _select_prompts(args.category, args.difficulty)

    if not prompts:
        logger.error("No prompts match the selected filters.")
        sys.exit(1)

    logger.info("Model: %s | Prompts: %d | Workers: %d", args.model, len(prompts), args.workers)

    if not client.is_healthy():
        logger.error("Backend unreachable at %s. Is Ollama/vLLM running?", args.url)
        sys.exit(1)

    runner = BenchmarkRunner(
        client=client,
        model=args.model,
        max_workers=args.workers,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
    )

    report = runner.run(prompts)
    report.print_summary()

    out_path = args.output or f"report_{args.model}_{report.timestamp[:10]}.json"
    report.save_json(out_path)
    print(f"  Full report saved to: {out_path}\n")


def cmd_compare(args: argparse.Namespace) -> None:
    model_names = [m.strip() for m in args.models.split(",")]
    prompts     = _select_prompts(args.category, args.difficulty)

    if not prompts:
        logger.error("No prompts match the selected filters.")
        sys.exit(1)

    client = _build_client(args.url, getattr(args, "api_key", None))

    if not client.is_healthy():
        logger.error("Backend unreachable at %s", args.url)
        sys.exit(1)

    clients_models = [(client, m) for m in model_names]
    reports = compare_models(clients_models, prompts, max_workers=args.workers)

    for model, report in reports.items():
        report.print_summary()

    print_comparison_table(reports)

    # ─── Fix: save combined output correctly ──────────────────────────────
    if args.output:
        combined = {
            model: json.loads(report.to_json())
            for model, report in reports.items()
        }
        Path(args.output).write_text(json.dumps(combined, indent=2))
        print(f"  Combined report saved to: {args.output}\n")


def cmd_list_prompts(args: argparse.Namespace) -> None:
    prompts = _select_prompts(args.category, args.difficulty)

    print(f"\n  Prompt Library -- {len(prompts)} prompts")
    print(f"  Categories: {', '.join(get_categories())}")
    print(f"  Summary: {summary()}\n")
    print(f"  {'ID':>8}  {'Cat':>14}  {'D':>2}  {'Expected':>10}  Prompt")
    print("  " + "-" * 80)

    for p in prompts:
        snippet = p.text[:55].replace("\n", " ")
        print(f"  {p.id:>8}  {p.category:>14}  {p.difficulty:>2}  {p.expected_behavior:>10}  {snippet}...")

    print()

def main() -> None:
    parser = argparse.ArgumentParser(
        prog="llm-forge",
        description="LLM Deployments Forge -- Red-Team Benchmark Engine",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # health
    p_health = sub.add_parser("health", help="Check backend connectivity")
    p_health.add_argument("--url",     default="http://localhost:11434", help="Backend URL")
    p_health.add_argument("--api-key", default=None, dest="api_key",
                          help="Bearer API key for authenticated backends")

    # benchmark
    p_bench = sub.add_parser("benchmark", help="Benchmark a single model")
    p_bench.add_argument("--model",       required=True, help="Model name (e.g. llama3, mistral)")
    p_bench.add_argument("--url",         default="http://localhost:11434", help="Backend URL")
    p_bench.add_argument("--api-key",     default=None, dest="api_key",
                         help="Bearer API key for authenticated backends (e.g. OpenAI-compatible)")
    p_bench.add_argument("--category",    default=None, help="Filter by category")
    p_bench.add_argument("--difficulty",  type=int, default=None, choices=[1, 2, 3])
    p_bench.add_argument("--workers",     type=int, default=4, help="Concurrent workers (default: 4)")
    p_bench.add_argument("--temperature", type=float, default=0.7)
    p_bench.add_argument("--max-tokens",  type=int, default=512, dest="max_tokens")
    p_bench.add_argument("--output",      default=None, help="Output JSON path")

    # compare
    p_compare = sub.add_parser("compare", help="Compare multiple models side by side")
    p_compare.add_argument("--models",    required=True, help="Comma-separated model names")
    p_compare.add_argument("--url",       default="http://localhost:11434", help="Backend URL")
    p_compare.add_argument("--api-key",   default=None, dest="api_key",
                           help="Bearer API key for authenticated backends")
    p_compare.add_argument("--category",  default=None)
    p_compare.add_argument("--difficulty", type=int, default=None, choices=[1, 2, 3])
    p_compare.add_argument("--workers",   type=int, default=4)
    p_compare.add_argument("--output",    default=None, help="Combined report JSON path")

    # list-prompts
    p_list = sub.add_parser("list-prompts", help="List available prompts")
    p_list.add_argument("--category",   default=None)
    p_list.add_argument("--difficulty", type=int, default=None, choices=[1, 2, 3])

    args = parser.parse_args()
    dispatch = {
        "health":       cmd_health,
        "benchmark":    cmd_benchmark,
        "compare":      cmd_compare,
        "list-prompts": cmd_list_prompts,
    }
    dispatch[args.command](args)


if __name__ == "__main__":
    main()
