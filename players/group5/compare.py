"""Compare the working-tree Group 5 player against an earlier git version.

Each seed builds the same ocean for both versions, and each version runs alone
in its own engine, so both see identical zones, currents and start spots.

    uv run python -m players.group5.compare                      # vs main, seeds 1-5
    uv run python -m players.group5.compare --ref 0ceb00d --seeds 1 2 3
    uv run python -m players.group5.compare --memory 16 --days 7200
"""

from __future__ import annotations

import argparse
import importlib.util
import statistics
import subprocess
import tempfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from core.args import PLAYERS, Args
from core.geometry import torus_distance
from core.ocean import overlap_fraction
from core.runner import build_engine

_PLAYER_PATH = "players/group5/player.py"
_REPO = Path(__file__).resolve().parents[2]
_METRICS = (
    ("avg_population", "avg pop"),
    ("final_population", "final"),
    ("day_reached_100J", "day 1000"),
    ("births", "births"),
    ("deaths_predator", "eaten"),
    ("eat_days", "eat days"),
)


def _load_player(ref: str | None) -> type:
    """Load Player5 from the working tree (ref=None) or from a git ref."""
    if ref is None:
        source = (_REPO / _PLAYER_PATH).read_text()
    else:
        source = subprocess.run(
            ["git", "show", f"{ref}:{_PLAYER_PATH}"],
            cwd=_REPO,
            check=True,
            capture_output=True,
            text=True,
        ).stdout
    path = Path(tempfile.mkdtemp(prefix="group5_")) / "player.py"
    path.write_text(source)
    spec = importlib.util.spec_from_file_location(f"group5_{ref or 'worktree'}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.Player5


def _run(job: tuple[str | None, int, argparse.Namespace]) -> dict:
    ref, seed, opts = job
    PLAYERS["cmp"] = _load_player(ref)
    args = Args(
        gui=False,
        seed=seed,
        players=["cmp"],
        memory=opts.memory,
        lifetime_years=opts.lifetime,
        initial_jellyfish=opts.initial,
        days=opts.days,
        max_population=opts.max_population,
        map=None,
        json_only=True,
    )
    engine = build_engine(args)
    engine.run()
    stats = engine.groups[0].stats.to_dict()
    history = engine.groups[0].stats.history
    stats["avg_population"] = round(sum(history) / len(history))
    if stats["day_reached_100J"] is None:
        stats["day_reached_100J"] = opts.days  # never reached; counts as the whole run
    stats["error"] = engine.groups[0].error
    return stats


def _describe(seed: int, opts: argparse.Namespace) -> str:
    """One line on the seed's layout, so outliers are easy to explain."""
    ocean = build_engine(
        Args(False, seed, ["r"], opts.memory, opts.lifetime, 1, 1, 1, None, True)
    ).ocean
    f, b = ocean.feeding, ocean.breeding
    gap = torus_distance(f.x, f.y, b.x, b.y)
    return (
        f"food-nursery {gap:4.0f}km, predator covers "
        f"{overlap_fraction(ocean.predator, f):3.0%} of food, "
        f"{overlap_fraction(ocean.predator, b):3.0%} of nursery"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--ref", default="main", help="git ref of the baseline (default: main)"
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    parser.add_argument("-T", "--days", type=int, default=3600)
    parser.add_argument("-M", "--memory", type=int, default=8)
    parser.add_argument("-S", "--lifetime", type=float, default=3.0)
    parser.add_argument("-J", "--initial", type=int, default=10)
    parser.add_argument("--max_population", type=int, default=2000)
    opts = parser.parse_args()

    versions = (opts.ref, None)
    jobs = [(ref, seed, opts) for seed in opts.seeds for ref in versions]
    with ProcessPoolExecutor() as pool:
        results = list(pool.map(_run, jobs))
    by_job = {(ref, seed): r for (ref, seed, _), r in zip(jobs, results)}

    names = {opts.ref: f"old ({opts.ref})", None: "new (working tree)"}
    header = f"{'seed':>4}  {'version':<22}" + "".join(
        f"{label:>10}" for _, label in _METRICS
    )
    print(header)
    print("-" * len(header))
    for seed in opts.seeds:
        print(f"{seed:>4}  {_describe(seed, opts)}")
        for ref in versions:
            r = by_job[ref, seed]
            row = "".join(f"{r[key]:>10}" for key, _ in _METRICS)
            note = f"  CRASHED: {r['error']}" if r["error"] else ""
            print(f"{'':>4}  {names[ref]:<22}{row}{note}")
    print("-" * len(header))
    for ref in versions:
        means = "".join(
            f"{statistics.mean(by_job[ref, s][key] for s in opts.seeds):>10.0f}"
            for key, _ in _METRICS
        )
        print(f"{'mean':>4}  {names[ref]:<22}{means}")
    new_avg = [by_job[None, s]["avg_population"] for s in opts.seeds]
    old_avg = [by_job[opts.ref, s]["avg_population"] for s in opts.seeds]
    wins = sum(n > o for n, o in zip(new_avg, old_avg))
    ties = sum(n == o for n, o in zip(new_avg, old_avg))
    print(
        f"\nnew beats old on average population in {wins}/{len(opts.seeds)} seeds ({ties} ties)"
    )
    print(
        "(day 1000 = first day with 1000 jellyfish; runs that never get there show the run length)"
    )


if __name__ == "__main__":
    main()
