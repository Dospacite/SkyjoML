"""Command-line interface: ``skyjo validate|benchmark|train|evaluate|play|replay``."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np


def _threads(n: int) -> None:
    import torch

    torch.set_num_threads(n)


def cmd_validate(a) -> int:
    from .envs.validate import validate
    from .game.rules import SOURCES, ruleset_hash
    from .training.manifest import schema_hash

    print(json.dumps({"ruleset_hash": ruleset_hash(), "schema_hash": schema_hash(), "sources": SOURCES}, indent=1))
    res = validate(a.matches, seed=a.seed)
    print(json.dumps(res, indent=1))
    if a.out:
        with open(a.out, "w") as f:
            json.dump(res, f, indent=1)
    return 0 if res["ok"] else 1


def cmd_benchmark(a) -> int:
    from .agents.base import HeuristicBatchPolicy
    from .envs.batch import BatchEnv
    from .evaluation.tournament import run_matches, seat_rotation_schedule
    from .game.rules import MAX_PLAYERS, MIN_PLAYERS
    from .training.manifest import SEED_RANGES

    out = {"engine_transitions_per_s": {}, "heuristic_baselines": {}}
    rng = np.random.default_rng(0)
    for n in range(MIN_PLAYERS, MAX_PLAYERS + 1):
        env = BatchEnv(1024)
        env.reset(np.arange(1024), np.arange(1024), np.full(1024, n))
        steps = 0
        t0 = time.time()
        while time.time() - t0 < 2.0:
            obs, masks = env.observe()
            # uniform random legal actions, vectorised
            r = rng.random(masks.shape) * masks
            acts = r.argmax(1)
            env.step(acts)
            over = np.flatnonzero(env.over)
            if len(over):
                env.reset(over, rng.integers(1 << 40, size=len(over)), np.full(len(over), n))
            steps += 1024
        out["engine_transitions_per_s"][n] = round(steps / (time.time() - t0))
    names = ["random", "greedy", "column", "finish"]
    pols = [HeuristicBatchPolicy(x, seed=i) for i, x in enumerate(names)]
    for n in a.players:
        base = SEED_RANGES["dev"][0] + 7_000_000 + 1000 * n
        for ci, cand in enumerate(names[1:], start=1):
            for oi, opp in enumerate(names[:ci]):
                seeds, seat = seat_rotation_schedule(np.arange(base, base + a.seeds), n, [ci] + [oi] * (n - 1))
                r = run_matches(pols, seeds, n, seat)
                out["heuristic_baselines"][f"{n}p {cand} vs {opp}"] = round(float(r.credit_of(ci).mean() * n), 3)
    print(json.dumps(out, indent=1))
    return 0


def cmd_train(a) -> int:
    from .training.ppo import PPOTrainer, TrainConfig

    cfg = TrainConfig.from_toml(a.config, run_name=a.run_name, seed=a.seed, init_from=a.init_from)
    PPOTrainer(cfg, resume=a.resume).train()
    return 0


def cmd_evaluate(a) -> int:
    from .evaluation.benchmark import run_benchmark

    res = run_benchmark(a.checkpoint, players=a.players, n_seeds=a.seeds, split=a.split, out=a.out,
                        greedy=not a.sample, opponents=a.opponents)
    print(json.dumps(res["summary"], indent=1))
    return 0


def _load_agent(path: str | None, greedy: bool = True, search: bool = False, rollouts: int = 256,
                deadline: float | None = 1.0):
    from .agents.base import HeuristicBatchPolicy
    from .agents.neural import NeuralBatchPolicy, load_model

    if path is None or path in {"random", "greedy", "column", "finish"}:
        return HeuristicBatchPolicy(path or "finish")
    net, _ = load_model(path)
    if search:
        from .search.root_mc import RootMCSearch, SearchConfig

        cfg = SearchConfig(rollouts=rollouts, top_k=5, horizon=16, deadline_s=deadline, skip_confident=0.95)
        return RootMCSearch(net, cfg, seed=int(time.time()), name=os.path.basename(path) + "+search")
    return NeuralBatchPolicy(net, os.path.basename(path), greedy=greedy)


def cmd_play(a) -> int:
    from .game import obs_layout as L
    from .game.actions import A_DRAW, A_KEEP, A_REVEAL, A_TAKE, PAIR_INDEX, describe
    from .game.replay import Recorder, render_board

    agent = _load_agent(a.checkpoint, search=a.search, rollouts=a.rollouts)
    n = a.players
    seed = a.seed if a.seed is not None else int(time.time())
    rec = Recorder(n, seed, agents=["you"] + [agent.name] * (n - 1))
    g = rec.game
    human = 0

    def show():
        print()
        print(f"Round {g.round + 1}   totals: " + "  ".join(f"P{p}{'(you)' if p == human else ''}={g.scores[p]}"
                                                         for p in range(n)))
        for p in range(n):
            tag = "you" if p == human else f"P{p}"
            print(f"  {tag:>4}: " + " | ".join(render_board(g, p)))
        top = g.discard[-1] if g.discard else None
        print(f"  discard top: {top}   draw pile: {len(g.draw)}"
              + (f"   final turns left: {g.final_left}" if g.finisher >= 0 else ""))

    def parse_slot(tok: str) -> int:
        tok = tok.strip().lower().replace("c", "").replace("r", " ").split()
        c, r = int(tok[0]), int(tok[1])
        return (c - 1) * 3 + (r - 1)

    print("Slots are named by column and row, e.g. '2 3' = column 2, row 3. Hidden cards show as '#'.")
    while not g.over:
        if g.cur != human:
            obs, mask = g.observation(), g.legal_mask()
            act, _, diag = agent.act(obs, mask)
            if g.phase != L.PH_SETUP:
                print(f"  P{g.cur}: {describe(act)}"
                      + (f" (drew {g.draw[-1]})" if act == A_DRAW and g.draw else ""))
            rec.step(act)
            continue
        show()
        mask = g.legal_mask()
        try:
            if g.phase == L.PH_SETUP:
                s = input("Reveal two cards (e.g. '1 1, 2 1'): ").split(",")
                act = PAIR_INDEX[parse_slot(s[0]), parse_slot(s[1])]
            elif g.phase == L.PH_SELECT:
                s = input("[d]raw from deck, or 't c r' to take the discard into column c row r: ").strip()
                act = A_DRAW if s.startswith("d") else A_TAKE + parse_slot(s[1:])
            else:
                print(f"  You drew {g.pending}.")
                s = input("'k c r' keep it at c r, or 'r c r' discard it and reveal c r: ").strip()
                act = (A_KEEP if s.startswith("k") else A_REVEAL) + parse_slot(s[1:])
        except (ValueError, IndexError):
            print("  could not parse that, try again")
            continue
        if act < 0 or not mask[act]:
            print("  illegal move, try again")
            continue
        res = rec.step(int(act))
        if res.round_ended:
            print(f"\n*** round over: {res.round_scores}  totals: {g.scores}")
    print(f"\nMatch over. Totals: {g.scores}. Win credit: {g.win_credit()}")
    if a.save:
        rec.save(a.save)
        print("saved replay to", a.save)
    return 0


def cmd_advise(a) -> int:
    """Advise one seat at a physical table from manually entered public events and own draws."""
    from .game import obs_layout as L
    from .game.actions import A_SETUP, SETUP_PAIRS, describe
    from .game.tracker import TableTracker

    agent = _load_agent(a.checkpoint, search=a.search, rollouts=a.rollouts)
    n, me = a.players, a.seat

    def ask(prompt: str) -> list[str]:
        return input(prompt).replace(",", " ").split()

    def slot(c: str, r: str) -> int:
        return (int(c) - 1) * 3 + (int(r) - 1)

    def suggest(t: TableTracker) -> int:
        act, _, diag = agent.act(t.observation(), t.legal_mask())
        extra = f"  (win-credit estimate {diag['value']:.2f})" if "value" in diag else ""
        print(f"  >>> suggestion: {describe(act)}{extra}")
        return act

    print("Slots are 'column row' (1-based). Values are card numbers. Seats are 0..n-1 clockwise.")
    t = TableTracker(n, me, int(ask("First discard card value: ")[0]))
    while True:
        # ---- setup
        act = suggest(t)
        i, j = SETUP_PAIRS[act - A_SETUP]
        print(f"  reveal your cards at c{i // 3 + 1}r{i % 3 + 1} and c{j // 3 + 1}r{j % 3 + 1}")
        for p in range(n):
            x = ask(f"Seat {p} revealed (c r value c r value): ")
            s1, s2 = slot(x[0], x[1]), slot(x[3], x[4])
            t.setup(p, (s1, s2))
            t.reveal_setup_values(p, (s1, s2), (int(x[2]), int(x[5])))
        t.start(int(ask("Seat taking the first turn: ")[0]))
        # ---- turns
        while t.phase != L.PH_OVER:
            p = t.cur
            if p == me:
                suggest(t)
            x = ask(f"Seat {p}{' (you)' if p == me else ''}: 't c r replaced' or 'd': ")
            try:
                if x[0] == "t":
                    t.take(slot(x[1], x[2]), int(x[3]))
                    continue
                if p == me:
                    t.draw(int(ask("  value you drew: ")[0]))
                    suggest(t)
                else:
                    t.draw()
                y = ask("  'k c r drawn replaced' or 'r c r drawn revealed': ")
                if y[0] == "k":
                    t.keep(slot(y[1], y[2]), int(y[3]), int(y[4]))
                else:
                    t.reveal(slot(y[1], y[2]), int(y[3]), int(y[4]))
            except (ValueError, IndexError, AssertionError) as e:
                print(f"  could not apply that ({e}); the tracker state is unchanged only if parsing failed")
        rs = [int(v) for v in ask(f"Round over. Enter the {n} round scores in seat order: ")]
        t.round_scores(rs)
        print(f"  totals: {t.scores}")
        if t.match_over:
            print("Match over.")
            return 0
        t.new_round(int(ask("New round - first discard card value: ")[0]))


def cmd_export(a) -> int:
    from .export import export

    meta = export(a.checkpoint, a.out, benchmark=a.benchmark, notes=a.notes)
    print(json.dumps({k: meta[k] for k in ("ruleset_hash", "schema_hash", "sha256")}, indent=1))
    return 0


def cmd_replay(a) -> int:
    from .game import replay as R

    d = R.load(a.file)
    g = R.verify(d)
    print(f"replay OK: {d['n_players']} players, seed {d['seed']}, {len(d['actions'])} decisions, "
          f"totals {g.scores}")
    if a.verbose:
        for e in d["public_trace"]:
            print(f"P{e['actor']}: {e['desc']}  {json.dumps(e['events']) if e['events'] else ''}")
    return 0


def main(argv: list[str] | None = None) -> int:
    os.environ.setdefault("NUMBA_NUM_THREADS", "2")
    ap = argparse.ArgumentParser(prog="skyjo")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("validate", help="randomized invariant gate over complete matches")
    p.add_argument("--matches", type=int, default=100_000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out")
    p.set_defaults(fn=cmd_validate)

    p = sub.add_parser("benchmark", help="engine throughput and heuristic baseline strengths")
    p.add_argument("--players", type=int, nargs="+", default=[2])
    p.add_argument("--seeds", type=int, default=200)
    p.set_defaults(fn=cmd_benchmark)

    p = sub.add_parser("train", help="PPO training")
    p.add_argument("--config", required=True)
    p.add_argument("--run-name")
    p.add_argument("--seed", type=int)
    p.add_argument("--init-from")
    p.add_argument("--resume")
    p.set_defaults(fn=cmd_train)

    p = sub.add_parser("evaluate", help="benchmark a checkpoint against the baseline suite")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--players", type=int, nargs="+", default=[2])
    p.add_argument("--seeds", type=int, default=500)
    p.add_argument("--split", choices=["dev", "release"], default="dev")
    p.add_argument("--opponents", nargs="+", default=["random", "greedy", "column", "finish", "finish_tuned_2p"])
    p.add_argument("--sample", action="store_true", help="sample actions instead of arg-max")
    p.add_argument("--out")
    p.set_defaults(fn=cmd_evaluate)

    p = sub.add_parser("play", help="play a match in the terminal against an agent")
    p.add_argument("--checkpoint", help="model checkpoint or heuristic name (default: finish)")
    p.add_argument("--players", type=int, default=2)
    p.add_argument("--seed", type=int)
    p.add_argument("--save", help="write the replay here")
    p.add_argument("--search", action="store_true", help="belief-sampled root search (slower, stronger)")
    p.add_argument("--rollouts", type=int, default=256)
    p.set_defaults(fn=cmd_play)

    p = sub.add_parser("advise", help="get move suggestions at a physical table (enter public events)")
    p.add_argument("--checkpoint", help="model checkpoint or heuristic name (default: finish)")
    p.add_argument("--players", type=int, default=2)
    p.add_argument("--seat", type=int, default=0, help="your seat (0-based, clockwise)")
    p.add_argument("--search", action="store_true", help="belief-sampled root search (slower, stronger)")
    p.add_argument("--rollouts", type=int, default=256)
    p.set_defaults(fn=cmd_advise)

    p = sub.add_parser("export", help="package a checkpoint for release")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--benchmark", help="benchmark JSON to embed")
    p.add_argument("--notes", default="")
    p.set_defaults(fn=cmd_export)

    p = sub.add_parser("replay", help="verify (and print) a replay file")
    p.add_argument("file")
    p.add_argument("-v", "--verbose", action="store_true")
    p.set_defaults(fn=cmd_replay)

    a = ap.parse_args(argv)
    _threads(1)
    try:
        return a.fn(a)
    except (EOFError, KeyboardInterrupt):
        print("\nstopped")
        return 1


if __name__ == "__main__":
    sys.exit(main())
