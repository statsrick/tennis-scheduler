from flask import Flask, render_template, request, send_file
import pandas as pd
from io import BytesIO
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
import csv
import random
from itertools import combinations
from collections import Counter
from typing import List, Dict, Optional, Tuple
import time
import math
import logging

app = Flask(__name__)
app.config['TEMPLATES_AUTO_RELOAD'] = True
logging.basicConfig(level=logging.INFO)

# def optimize_doubles_schedule(
#     N: int,
#     M: int,
#     player_names: Optional[List[str]] = None,
#     targets: Optional[Dict[str, float]] = None,
#     uniform_target: Optional[float] = None,
#     pair_penalty: float = 0.05,
#     seed: Optional[int] = 42,
#     msg: int = 0,
# ):
#     """
#     Solve an assignment of N players into M weeks (4 per week) by ILP.

#     Objective:
#       Minimize sum_p |sum_w x[p,w] - target_p|  +  pair_penalty * sum_{p<q} sum_w y[p,q,w]
#     where:
#       x[p,w] = 1 if player p plays in week w, else 0
#       y[p,q,w] = 1 if players p and q both play in week w (same foursome)

#     Args
#     ----
#     N : number of players (>=4)
#     M : number of weeks (>=1)
#     player_names : list of player names (len N). If None -> ["P1",...,"PN"]
#     targets : dict name->float desired total matches per player (can be fractional)
#     uniform_target : if given, everyone has this same target (ignored if targets provided)
#     pair_penalty : nonnegative; higher values discourage repeated pairings in the same foursome
#     seed : tie-break reproducibility for post-assignment team splits
#     msg : pulp solver verbosity (0 quiet)

#     Returns
#     -------
#     result : dict with keys:
#       - 'weeks': list[{week:int, players:[str], teams:[(str,str),(str,str)]}]
#       - 'counts': dict name->int
#       - 'deviation': dict name->float (final |count - target|)
#       - 'objective_value': float
#       - 'targets': dict name->float
#     """
#     if N < 4:
#         raise ValueError("Need at least 4 players.")
#     if M < 1:
#         raise ValueError("Need at least 1 week.")

#     if player_names is None:
#         player_names = [f"P{i+1}" for i in range(N)]
#     if len(player_names) != N:
#         raise ValueError("player_names must have length N.")

#     players = list(player_names)

#     # Build targets
#     if targets is None:
#         if uniform_target is not None:
#             targets = {p: float(uniform_target) for p in players}
#         else:
#             # Even target by default (may be fractional)
#             avg = 4.0 * M / N
#             targets = {p: avg for p in players}
#     else:
#         # ensure float
#         targets = {p: float(targets.get(p, 0.0)) for p in players}


#     # --- ILP model ---
#     import pulp

#     model = pulp.LpProblem("Tennis_Doubles_Scheduler", pulp.LpMinimize)

#     # Binary play variables x[p,w]
#     x = {(p, w): pulp.LpVariable(f"x_{p}_{w}", cat="Binary")
#         for p in players for w in range(M)}

#     # Pair co-participation variables y[p,q,w] (linearization for "both selected in same week")
#     pairs = [(p, q) for p, q in combinations(players, 2)]
#     y = {(p, q, w): pulp.LpVariable(f"y_{p}_{q}_{w}", cat="Binary")
#         for (p, q) in pairs for w in range(M)}

#     # Absolute deviation variables d_plus, d_minus for each player
#     d_plus  = {p: pulp.LpVariable(f"dplus_{p}", lowBound=0) for p in players}
#     d_minus = {p: pulp.LpVariable(f"dminus_{p}", lowBound=0) for p in players}

#     # Constraints:
#     # 1) Exactly 4 players each week
#     for w in range(M):
#         model += (pulp.lpSum(x[(p, w)] for p in players) == 4, f"four_players_week_{w}")

#     # 2) Deviation definition: sum_w x[p,w] - target_p = d_plus - d_minus
#     for p in players:
#         model += (
#             pulp.lpSum(x[(p, w)] for w in range(M)) - targets[p] == d_plus[p] - d_minus[p],
#             f"abs_dev_balance_{p}"
#         )

#     # 3) Pair linearization: y[p,q,w] = 1 if both x[p,w] and x[q,w] are 1
#     #    Enforce: y <= x[p,w], y <= x[q,w], y >= x[p,w] + x[q,w] - 1
#     for (p, q) in pairs:
#         for w in range(M):
#             model += y[(p, q, w)] <= x[(p, w)]
#             model += y[(p, q, w)] <= x[(q, w)]
#             model += y[(p, q, w)] >= x[(p, w)] + x[(q, w)] - 1

#     # Objective: minimize L1 deviation + pair_penalty * repeats
#     total_abs_dev = pulp.lpSum(d_plus[p] + d_minus[p] for p in players)
#     total_pair_repeats = pulp.lpSum(y[(p, q, w)] for (p, q) in pairs for w in range(M))

#     model += total_abs_dev + pair_penalty * total_pair_repeats

#     # Solve
#     solver = pulp.PULP_CBC_CMD(msg=msg)  # CBC ships with PuLP
#     status = model.solve(solver)
#     if pulp.LpStatus[status] != "Optimal" and pulp.LpStatus[status] != "Feasible":
#         raise RuntimeError(f"Solver status: {pulp.LpStatus[status]} (no solution found).")

#     # Extract solution: which players each week
#     weeks_players: List[List[str]] = []
#     for w in range(M):
#         chosen = [p for p in players if pulp.value(x[(p, w)]) > 0.5]
#         # numerical robustness: enforce size 4 by closest
#         if len(chosen) != 4:
#             # pick top 4 by value if slight fractional remains
#             vals = sorted([(p, pulp.value(x[(p, w)])) for p in players],
#                           key=lambda t: t[1], reverse=True)
#             chosen = [p for p, _ in vals[:4]]
#         weeks_players.append(chosen)

#     # Post-process: form teams for each foursome to reduce repeated pairings
#     if seed is not None:
#         random.seed(seed)

#     # Count historical pairings as we assign teams week by week
#     pair_count = Counter()

#     def best_pairing(four: List[str]) -> List[Tuple[str, str]]:
#         a, b, c, d = four
#         opts = [
#             [(a, b), (c, d)],
#             [(a, c), (b, d)],
#             [(a, d), (b, c)],
#         ]
#         def score(opt):
#             return sum(pair_count[frozenset(t)] for t in opt)
#         best = min(opts, key=score)
#         # tiebreak
#         best_score = score(best)
#         cands = [o for o in opts if score(o) == best_score]
#         return random.choice(cands)

#     schedule = []
#     for w in range(M):
#         four = weeks_players[w]
#         teams = best_pairing(four)
#         # update pair history
#         for t in teams:
#             pair_count[frozenset(t)] += 1
#         schedule.append({
#             "week": w + 1,
#             "players": four,
#             "teams": [(a, b) for (a, b) in teams]
#         })

#     # Stats
#     counts = Counter()
#     for w in range(M):
#         for p in weeks_players[w]:
#             counts[p] += 1

#     deviation = {p: abs(counts[p] - targets[p]) for p in players}
#     obj_val = pulp.value(model.objective)

#     return {
#         "weeks": schedule,
#         "counts": dict(counts),
#         "deviation": deviation,
#         "objective_value": obj_val,
#         "targets": targets,
#     }

# -------------------------
# Core optimizer
# -------------------------
def optimize_doubles_schedule(
    N: int,
    M: int,
    player_names: Optional[List[str]] = None,
    targets: Optional[Dict[str, float]] = None,
    uniform_target: Optional[float] = None,
    pair_penalty: float = 0.0,      # set <=0 to disable pair binaries for speed
    seed: Optional[int] = 42,
    msg: int = 0,
    time_limit: int = 20,           # seconds
    frac_gap: float = 0.01,         # 1% optimality gap
    threads: int = 0,               # 0 = all cores
    soft_bounds: bool = True,
    bounds_slack: int = 1           # +/- slack around targets for pruning
):
    """
    Assign N players into M weeks (exactly 4 per week) by MILP with speedups.
    Returns dict with schedule + stats + solver metadata.
    """
    if N < 4:
        raise ValueError("Need at least 4 players.")
    if M < 1:
        raise ValueError("Need at least 1 week.")

    if player_names is None:
        player_names = [f"P{i+1}" for i in range(N)]
    if len(player_names) != N:
        raise ValueError("player_names must have length N.")

    players = [str(p) for p in player_names]

    # Build targets
    if targets is None:
        if uniform_target is not None:
            targets = {p: float(uniform_target) for p in players}
        else:
            avg = 4.0 * M / N
            targets = {p: avg for p in players}
    else:
        targets = {p: float(targets.get(p, 0.0)) for p in players}

    import pulp

    # Model
    model = pulp.LpProblem("Tennis_Doubles_Scheduler", pulp.LpMinimize)

    # Variables
    x = {(p, w): pulp.LpVariable(f"x_{p}_{w}", cat="Binary")
         for p in players for w in range(M)}

    # Absolute deviation vars
    d_plus  = {p: pulp.LpVariable(f"dplus_{p}", lowBound=0) for p in players}
    d_minus = {p: pulp.LpVariable(f"dminus_{p}", lowBound=0) for p in players}

    # Constraints
    for w in range(M):
        model += (pulp.lpSum(x[(p, w)] for p in players) == 4, f"four_players_week_{w}")

    for p in players:
        model += (
            pulp.lpSum(x[(p, w)] for w in range(M)) - targets[p] == d_plus[p] - d_minus[p],
            f"abs_dev_balance_{p}"
        )

    # Optional pair variables (DISABLED by default, because the term is constant per week)
    use_pairs = bool(pair_penalty and pair_penalty > 0)
    total_pair_repeats = 0
    if use_pairs:
        pairs = [(p, q) for p, q in combinations(players, 2)]
        y = {(p, q, w): pulp.LpVariable(f"y_{p}_{q}_{w}", cat="Binary")
             for (p, q) in pairs for w in range(M)}
        for (p, q) in pairs:
            for w in range(M):
                model += y[(p, q, w)] <= x[(p, w)]
                model += y[(p, q, w)] <= x[(q, w)]
                model += y[(p, q, w)] >= x[(p, w)] + x[(q, w)] - 1
        total_pair_repeats = pulp.lpSum(y[(p, q, w)] for (p, q) in pairs for w in range(M))
    else:
        total_pair_repeats = 0

    # Soft bounds around target totals to prune the search
    if soft_bounds:
        for p in players:
            total_p = pulp.lpSum(x[(p, w)] for w in range(M))
            lo = max(0, math.floor(targets[p]) - bounds_slack)
            hi = min(M, math.ceil(targets[p]) + bounds_slack)
            model += (total_p >= lo, f"lb_{p}")
            model += (total_p <= hi, f"ub_{p}")

    # Objective
    total_abs_dev = pulp.lpSum(d_plus[p] + d_minus[p] for p in players)
    model += total_abs_dev + (pair_penalty * total_pair_repeats)

    # Solve (with time limit, mip gap, threads)
    solver = pulp.PULP_CBC_CMD(
        msg=msg,
        timeLimit=time_limit,
        fracGap=frac_gap,
        threads=threads
    )
    t0 = time.time()
    status = model.solve(solver)
    solve_seconds = time.time() - t0
    solver_status = pulp.LpStatus.get(status, "Unknown")

    # Helper: build teams to reduce repeated pairings across weeks (post-process)
    if seed is not None:
        random.seed(seed)
    pair_count = Counter()

    def best_pairing(four: List[str]) -> List[Tuple[str, str]]:
        a, b, c, d = four
        opts = [[(a, b), (c, d)], [(a, c), (b, d)], [(a, d), (b, c)]]
        def score(opt):
            return sum(pair_count[frozenset(t)] for t in opt)
        best = min(opts, key=score)
        best_score = score(best)
        cands = [o for o in opts if score(o) == best_score]
        return random.choice(cands)

    def build_schedule_from_weeks(weeks_players: List[List[str]]):
        schedule = []
        for w in range(M):
            four = weeks_players[w]
            teams = best_pairing(four)
            for t in teams:
                pair_count[frozenset(t)] += 1
            schedule.append({
                "week": w + 1,
                "players": four,
                "teams": [(a, b) for (a, b) in teams]
            })
        counts = Counter()
        for w in range(M):
            for p in weeks_players[w]:
                counts[p] += 1
        deviation = {p: abs(counts[p] - targets[p]) for p in players}
        obj_val = None
        try:
            obj_val = float(pulp.value(model.objective))
        except Exception:
            pass
        return schedule, dict(counts), deviation, obj_val

    # Extract x solution if feasible/optimal, else greedy fallback
    fallback_used = False
    if solver_status in ("Optimal", "Feasible"):
        weeks_players: List[List[str]] = []
        for w in range(M):
            chosen = [p for p in players if pulp.value(x[(p, w)]) > 0.5]
            if len(chosen) != 4:
                # numerical cleanup: take top 4 by value
                vals = sorted([(p, pulp.value(x[(p, w)])) for p in players],
                              key=lambda t: t[1], reverse=True)
                chosen = [p for p, _ in vals[:4]]
            weeks_players.append(chosen)
        schedule, counts, deviation, obj_val = build_schedule_from_weeks(weeks_players)
    else:
        # Greedy fallback: pick 4 players with largest remaining need (targets - current count)
        fallback_used = True
        counts_g = {p: 0 for p in players}
        weeks_players = []
        for w in range(M):
            need = {p: (targets[p] - counts_g[p]) for p in players}
            # pick 4 with highest need (tie-break random)
            cands = sorted(players, key=lambda p: (need[p], random.random()), reverse=True)[:4]
            weeks_players.append(cands)
            for p in cands:
                counts_g[p] += 1
        schedule, counts, deviation, obj_val = build_schedule_from_weeks(weeks_players)

    return {
        "weeks": schedule,
        "counts": counts,
        "deviation": deviation,
        "objective_value": obj_val if obj_val is not None else float('nan'),
        "targets": targets,
        "solver_status": solver_status,
        "fallback_used": fallback_used,
        "solve_seconds": solve_seconds,
    }

# -------------------------
# Wrapper to generate schedule + stats
# -------------------------
def generate_schedule(N, M, player_names, targets=None, uniform_target=None, pair_penalty=0.0):
    players = list(player_names)
    if targets is None:
        if uniform_target is not None:
            targets = {p: float(uniform_target) for p in players}
        else:
            avg = 4.0 * M / N
            targets = {p: avg for p in players}
    else:
        targets = {p: float(targets.get(p, 0.0)) for p in players}

    app.logger.info("Starting MILP: N=%d, M=%d, targets=%s", N, M, targets)
    t0 = time.time()
    res = optimize_doubles_schedule(
        N, M,
        player_names=players,
        targets=targets,
        uniform_target=None,
        pair_penalty=pair_penalty,     # 0 disables pair binaries (fast)
        seed=23748,
        msg=0,
        time_limit=20,
        frac_gap=0.01,
        threads=0,
        soft_bounds=True,
        bounds_slack=1
    )
    app.logger.info("MILP finished in %.2fs (status=%s, fallback=%s)",
                    time.time() - t0, res["solver_status"], res["fallback_used"])

    return res, res["counts"], res["deviation"], res["objective_value"]

# -------------------------
# File generation helpers
# -------------------------
def generate_csv(res):
    output = BytesIO()
    writer = csv.writer(output, newline="")
    header = ["Player/Week"] + [f"Week {i + 1}" for i in range(len(res['weeks']))]
    writer.writerow(header)
    for i in range(4):  # 4 players per week
        row = [f"Player {i + 1}"]
        for week in res["weeks"]:
            row.append(week["players"][i])
        writer.writerow(row)
    output.seek(0)
    return output

def generate_excel(res):
    df = pd.DataFrame(columns=["Player/Week"] + [f"Week {i + 1}" for i in range(len(res['weeks']))])
    for i in range(4):
        row = [f"Player {i + 1}"] + [week["players"][i] for week in res["weeks"]]
        df.loc[i] = row
    output = BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df.to_excel(writer, index=False)
    output.seek(0)
    return output

def generate_pdf(res):
    output = BytesIO()
    c = canvas.Canvas(output, pagesize=letter)
    text = c.beginText(40, 750)
    text.setFont("Helvetica", 10)
    text.textLines("Schedule:")
    for week in res["weeks"]:
        week_info = f"Week {week['week']}: {', '.join(week['players'])}"
        text.textLine(week_info)
    c.drawText(text)
    c.showPage()
    c.save()
    output.seek(0)
    return output

# -------------------------
# Utilities for robust form parsing
# -------------------------
def parse_player_names(form_val: str) -> List[str]:
    if not form_val:
        return []
    return [p.strip() for p in form_val.split(',') if p.strip()]

def parse_targets_from_form(form, player_names: List[str]) -> Dict[str, float]:
    """Read target by exact (trimmed) player name first; fall back to t_i."""
    targets = {}
    for i, p in enumerate(player_names):
        val = form.get(p)  # preferred: visible input named with trimmed player
        if val is None:
            val = form.get(f"t_{i}")  # fallback by index
        try:
            v = float(val) if val not in (None, "") else 0.0
        except Exception:
            v = 0.0
        targets[p] = v
    return targets

# -------------------------
# Routes
# -------------------------
@app.route("/", methods=["GET", "POST"])
def home():
    if request.method == "GET":
        return render_template("index.html")

    # POST
    raw_names = request.form.get("player_names", "")
    player_names = parse_player_names(raw_names)
    if not player_names:
        return render_template("index.html", error="Please enter player names.")

    M = int(request.form.get("num_weeks", "0") or 0)
    if M < 1:
        return render_template("index.html", error="Number of weeks must be ≥ 1.", player_names=player_names, M=M)

    N = len(player_names)

    # Build targets from form
    targets = parse_targets_from_form(request.form, player_names)
    total = sum(targets.values())
    max_slots = 4 * M
    if total > max_slots + 1e-9:
        return render_template(
            "index.html",
            error=f"Total target ({total:.0f}) exceeds maximum {max_slots} (4 × {M}).",
            player_names=player_names,
            M=M,
            custom_targets=targets
        )

    # Run optimizer (pair_penalty=0 for speed)
    res, counts, deviation, obj_val = generate_schedule(
        N, M, player_names, targets=targets, uniform_target=None, pair_penalty=0.0
    )

    # Prepare table for schedule
    schedule_data = [w["players"] for w in res["weeks"]]

    return render_template(
        "index.html",
        player_names=player_names,
        M=M,
        custom_targets=targets,
        counts=counts,
        deviation=deviation,
        obj_val=obj_val,
        schedule_data=schedule_data
    )

@app.route("/download", methods=["POST"])
def download():
    file_format = request.form.get("file_format")
    raw_names = request.form.get("player_names", "")
    player_names = parse_player_names(raw_names)
    if not player_names:
        return render_template("index.html", error="Please enter player names to download.")

    M = int(request.form.get("num_weeks", "0") or 0)
    if M < 1:
        return render_template("index.html", error="Number of weeks must be ≥ 1.", player_names=player_names, M=M)

    N = len(player_names)
    targets = parse_targets_from_form(request.form, player_names)

    # Guard on total slots
    total = sum(targets.values())
    max_slots = 4 * M
    if total > max_slots + 1e-9:
        return render_template(
            "index.html",
            error=f"Total target ({total:.0f}) exceeds maximum {max_slots} (4 × {M}).",
            player_names=player_names,
            M=M,
            custom_targets=targets
        )

    # Recompute schedule with same settings
    res, counts, deviation, obj_val = generate_schedule(
        N, M, player_names, targets=targets, uniform_target=None, pair_penalty=0.0
    )

    if file_format == "csv":
        file = generate_csv(res)
        return send_file(file, as_attachment=True, download_name="schedule.csv", mimetype="text/csv")
    elif file_format in ("xls", "xlsx"):
        file = generate_excel(res)
        return send_file(file, as_attachment=True,
                         download_name="schedule.xlsx",
                         mimetype="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
    elif file_format == "pdf":
        file = generate_pdf(res)
        return send_file(file, as_attachment=True, download_name="schedule.pdf", mimetype="application/pdf")
    else:
        return render_template("index.html", error="Unknown file format.", player_names=player_names, M=M)

if __name__ == "__main__":
    # Consider disabling reloader to avoid double-exec of the solver in dev
    app.run(debug=True, use_reloader=False)