"""
Group C — rendered-physics trace (the hidden-generator anchor).

A ball moves in a unit box at constant speed, reflecting off walls. The TRUE state
is continuous (position + velocity); the observable SURFACE is a coarse GxG grid
with the ball's quantized cell marked. Velocity is NOT in the surface, and
quantization loses sub-cell info -> the next frame is only partially recoverable
from the rendered stream within a bounded window. This is the 'structure exists but
the productive engine is upstream and only partly expressed' case.

Serialization (like CA): each frame = G rows of G cells, ROWSEP between rows,
FRAMESEP between frames. Tokens: 0 empty, 1 ball, 2 ROWSEP, 3 FRAMESEP.

Oracle: the simulator itself (true next frame, deterministic from continuous state).
Validator (productivity): is a frame-sequence consistent with constant-velocity +
wall-reflection motion within +/-1 cell? Used to score generated 'worlds'.
"""
import random
from typing import List, Tuple

EMPTY, BALL, ROWSEP, FRAMESEP = 0, 1, 2, 3
VOCAB = 4


def simulate(G=12, n_frames=40, speed=0.07, seed=0, x0=None, y0=None, ang=None):
    """Return list of (cx,cy) quantized ball cells, plus continuous trace."""
    rng = random.Random(seed)
    import math
    x = rng.random() if x0 is None else x0
    y = rng.random() if y0 is None else y0
    a = rng.uniform(0, 2 * math.pi) if ang is None else ang
    vx, vy = speed * math.cos(a), speed * math.sin(a)
    cells = []
    for _ in range(n_frames):
        cx, cy = min(int(x * G), G - 1), min(int(y * G), G - 1)
        cells.append((cx, cy))
        x += vx; y += vy
        if x < 0: x = -x; vx = -vx
        elif x >= 1: x = 2 - x; vx = -vx
        if y < 0: y = -y; vy = -vy
        elif y >= 1: y = 2 - y; vy = -vy
    return cells


def render_tokens(cells, G):
    """Flatten frames to a token stream: rows of cells + ROWSEP, FRAMESEP between frames."""
    toks = []
    for (cx, cy) in cells:
        for r in range(G):
            for c in range(G):
                toks.append(BALL if (c == cx and r == cy) else EMPTY)
            toks.append(ROWSEP)
        toks.append(FRAMESEP)
    return toks


def generate_split(n_traj, G=12, n_frames=40, speed=0.07, seed0=0):
    seqs = []
    for i in range(n_traj):
        cells = simulate(G, n_frames, speed, seed=seed0 + i)
        seqs.append(render_tokens(cells, G))
    return seqs


def stream(n_traj=2000, G=12, n_frames=40, speed=0.07, seed0=0):
    return generate_split(n_traj, G, n_frames, speed, seed0), VOCAB


# ---- decode + validator (productivity) ----
def decode_frames(tokens, G):
    """Recover ball cell per frame from a token stream; None if frame malformed/empty."""
    frames, cur, row, ncols, nrows = [], None, [], 0, 0
    cells = []
    # split on FRAMESEP
    frame_toks, buf = [], []
    for t in tokens:
        if t == FRAMESEP:
            frame_toks.append(buf); buf = []
        else:
            buf.append(t)
    for ft in frame_toks:
        # split rows on ROWSEP, find BALL cell
        r = 0; pos = None; col = 0
        for t in ft:
            if t == ROWSEP:
                r += 1; col = 0
            else:
                if t == BALL:
                    pos = (col, r)
                col += 1
        cells.append(pos)
    return cells


def is_valid_physics(tokens, G, min_frames=6):
    """
    Valid bouncing-ball motion (speed < 1 cell/frame):
      - adjacency: |dcx|<=1 and |dcy|<=1 each frame (no teleporting),
      - moving: visits a range of cells (not stuck),
      - reflections: per-axis direction reversals happen only near a wall.
    Real traces pass; random traces fail (they teleport).
    """
    cells = [c for c in decode_frames(tokens, G) if c is not None]
    if len(cells) < min_frames:
        return False, 'too_short_or_missing_ball'
    dxs = [cells[i + 1][0] - cells[i][0] for i in range(len(cells) - 1)]
    dys = [cells[i + 1][1] - cells[i][1] for i in range(len(cells) - 1)]
    adj = sum(1 for a, b in zip(dxs, dys) if abs(a) <= 1 and abs(b) <= 1) / len(dxs)
    moving = len(set(cells)) > max(3, len(cells) // 4)

    def refl_ok(ds, coords):
        bad = tot = 0
        prev = None
        for i, d in enumerate(ds):
            if d == 0:
                continue
            s = 1 if d > 0 else -1
            if prev is not None and s != prev:
                tot += 1
                c = coords[i]
                if not (c <= 1 or c >= G - 2):
                    bad += 1
            prev = s
        return 1.0 if tot == 0 else 1 - bad / tot

    xs = [c[0] for c in cells]; ys = [c[1] for c in cells]
    rx, ry = refl_ok(dxs, xs), refl_ok(dys, ys)
    valid = adj >= 0.95 and moving and rx >= 0.7 and ry >= 0.7
    return valid, f'adj={adj:.2f} moving={moving} reflx={rx:.2f} refly={ry:.2f}'
