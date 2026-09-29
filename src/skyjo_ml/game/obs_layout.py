"""Layout of the raw, seat-relative observation vector shared by every engine and agent.

All entries are publicly reconstructible from the actor's point of view (plus the actor's own pending draw
in the resolve phase). Seats are listed relative to the actor: index 0 is the actor, 1 the next player
clockwise, and so on. Unused seats are padded with ``CODE_NOPLAYER``.
"""

from __future__ import annotations

from .rules import MAX_PLAYERS, N_SLOTS, N_VALUES

# Slot codes
CODE_HIDDEN = N_VALUES  # 15
CODE_ABSENT = N_VALUES + 1  # 16
CODE_NOPLAYER = N_VALUES + 2  # 17
N_CODES = N_VALUES + 3

# Phases (engine and observation)
PH_SETUP = 0
PH_SELECT = 1
PH_RESOLVE = 2
PH_OVER = 3

O_NPLAYERS = 0
O_PHASE = 1
O_PENDING = 2  # value code of the actor's pending draw, -1 otherwise
O_TOP = 3  # top discard code, -1 if the discard pile is empty
O_DRAW_N = 4
O_DISC_N = 5
O_DRAW_KNOWN = 6  # 1 if the draw pile was rebuilt from the (public) discard pile this round
O_FINISHER = 7  # relative seat of the finisher, -1 if the end sequence has not started
O_FINAL_LEFT = 8  # final turns not yet completed, including the current one (0 if not in end sequence)
O_ROUND = 9  # 0-based round index
O_ROUND_TURNS = 10  # decisions taken in the current round
O_SCORES = 11  # MAX_PLAYERS cumulative totals, relative seats
O_GRID = O_SCORES + MAX_PLAYERS  # MAX_PLAYERS * N_SLOTS slot codes
O_POOL = O_GRID + MAX_PLAYERS * N_SLOTS  # counts of cards whose location is not public, per value
O_DRAWK = O_POOL + N_VALUES  # known draw-pile composition (only when O_DRAW_KNOWN)
O_DISCC = O_DRAWK + N_VALUES  # discard-pile composition
OBS_SIZE = O_DISCC + N_VALUES
