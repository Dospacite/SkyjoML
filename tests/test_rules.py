"""Hand-authored classic_v1_2p rule fixtures driven through the reference engine."""

import numpy as np
import pytest

from skyjo_ml.game import obs_layout as L
from skyjo_ml.game.actions import A_DRAW, A_REVEAL, A_SETUP, A_TAKE, PAIR_INDEX
from skyjo_ml.game.engine import ABSENT, HIDDEN, VISIBLE, IllegalAction, SkyjoGame
from skyjo_ml.game.rules import DECK_SIZE, INVENTORY_COUNTS


def setup_all(g: SkyjoGame, pair=(0, 3)):
    for _ in range(g.n):
        g.step(A_SETUP + PAIR_INDEX[pair])


def test_inventory():
    assert DECK_SIZE == 150
    assert INVENTORY_COUNTS.tolist() == [5, 10, 15] + [10] * 12


def test_deal_and_census():
    g = SkyjoGame(2, 1)
    assert len(g.draw) == 150 - 24 - 1 and len(g.discard) == 1
    assert (g.card_census() == INVENTORY_COUNTS).all()
    assert g.phase == L.PH_SETUP


def test_setup_reveals_are_simultaneous():
    g = SkyjoGame(2, 2)
    g.step(A_SETUP + PAIR_INDEX[0, 1])
    # the first commit must not be visible to the next player
    o = g.observation()
    grid = o[L.O_GRID : L.O_GRID + 2 * 12]
    assert (grid == L.CODE_HIDDEN).all()
    g.step(A_SETUP + PAIR_INDEX[0, 1])
    assert g.phase == L.PH_SELECT
    for p in range(2):
        assert g.vis[p][0] == VISIBLE and g.vis[p][1] == VISIBLE


def test_highest_initial_sum_starts():
    for vals, expect in [((1, 1, 12, 11), 1), ((9, 5, 2, 3), 0)]:
        g = SkyjoGame(2, 3)
        g.val[0][0], g.val[0][1], g.val[1][0], g.val[1][1] = vals
        setup_all(g, (0, 1))
        assert g.cur == expect


def test_take_discard_exchanges_and_reveals():
    g = SkyjoGame(2, 4)
    setup_all(g)
    p = g.cur
    g.discard[-1] = 7
    old = g.val[p][5]
    g.step(A_TAKE + 5)
    assert g.val[p][5] == 7 and g.vis[p][5] == VISIBLE
    assert g.discard[-1] == old
    assert g.cur == 1 - p


def test_draw_then_keep_or_reveal():
    g = SkyjoGame(2, 5)
    setup_all(g)
    p = g.cur
    top_draw = g.draw[-1]
    g.step(A_DRAW)
    assert g.phase == L.PH_RESOLVE and g.pending == top_draw and g.cur == p
    o = g.observation()
    assert o[L.O_PENDING] == top_draw + 2
    with pytest.raises(IllegalAction):
        g.step(A_REVEAL + 0)  # slot 0 was revealed at setup
    g.step(A_REVEAL + 1)
    assert g.discard[-1] == top_draw and g.vis[p][1] == VISIBLE and g.cur == 1 - p


def test_column_clear_order_exchanged_card_first():
    g = SkyjoGame(2, 6)
    setup_all(g)
    p = g.cur
    g.val[p][0:3] = [4, 4, 9]
    g.vis[p][0:3] = [VISIBLE, VISIBLE, VISIBLE]
    g.discard[-1] = 4
    n_disc = len(g.discard)
    g.step(A_TAKE + 2)
    assert all(g.vis[p][s] == ABSENT for s in range(3))
    # discard pile: ..., 9 (exchanged out), 4, 4, 4
    assert g.discard[n_disc - 1 :] == [9, 4, 4, 4]
    g.step(A_DRAW)
    g.step(A_REVEAL + [s for s in range(12) if g.vis[g.cur][s] == HIDDEN][0])
    assert g.cur == p
    m = g.legal_mask()
    assert not m[A_TAKE + 0] and not m[A_TAKE + 2] and m[A_TAKE + 3]


def test_final_turns_and_doubling():
    g = SkyjoGame(2, 7)
    setup_all(g)
    p = g.cur
    for s in range(12):
        g.vis[p][s] = VISIBLE
        g.val[p][s] = 1
    g.vis[p][11] = HIDDEN
    g.val[p][0:3] = [1, 2, 1]
    g.val[p][3:6] = [1, 2, 1]
    g.val[p][6:9] = [1, 2, 1]
    g.val[p][9:12] = [1, 2, 3]
    g.step(A_DRAW)
    g.step(A_REVEAL + 11)  # finishes: score 4*(1+2+1)=... +1+2+3
    assert g.finisher == p and g.final_left == 1
    others = [1 - p]
    for q in others:
        g.val[q] = [0] * 12  # scores 0 -> finisher not strictly lowest
    for _ in range(1):
        g.step(A_DRAW)
        hidden = [s for s in range(12) if g.vis[g.cur][s] == HIDDEN]
        g.step(A_REVEAL + hidden[0])
    assert len(g.round_history) == 1
    rs = g.round_history[0]
    assert rs[p] == 2 * (3 * 4 + 6)
    assert rs[others[0]] == 0
    assert g.cur == p and g.phase == L.PH_SETUP and g.round == 1


def _finish_two_player_round(fin_vals, other_vals):
    g = SkyjoGame(2, 8)
    setup_all(g)
    p = g.cur
    q = 1 - p
    g.val[p] = list(fin_vals)
    for s in range(11):
        g.vis[p][s] = VISIBLE
    g.vis[p][11] = HIDDEN
    g.val[q] = list(other_vals)
    g.step(A_DRAW)
    g.step(A_REVEAL + 11)
    g.step(A_DRAW)
    g.step(A_REVEAL + [s for s in range(12) if g.vis[q][s] == HIDDEN][0])
    rs = g.round_history[0]
    return rs[p], rs[q]


@pytest.mark.parametrize(
    "fin,other,expect_fin",
    [
        ([1, 1, 2] * 4, [2, 2, 3] * 4, 16),  # strictly lowest: no doubling
        ([1, -1, 0] * 4, [1, -1, 0] * 4, 0),  # zero is never doubled
        ([-1, -2, 0] * 4, [-2, -1, -1] * 4, -12),  # negative is never doubled
        ([1, 1, 2] * 4, [1, 1, 2] * 4, 32),  # tie: doubled
        ([1, 2, 3] * 4, [0, 1, 2] * 4, 48),  # higher: doubled
    ],
)
def test_finisher_doubling(fin, other, expect_fin):
    rf, _ = _finish_two_player_round(fin, other)
    assert rf == expect_fin


def test_tie_with_finisher_doubles():
    g = SkyjoGame(2, 9)
    setup_all(g)
    p = g.cur
    q = 1 - p
    vals = [1, 2, 3] * 4
    g.val[p] = list(vals)
    for s in range(11):
        g.vis[p][s] = VISIBLE
    g.val[q] = list(vals)
    g.step(A_DRAW)
    g.step(A_REVEAL + 11)
    g.step(A_DRAW)
    g.step(A_REVEAL + [s for s in range(12) if g.vis[q][s] == HIDDEN][0])
    rs = g.round_history[0]
    assert rs[q] == 24 and rs[p] == 48


def test_final_scoring_clears_columns():
    g = SkyjoGame(2, 10)
    setup_all(g)
    p = g.cur
    q = 1 - p
    g.val[p] = [1, 2, 3] * 4
    for s in range(11):
        g.vis[p][s] = VISIBLE
    g.val[q] = [7, 7, 7, 5, 5, 5, 1, 2, 3, 0, 0, 0]
    for s in range(12):
        g.vis[q][s] = HIDDEN
    g.step(A_DRAW)
    g.step(A_REVEAL + 11)
    new6 = g.discard[-1]
    g.step(A_TAKE + 6)  # q exchanges slot 6 with discard, hidden elsewhere remain
    rs = g.round_history[0]
    assert rs[q] == new6 + 2 + 3  # three columns cleared at scoring


def test_recycle_keeps_top_discard():
    g = SkyjoGame(2, 11)
    setup_all(g)
    top = g.discard[-1]
    g.discard = g.draw + g.discard
    g.draw = []
    n = len(g.discard)
    g.step(A_DRAW)
    assert g.draw_known
    assert g.discard == [top]
    assert len(g.draw) == n - 2  # n-1 recycled, one drawn
    o = g.observation()
    assert o[L.O_DRAW_KNOWN] == 1
    assert (g.card_census() == INVENTORY_COUNTS).all()


def test_draw_masked_when_nothing_to_draw():
    g = SkyjoGame(2, 12)
    setup_all(g)
    g.draw = []
    g.discard = [g.discard[-1]]
    m = g.legal_mask()
    assert not m[A_DRAW] and m[A_TAKE + 0]


def test_empty_grid_triggers_end():
    g = SkyjoGame(2, 13)
    setup_all(g)
    p = g.cur
    for s in range(9):
        g.vis[p][s] = ABSENT
    g.val[p][9:12] = [6, 6, 1]
    g.vis[p][9:12] = [VISIBLE, VISIBLE, VISIBLE]
    g.discard[-1] = 6
    g.step(A_TAKE + 11)
    assert g.finisher == p
    assert all(v == ABSENT for v in g.vis[p])


def test_opponent_revealing_everything_on_final_turn_keeps_finisher():
    g = SkyjoGame(2, 14)
    setup_all(g)
    p = g.cur
    q = 1 - p
    for s in range(11):
        g.vis[p][s] = VISIBLE
    g.step(A_DRAW)
    g.step(A_REVEAL + 11)
    for s in range(12):
        g.vis[q][s] = VISIBLE
    g.vis[q][11] = HIDDEN
    assert g.finisher == p and g.final_left == 1 and g.cur == q
    g.step(A_DRAW)
    g.step(A_REVEAL + 11)  # q also has no hidden cards now, but p remains the finisher
    assert len(g.round_history) == 1 and g.finisher_history == [p]


def test_match_end_and_shared_win():
    g = SkyjoGame(2, 15)
    setup_all(g)
    p = g.cur
    q = 1 - p
    for r in range(2):
        g.val[r] = [1, 2, 3] * 4  # both score 24; the finisher is tied, so doubled to 48
    g.scores[p], g.scores[q] = 71, 95  # 71 + 48 == 95 + 24 == 119: shared win
    for s in range(11):
        g.vis[p][s] = VISIBLE
    g.step(A_DRAW)
    g.step(A_REVEAL + 11)
    g.step(A_DRAW)
    g.step(A_REVEAL + [s for s in range(12) if g.vis[g.cur][s] == HIDDEN][0])
    assert g.over and g.scores == [119, 119]
    assert g.win_credit() == [0.5, 0.5]


def test_invalid_action_fails_loudly():
    g = SkyjoGame(2, 16)
    with pytest.raises(IllegalAction):
        g.step(A_DRAW)
    with pytest.raises(IllegalAction):
        g.step(999)
    assert np.isfinite(g.observation()).all()
