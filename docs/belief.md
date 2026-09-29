# Card accounting and beliefs

**Finding: the exact public belief is a pair of card counts; no particle filter is needed.**

In Skyjo nobody — including a card's owner — sees an unrevealed grid card, and every card drawn from the deck
becomes public when the drawer resolves it (it is either placed face up or discarded face up). The only
private information a player ever holds is their own pending draw, for the duration of one decision.
Therefore, conditional on everything public, the unseen cards are exchangeable:

- **Before the first recycle of a round** the unseen cards are the hidden grid slots of all players plus the
  draw pile. Their joint distribution is a uniform random assignment of the multiset
  `pool = inventory − visible grid cards − discard pile − (own pending draw)` to those locations.
- **After a recycle** the new draw pile is the old discard pile minus its top card, whose composition is
  public. It is tracked as `drawk` (decremented as cards are drawn). Hidden grid slots are then a uniform
  assignment of `pool = inventory − visible − discard − drawk − pending`.

Behaviour cannot carry information about unseen values because no actor observes them; hence no
opponent-dependent reweighting is required (PLAN.md section 5 anticipated particles "when observations couple
assignments between pools"; for classic Skyjo they never do). Completed columns enter the discard pile and are
counted there, so recycled cards are not double-counted.

Where this is used:

- The raw observation carries `pool`, `drawk` and the discard composition (`obs_layout.O_POOL/O_DRAWK/O_DISCC`).
- `models/features.py` turns them into marginals: hidden-card distribution, next-draw distribution (known
  composition after a recycle, recycled discard pile if the pile is empty), means and spread.
- `search/root_mc.build_world` samples joint worlds (not independent per-slot samples, so decks are never
  impossible).

Tests: `tests/test_properties.py::test_conservation_and_accounting` checks after every step that the public
`pool` equals the true multiset of hidden-slot (+ draw pile, pre-recycle) cards, including long stalling
traces that recycle the deck; `test_observation_invariant_to_hidden_state` permutes all unseen cards and checks
that observations and masks are unchanged; `tests/test_search.py` checks that sampled worlds reproduce the root
observation exactly and that search output is unchanged when the real hidden state is permuted.
