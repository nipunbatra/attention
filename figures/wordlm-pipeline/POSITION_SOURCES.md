# Position teaching revision

The 21 September 2026 revision reads both user-supplied videos through their
English captions. No video frames or transcripts are redistributed. The SVGs
are original illustrations with explicitly separate toy vectors.

The worked opening uses Maya / Ravi / today and our own two-coordinate rows.
It shows the scores and weighted contributions in both word orders before
introducing position. The `today` row `[0.8,0.2]` makes the two person weights
unequal. Adding slot rows changes Maya's key and score, which the next example
calculates explicitly. Both video credits are visible on the topic-break slide.

The diagram-first continuation keeps these same vectors. Each sentence has four
reversible SVG builds, one slot offset per word, before the numeric addition
table. The endpoints and residual results read the live experiment's data.
Adding a position row is a forward-pass operation, not a training update to
the shared word table. The closing full attention map distinguishes the input
addition from the later contextual update and retains the hidden predictor.

The circle controls use illustrative periods 4 and 12. The standard d=4 wave
plots instead use rates 1 and 0.01 radians per slot, whose continuous periods
are about 6.283 and 628.319. Integer token indices do not exactly repeat at 6
or 628. Multiple rates do not guarantee unlimited unique positions or useful
length extrapolation. The absolute-position counterexample holds content at
q=k=[1,0] and adds [cos(i*pi/6),sin(i*pi/6)]. Matches at 3/2 and 8/7 differ,
whereas the corresponding pure rotary matches agree. This contrasts structural
properties; it does not claim absolute methods cannot learn relative cues.

- [Luis Serrano Academy: positional encoding](https://www.youtube.com/watch?v=IHu3QehUmrQ),
  especially 1:26 (geometric displacement), 3:29 (sine/cosine coordinates) and
  4:47 (several rates). The lecture distinguishes repeated toy clocks from the
  original frequency schedule and avoids implying a guarantee of recovery.
- [Jia-Bin Huang: RoPE](https://www.youtube.com/watch?v=SMBkImDWOyQ), especially
  2:58 (rotating a matching vector), 4:02 (shared shifts), 5:30 (coordinate
  pairs) and 8:06 (context extension). The live common-shift experiment holds
  content vectors fixed; it does not claim an entire network is shift invariant.

Primary mathematical references:

- [Attention Is All You Need, §3.5](https://arxiv.org/html/1706.03762v7#S3.SS5)
- [RoFormer, rotary-position derivation](https://arxiv.org/html/2104.09864v5)
- [Position Interpolation](https://arxiv.org/abs/2306.15595)

Queries/keys use the local column-vector convention only for the 2D rotation
identity. The projection calculations elsewhere retain row-vector notation.
The main bank worksheet and notebook still use additive positional rows.
