# Position teaching revision

The 21 September 2026 revision reads both user-supplied videos through their
English captions. No video frames or transcripts are redistributed. The SVGs
are original illustrations with explicitly separate toy vectors.

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
