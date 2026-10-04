# Epilogue accumulator shuffle (`AccumShuffle`)

How `VectorWidthA > 1` is made legal on `ds_read_tr` kernels, and what to watch for when
touching it.

Code: `accumShuffleForLDSTrVW` in `Tensile/KernelWriterAssembly.py`, called from
`endSummation`. Preconditions and the parameter's side effects are in
`Tensile/SolutionStructs/Solution.py`. Test config:
`Tensile/Tests/common/sparse/gfx1250/accshuf_validate.yaml`.

---

## 1. The problem

Two parts of the kernel disagree about which M rows a lane owns.

**The local read is fixed at VW=1 in the tile direction.** `ds_read_tr` hands lane `L` the tile
row `m_base + L`. A transpose group reads `NUM_CONT_READ_ELEMENTS` *contiguous* rows and every
lane in the group shares one `m_base`, so the lane→row stride is locked at 1. Widening it is not
a tuning choice; the instruction does not offer it. See `mTileOffset` in
`Tensile/Components/LocalRead.py`.

**The store wants VW consecutive rows per lane.** It takes `vectorWidth0 = VectorWidthA` and
`coord0 = VectorWidthA * (wave_id0 + tid0)` (`Tensile/Components/NotLocalFullTileElements.py`),
grouping VW consecutive wave tiles into one wide store.

With both layouts live and nothing bridging them, the kernel is simply wrong. Concretely, a
`VectorWidthA=2` LDSTr kernel gave each lane A rows `(L%16)` and `(L%16)+1` while the store wrote
rows `2*(L%16)` and `2*(L%16)+1`: M rows 17..31 of each 32-row tile were never read, and only
lane 0 landed correctly.

Historically this was avoided by clamping: `enableLDSTrA` forced `VectorWidthA = 1`.

## 2. The idea

Keep the read at its natural VW=1 geometry, keep the store at VW=2/4/8, and bridge the two
**once, in the epilogue**, after the MFMAs have finished and before any `ValuC` consumer runs.

With `MI = MatrixInstM`, `l = L % MI`, `T = MIWaveTile[0]` tiles and `VW = VectorWidthA`
(`T % VW == 0`):

```
after MFMA    read tile t,  lane l  ->  M row MI*t + l
after shuffle store tile t', lane l ->  M row MI*VW*(t'//VW) + VW*l + (t'%VW)
```

Both index the same `MI*T` rows. Per group of VW tiles this is exactly a **VW × MI transpose**
spread over (VW registers) × (MI lanes). Groups are independent: group `g` covers M rows
`[MI*VW*g, MI*VW*(g+1))`, which is exactly read tiles `VW*g .. VW*g+VW-1`.

## 3. How it works

For destination slot `s` in a group, the source lane and source register are

```
lane = ((L << log2VW) & (MI-1)) | s      -- one address, reused for all VW sources
reg  = (L % MI) >> (log2MI - log2VW)     -- bits [log2MI-1 : log2MI-log2VW] of L
```

The lane part is a cross-lane move, so it is a `ds_bpermute`. The register part depends on the
**destination** lane, not the source, so it cannot be folded into the address — it has to happen
*after* the gather. Hence, per destination slot:

1. **VW `ds_bpermute`**, one per candidate source register, all sharing the same address vgpr
   (slot `s` only changes the instruction's immediate `offset:4*s`).
2. **A `log2(VW)`-deep `cndmask` tree** that selects among those VW candidates using masks
   precomputed from `Serial`. Mask bit `b` tests bit `(log2MI - log2VW + b)` of `Serial`.

Results are staged in `res[]` and committed to `ValuC` **once per group**. This is not an
optimisation: every destination tile of a group reads *all* VW source tiles, so writing in place
corrupts the group. It happens to be silently harmless at VW=2 and 4 and wrong at VW=8 (1536 of
4096 values), which is exactly the kind of bug that survives a narrow test matrix.

### 3.1 Cost

Per group of VW tiles: `VW*VW` `ds_bpermute` + `VW*(VW-1)` `cndmask` — quadratic in VW.
But `numGroups = MIWaveTile[0] / VW`, so the **total** is

```
numNTiles * MIWaveTile[0] * accPerTile * VW      -- linear in VW
```

Against this it saves store instructions (32 b32 → 16 b64 at VW=2, MT0=32). The b32 stores it
replaces are already fully coalesced, so this trades instruction count only, never bandwidth.

### 3.2 Pass structure and batching

The walk is `numNTiles * numGroups * accPerTile` passes, each covering `slotBatch` slots. A
**slot** is a coordinate — one of the VW destination rows in a group, the `s` in
`M row = MI*VW*g + VW*l + s`. A **pass** is a unit of emission: issue the gathers, wait,
consume. Passes come from chopping a group's VW slots into chunks of `slotBatch`, so each
`(n, g, i)` yields `ceil(VW / slotBatch)` of them, and each pass issues `VW * len(slots)`
`ds_bpermute` — VW candidates per slot.

Batching exists to overlap LDS round trips: doing one slot at a time serialises VW of them,
each fully exposed. The size is bounded by the temp vgprs it needs — `VW` for `res` plus
`buffers * VW * slotBatch` for the gather buffer(s), where `buffers` is 2 under
`AccumShuffle=2`:

```python
TMP_BUDGET = 24
buffers   = 2 if pipelined else 1
want      = min(VW, max(2, (TMP_BUDGET - VW) // VW))
free      = self.vgprPool.available()
fits      = (free - VW) // (buffers * VW) if free > VW else 0
slotBatch = max(1, min(want, fits))
```

| VW | slotBatch | `res` | `gat` | `gat` + `gatB` (AccumShuffle=2) | bpermute per pass |
|----|-----------|-------|-------|--------------------------------|-------------------|
| 2  | 2         | 2     | 4     | 8                              | 4                 |
| 4  | 4         | 4     | 16    | 32                             | 16                |
| 8  | 2         | 8     | 16    | 32                             | 16                |
| 16 | 2         | 16    | 32    | 64                             | 32                |

**The passes are mutually independent.** Distinct `(n, g, i)` touch disjoint accumulators, and
the chunks of one `(n, g, i)` share only read-only source registers. That independence is what
`AccumShuffle=2` exploits.

### 3.3 Why 24, and why a separate floor

The three pieces of that expression answer different questions, and only one of them is a
policy choice.

**`TMP_BUDGET = 24` is derived, not guessed.** It is the smallest value that keeps every
supported VW off `slotBatch == 1`: VW=4 reaches 4 at 20, VW=8 reaches 2 at 24. Measured on
gfx1250, `slotBatch == 1` is the one setting that is clearly bad — at VW=4 it costs 14%
(`AccumShuffle=1`) to 28% (`AccumShuffle=2`) against the 3 that the old budget of 16 produced
there, and VW=8 orders the same way.
Above that the curve flattens: 4 vs 8 at VW=8 measured -0.2% and -2.9%, both inside this
benchmark's run-to-run noise, so a bigger batch buys nothing.

**The `max(2, ...)` floor is separate from the budget on purpose.** A pure budget has to grow
like `3*VW` to hold `slotBatch >= 2`, so no single constant serves every VW: raising it enough
for VW=16 (48) would drag VW=8 up to 5, paying registers for a batch size already measured to
be worthless. The explicit floor scales to new VW values without touching the budget.

**The `fits` clamp is a safety bound, not a policy.** `checkOut()` past what the pool has free
*grows* the pool and raises the kernel's high-water mark. This code runs in `endSummation`,
before `globalWriteElements` sizes its store batch, so a higher mark here can cost occupancy for
the rest of the epilogue. Note it caps the batch but does not promise the pool never grows:
`slotBatch 1` still needs `VW*(1 + buffers)` registers and the shuffle is not optional, so if
the pool were tighter than that, growing would be the only correct outcome. In practice there is
no contest — the main loop's temporaries are all dead by `endSummation`, leaving ~474 free on a
1020-vgpr MT128x256x256 kernel.

One caveat on the measurements behind all this. Benchmarking variants as fork values inside a
*single* Tensile invocation is unsafe here: solution order correlates with GPU warm-up, and we
measured `r = -0.91` between benchmark position and time, a 26% drift from first solution to
last. Budget values enumerate in ascending order, so "bigger budget is faster" appeared as a
clean 35% effect that was entirely an artifact. The numbers above come from one invocation per
variant, position-matched, with byte-identical-code pairs included as a built-in noise probe.
Anything under ~10% from this harness should be treated as unresolved.

## 4. End to end: MI output → shuffle → `buffer_store`

### 4.1 Where M and N live

Under `SourceSwap` the two output dimensions are carried by different things, and this is the
single most useful fact for reading the code:

```
          M  ──►  the LANE index          l = Serial % MatrixInstM
          N  ──►  the ACCUMULATOR regs    accPerTile values per lane,
                  plus  Serial / MatrixInstM
```

From `ComputeStoreVgprsMFMASwap` (`Tensile/Components/ComputeStoreVgprs.py`):

```
coord0 (M) = VectorWidthA * ( (Serial % MI) + wave_id0 * MIBShape0 ) + MT0*WG0
                              ^^^^^^^^^^^^^
                              lane position -> M row

coord1 (N) = VectorWidthB * ( (Serial / MI) * MIOutputVectorWidth
                              + wave_id1 * MIBShape1 ) + MT1*WG1
```

**The shuffle only ever moves data along M.** It permutes which lane holds which row, for a
fixed `(n, i)`. N is never touched — `accIdx(n, mTile, i)` only varies `mTile` within a group.

### 4.2 The three stages

Worked example: `MatrixInstruction 16x16x64`, wave32, f16 in and out, `VectorWidthA = 4`,
`MIWaveTile = [4, 2]` — so `numGroups = 4/4 = 1` and this wave owns

```
MatrixInstM * MIWaveTile[0] = 16 * 4 = 64 M rows
```

The K of the MI shape plays no part in any of this. The output tile is `MatrixInstM x
MatrixInstN` regardless of how deep the instruction reduces, so

```
accPerTile = MatrixInstM * MatrixInstN / WavefrontSize = 16*16/32 = 8
```

— the same 8 accumulators per lane whether the instruction is 16x16x64 (f16) or 16x16x128 (f8).
K only decides how many MFMAs feed those accumulators, not where the results land, so everything
below holds unchanged for the f8 shapes.

The tables show **which lane holds each (M, N) element**, over the 64 M rows this wave owns and
the 16 N columns of one n-tile. `MIWaveTile[1] = 2`, so there is a second, identically laid out
block of 16 N columns — the shuffle runs independently per n-tile.

#### Stage 1 — MI output

`ds_read_tr` fixed the lane→row stride at 1, so each tile is a flat block of `MI` rows and lane
`l` owns exactly one row of it: `M row = MI*t + l`. Tile 0 shown; tiles 1/2/3 repeat the same
lanes at M+16, M+32, M+48.

| M rows | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 | 13 | 14 | 15 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **0** | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 16 | 16 | 16 | 16 | 16 | 16 | 16 | 16 |
| **1** | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 17 | 17 | 17 | 17 | 17 | 17 | 17 | 17 |
| **2** | 2 | 2 | 2 | 2 | 2 | 2 | 2 | 2 | 18 | 18 | 18 | 18 | 18 | 18 | 18 | 18 |
| **3** | 3 | 3 | 3 | 3 | 3 | 3 | 3 | 3 | 19 | 19 | 19 | 19 | 19 | 19 | 19 | 19 |
| ⋮ | ⋮ | ⋮ | ⋮ | ⋮ | ⋮ | ⋮ | ⋮ | ⋮ | ⋮ | ⋮ | ⋮ | ⋮ | ⋮ | ⋮ | ⋮ | ⋮ |
| **15** | 15 | 15 | 15 | 15 | 15 | 15 | 15 | 15 | 31 | 31 | 31 | 31 | 31 | 31 | 31 | 31 |

A lane's four rows are **strided by 16** — scattered one per tile. Lane 0 holds M rows 0, 16,
32 and 48.

#### Stage 2 — shuffle

`accumShuffleForLDSTrVW` turns that into `M row = MI*VW*g + VW*l + s`:

| M rows | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 | 13 | 14 | 15 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **0-3** | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 16 | 16 | 16 | 16 | 16 | 16 | 16 | 16 |
| **4-7** | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 17 | 17 | 17 | 17 | 17 | 17 | 17 | 17 |
| **8-11** | 2 | 2 | 2 | 2 | 2 | 2 | 2 | 2 | 18 | 18 | 18 | 18 | 18 | 18 | 18 | 18 |
| **12-15** | 3 | 3 | 3 | 3 | 3 | 3 | 3 | 3 | 19 | 19 | 19 | 19 | 19 | 19 | 19 | 19 |
| **16-19** | 4 | 4 | 4 | 4 | 4 | 4 | 4 | 4 | 20 | 20 | 20 | 20 | 20 | 20 | 20 | 20 |
| **20-23** | 5 | 5 | 5 | 5 | 5 | 5 | 5 | 5 | 21 | 21 | 21 | 21 | 21 | 21 | 21 | 21 |
| **24-27** | 6 | 6 | 6 | 6 | 6 | 6 | 6 | 6 | 22 | 22 | 22 | 22 | 22 | 22 | 22 | 22 |
| **28-31** | 7 | 7 | 7 | 7 | 7 | 7 | 7 | 7 | 23 | 23 | 23 | 23 | 23 | 23 | 23 | 23 |
| **32-35** | 8 | 8 | 8 | 8 | 8 | 8 | 8 | 8 | 24 | 24 | 24 | 24 | 24 | 24 | 24 | 24 |
| **36-39** | 9 | 9 | 9 | 9 | 9 | 9 | 9 | 9 | 25 | 25 | 25 | 25 | 25 | 25 | 25 | 25 |
| **40-43** | 10 | 10 | 10 | 10 | 10 | 10 | 10 | 10 | 26 | 26 | 26 | 26 | 26 | 26 | 26 | 26 |
| **44-47** | 11 | 11 | 11 | 11 | 11 | 11 | 11 | 11 | 27 | 27 | 27 | 27 | 27 | 27 | 27 | 27 |
| **48-51** | 12 | 12 | 12 | 12 | 12 | 12 | 12 | 12 | 28 | 28 | 28 | 28 | 28 | 28 | 28 | 28 |
| **52-55** | 13 | 13 | 13 | 13 | 13 | 13 | 13 | 13 | 29 | 29 | 29 | 29 | 29 | 29 | 29 | 29 |
| **56-59** | 14 | 14 | 14 | 14 | 14 | 14 | 14 | 14 | 30 | 30 | 30 | 30 | 30 | 30 | 30 | 30 |
| **60-63** | 15 | 15 | 15 | 15 | 15 | 15 | 15 | 15 | 31 | 31 | 31 | 31 | 31 | 31 | 31 | 31 |

Now a lane's four rows are **contiguous**, and the whole 64-row span is covered by 16 lanes.
Note the left half (N 0-7) and right half (N 8-15) are each a solid block — **the shuffle moves
M only, N is never touched**. That is the property to check first when reading generated code.

How one element gets there — lane 1, slot 0 wants M row `4*1+0 = 4`, which lives in tile 0,
lane 4:

```
lane = ((1 << 2) & 15) | 0 = 4        -> ds_bpermute address
reg  = 1 >> (log2 16 - log2 4) = 0    -> cndmask picks tile 0
```

Lane 5, slot 0 wants M row 20, which lives in tile 1, lane 4:

```
lane = ((5 << 2) & 15) | 0 = 4        -> the SAME address
reg  = 5 >> 2 = 1                     -> cndmask picks tile 1
```

One `ds_bpermute` address serves all VW candidate registers; only the cndmask tree differs per
lane. That is why the gather comes first and the register select second — and why the VW
candidates have to come from VW *different tiles*: inside a single tile a lane holds only one M
row, so there would be nothing to choose between.

#### Stage 3 — `buffer_store`

`coord0 = VWA * (wave_id0 + tid0)`, so lane `l` addresses M row `VW*l` and its VW rows are
contiguous in D — one wide store each:

```
lane  0: D[M rows  0.. 3]  ─┐
lane  1: D[M rows  4.. 7]   │  buffer_store_b64 (4 x f16)
lane  2: D[M rows  8..11]   │  instead of 4 x buffer_store_b16
  ...                       │
lane 15: D[M rows 60..63]  ─┘
```

Before the shuffle this had to be VW separate narrow stores, because lane `l` held rows
`l, 16+l, 32+l, 48+l` — not adjacent in D.

### 4.3 What each stage costs

| Stage | Per group of VW tiles | Direction touched |
|-------|-----------------------|-------------------|
| MI output | — (already in `ValuC`) | — |
| shuffle | `VW*VW` `ds_bpermute` + `VW*(VW-1)` `cndmask` + `VW` `v_mov` commits | M only |
| store | `MIWaveTile[0]/VW` wide stores instead of `MIWaveTile[0]` narrow ones | M only |

The trade is instruction count, not bandwidth: the narrow stores it replaces were already fully
coalesced. See §6.5 for the counter-intuitive consequence — a *narrow* `VectorWidthA` produces
*more* store instructions, which is why VW=2 is the worst case for epilogue code size.

### 4.4 Why not shuffle inside a single tile?

A natural first instinct is to permute *within* each 16-row MI tile rather than across the VW
tiles of a group. That layout looks like this (tile 0 shown; it repeats every 16 M rows):

| M rows | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 | 13 | 14 | 15 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **0-3** | 0 | 4 | 8 | 12 | 0 | 4 | 8 | 12 | 16 | 20 | 24 | 28 | 16 | 20 | 24 | 28 |
| **4-7** | 1 | 5 | 9 | 13 | 1 | 5 | 9 | 13 | 17 | 21 | 25 | 29 | 17 | 21 | 25 | 29 |
| **8-11** | 2 | 6 | 10 | 14 | 2 | 6 | 10 | 14 | 18 | 22 | 26 | 30 | 18 | 22 | 26 | 30 |
| **12-15** | 3 | 7 | 11 | 15 | 3 | 7 | 11 | 15 | 19 | 23 | 27 | 31 | 19 | 23 | 27 | 31 |

Compare it with the Stage 2 table: **the lane changes as you move along N.** Column 0 is lane 0,
column 1 is lane 4, column 2 is lane 8. In the accepted layout each horizontal half is a solid
block. That difference is the whole story, and it rules the intra-tile variant out three times
over:

**1. It does not match the store's M addressing.** Under `SourceSwap`,
`NotLocalFullTileElements` computes

```
totalTT0     = MatrixInstBM * MIWaveTile[0]   = 4
vectorWidth0 = VectorWidthA                   = 4
tt0 range    = ceil(4 / 4)                    = 1      <- ONE M store group
```

With a single group there is no second index to say "which tile" — M comes entirely from
`coord0 = VWA * (Serial % MI)`, which hands lane `l` rows `4l..4l+3` across the *whole* 64-row
wave span. Intra-tile would give lane 0 rows 0-3 in every tile, and the store would write all
four of them to M 0-3, overwriting tile 0 three times.

**2. It forces N to move as well.** Inside one tile a lane holds exactly *one* M row spread
across its 8 accumulators. To collect VW consecutive rows it must take data from VW lanes —
4 rows x 8 N = 32 values — but it only has 8 registers for that tile. So it has to shed three
quarters of its N columns to other lanes. Intra-tile is necessarily a 2D redistribution.
Cross-tile is 1D: the VW candidates come from VW *different tiles*, i.e. from the register
dimension, so the register count per lane is conserved (4 tiles x 8 N in, 4 rows x 8 N out) and
N never moves.

**3. The resulting N set is strided.** Lane 0 would end up holding N columns {0, 4}, not a
contiguous run. `MIOutputVectorWidth = 8` and `vectorWidth1 = VectorWidthB * MIOutputVectorWidth`
both encode "each lane owns 8 *contiguous* N", so `coord1` and the element enumeration would
have to change too.

In short, the cross-tile layout is not a design choice — it is the only layout that satisfies
the store's existing address computation, and it happens to be the cheaper one because N stays
put.

## 5. The `AccumShuffle` parameter

One knob, three values. It is both the feature switch and the schedule selector.

| Value | Meaning |
|-------|---------|
| **0** (default) | Off. `enableLDSTrA` clamps `VectorWidthA` to 1, exactly as before this feature existed. No VW>1 means nothing to bridge, so no shuffle is emitted. |
| **1** | On. Drains LDS (`s_wait_dscnt 0`) before consuming each pass, so the full `ds_bpermute` latency is exposed once per pass. |
| **2** | On, software-pipelined. Issues the next pass's gather *before* consuming the current one and waits on a staggered `dscnt` that leaves exactly that gather outstanding, so the LDS latency hides behind the previous pass's cndmask tree. |

1 and 2 emit the **same instruction multiset** — only the order and the wait operands differ.
2 additionally needs a second gather buffer (`VW*slotBatch` vgprs).

If it helps to anchor it: `AccumShuffle=2` is structurally the same trick as `PrefetchLocalRead`,
moved into the epilogue — double-buffer the destination, issue one iteration ahead, consume the
previous one while the new one is in flight. The analogy is worth holding loosely in one
respect, which is why this document says "issue-ahead" rather than "prefetch": every other
`Prefetch*` parameter in TensileLite moves data through the memory hierarchy, and this one moves
none. `ds_bpermute` is a lane crossbar, not an LDS access — it reads no LDS storage and consumes
no bandwidth. What is being hidden is queue latency, nothing more.

Three hazards are in play here and it is easy to credit the wrong mechanism for each:

| Hazard | Handled by |
|--------|------------|
| RAW — cndmask reads a gather result before it lands | `s_wait_dscnt` |
| WAR — the next pass's gather overwrites a buffer still being read | **the second buffer**, not the wait |
| WAR — a commit writes `ValuC` while a bpermute still sourcing it is in flight | `s_wait_alu depctr_vm_vsrc` (§6.3) |

The middle row is the reason 2 costs registers at all. No `dscnt` value can prevent the
overwrite: with a single buffer you would either have to wait for the new gather to land
(which is exactly `AccumShuffle=1`, no overlap left) or let it clobber data the cndmask tree
has not read yet. Pipelining here is bought with space, not with waiting.

### 5.1 What 1 and 2 actually emit

Both walk the same `passes` list. The difference is one level of software pipelining:

```python
# AccumShuffle = 1 — drain per pass
for p in passes:
    issueGather(gat, p)                 # VW * len(p.slots) ds_bpermute
    SWaitCnt(dscnt=0)                   # <-- full drain, nothing left in flight
    consume(gat, p)                     # cndmask tree (+ commit on the group's last chunk)

# AccumShuffle = 2 — issue-ahead, staggered wait
issueGather(bufs[0], passes[0])         # prime the pipeline
for idx, p in enumerate(passes):
    if idx + 1 < len(passes):
        nxt = passes[idx + 1]
        issueGather(bufs[(idx+1) % 2], nxt)   # <-- next pass's gather goes out FIRST
        SWaitCnt(dscnt=VW * len(nxt.slots))   # <-- leave exactly nxt's gathers outstanding
    else:
        SWaitCnt(dscnt=0)                     # only the very last pass drains
    consume(bufs[idx % 2], p)
```

Two buffers alternate (`bufs[idx % 2]`) so pass `p+1`'s gather can be in flight while pass `p`'s
cndmask tree is still reading pass `p`'s results — that is the entire reason `gatB` exists.

**The one thing to look for in the generated assembly is the floor of the wait.** At VW=4,
`slotBatch=4`, so a group is a single 16-bpermute chunk. The four `offset:` values are the four
destination slots; the four `ValuC+` operands within each are the four candidate source tiles:

```
AccumShuffle = 1                            AccumShuffle = 2
─────────────────────────────────────       ─────────────────────────────────────
ds_bpermute v518, v512, ValuC+0             ds_bpermute v518, v512, ValuC+0
ds_bpermute v519, v512, ValuC+8             ds_bpermute v519, v512, ValuC+8
ds_bpermute v520, v512, ValuC+16            ds_bpermute v520, v512, ValuC+16
ds_bpermute v521, v512, ValuC+24            ds_bpermute v521, v512, ValuC+24
   ... 16 for this pass ...                    ... 16 for this pass ...
ds_bpermute v533, v512, ValuC+24            ds_bpermute v533, v512, ValuC+24
                                            ds_bpermute v534, v512, ValuC+1    <-- next
                                            ds_bpermute v535, v512, ValuC+9        pass
                                               ... 16 more, into gatB ...          issued
                                            ds_bpermute v549, v512, ValuC+25       ahead
s_wait_dscnt 14                             s_wait_dscnt 30
v_cndmask_b32 v518, v519, v518, s52         v_cndmask_b32 v518, v519, v518, s52
s_wait_dscnt 12                             s_wait_dscnt 28
v_cndmask_b32 v519, v521, v520, s52         v_cndmask_b32 v519, v521, v520, s52
   ...                                         ...
s_wait_dscnt 0   <-- reaches ZERO           s_wait_dscnt 16  <-- floors at 16, never 0
```

Note the descending staircase in *both* columns. It is **not** the `SWaitCnt` this function
emits being split — that one survives verbatim. StinkyTofu strips nothing here and instead
*adds* its own per-consumer waits: `StinkyWaitCntInsertionPass` runs a use-def dataflow and
gives every `cndmask` the exact `dscnt` that drains just the gather it reads. Over this kernel
that is 1132 inserted waits against the 128 emitted from Python.

The consequence is worth being explicit about: **the pipelining win does not come from the
`SWaitCnt` value this function computes.** StinkyTofu derives the floor itself, from the order
the gathers were issued in. `AccumShuffle=2` earns its speedup purely by issuing the next
pass's gathers before consuming the current one — the wait is bookkeeping, not the mechanism.
**Only the floor distinguishes the two schedules** — 1 bottoms out at `dscnt 0` (nothing in
flight, so the next pass's gather starts cold), 2 bottoms out at the next pass's gather count.

Counted over a whole MT256x256 VWA4 kernel:

```
128 passes, each gathering VW * slotBatch = 4 * 4 = 16 slots. Histogram of the
dscnt value on every accShuffle-tagged wait:

AccumShuffle = 1    {0: 128}          every pass drains to zero
AccumShuffle = 2    {0: 1, 16: 127}   only the final pass drains; the rest
                                      floor at the 16 still in flight for the next pass
```

The payoff, measured on gfx1250 MT128x256x256 / MIWT4_8 / VWA4, is per-wait latency rather
than fewer waits — the count is identical:

| | waits | Stall in the shuffle | cycles per wait |
|---|---|---|---|
| AccumShuffle = 1 | 15360 | 904,908 | **58.9** |
| AccumShuffle = 2 | 15360 | 246,140 | **16.0** |

58.9 cycles is roughly a bare LDS round trip; 16.0 is what is left once the cndmask tree of the
previous pass covers it. This was reproduced across two ClusterDim settings and two code
branches with the ASP=1 figure landing within 0.1% each time — the mechanism is stable even
though whether it *wins overall* is not (§6.2).

### 5.2 Why 0 clamps instead of rejecting

Turning the shuffle off while leaving `VectorWidthA > 1` would produce wrong results, so "off"
*has* to also mean "VW=1". The clamp is silent rather than a rejection because that is precisely
what the `enableLDSTrA` clamp did before — an explicit `VectorWidthA: 4` in the yaml was
overridden then too. `AccumShuffle: 0` reproduces the old behaviour bit for bit.

Every *other* unmet precondition is a rejection with a specific reason, so a yaml that asks for
the shuffle and does not get it reports why:

```
enableLDSTrA with VectorWidthA=N: only 1, 2, 4 and 8 are implemented
enableLDSTrA with VectorWidthA(N) > MatrixInstM(N)
enableLDSTrA with VectorWidthA=N requires MIWaveTile[0](N) to be a multiple of VectorWidthA
enableLDSTrA with VectorWidthA>1 requires MatrixInstBM(N) == 1
enableLDSTrA with VectorWidthA>1 requires SourceSwap
enableLDSTrA with VectorWidthA>1 does not support LocalSplitU>1
enableLDSTrA with VectorWidthA>1 is only implemented for WavefrontSize 32
```

`MIWaveGroup[0] > 1` **is** supported: `strideWave` (`Tensile/Components/LraTileAssignment.py`)
keeps the real `VectorWidth`, so each wave owns exactly the `VW*MatrixInstM` rows the store
assigns it and the transpose never crosses a wave boundary.

### 5.3 Triggering it from yaml

```yaml
- LDSTrInst: [True]            # -> enableLDSTrA
- AccumShuffle: [1]            # or [2]; 0 clamps VectorWidthA to 1
- VectorWidthA: [2]            # or 4, 8; no-op at 1
- StoreRemapVectorWidth: [0]   # storeRemap is the alternative bridge, not a companion
# SourceSwap: 1 is implied by Sparse: 2
```

---

## 6. What to take care of

### 6.1 The default must stay 0

`VectorWidthA` defaults to `-1` (auto). If `enableLDSTrA` were dropped from the clamp
unconditionally, every LDSTr kernel that never mentions `VectorWidthA` would silently move onto
the shuffle path with an auto-derived width:

| dtype | MIWaveTile[0] | auto VWA, sparse | auto VWA, dense |
|-------|---------------|------------------|-----------------|
| f16   | 4             | 2                | **4**           |
| f16   | 8             | 2                | **8**           |
| f8    | 8             | 2                | **8**           |
| f8    | 16            | 4                | **16**          |

Dense has no `findSparseVectorWidth` clamp, so it takes `optVW = 4 // regPerElem` directly.
VW=16 is not implemented and would be rejected — kernels that previously built would stop
building. VW=8 is implemented but measured ~60% slower than VW=2 on one MT256x256 case.

`accshuf_validate.yaml` group 3 is the regression for this: `VectorWidthA: [-1]` with
`AccumShuffle: [0, 1]`, asserted on the generated kernel names (0 must yield `VWA1` for every
MacroTile, 1 must yield VW>1).

### 6.2 `AccumShuffle=1` vs `2` is not predictable — benchmark it

Measured on gfx1250 across MT/VWA/VWB combinations, the pipelined variant ranged from **−11% to
+21%** against the drain variant, winning 9 of 14. Nothing in the configuration predicts the
sign. Gating on `VectorWidthA`, on `MacroTile`, or on `slotBatch` was each tried against the data
and each was contradicted. Treat it as a tuning parameter; fork `[1, 2]` and let tuning pick.

Where it does help, the mechanism is clean and reproducible: on MT128x256x256 / MIWT4_8 / VWA4
the shuffle's `s_wait_dscnt` goes from **58.9 to 16.0 cycles per wait** over the same 15360
waits, measured identically across two ClusterDim settings and two code branches.

### 6.3 The WAR hazard on `ValuC` — `s_wait_dscnt` does not cover it

The commits write `ValuC` registers that in-flight `ds_bpermute` instructions *source*.
`s_wait_dscnt` gates the DS **result**, not the operand read. The operand read is gated by
`s_wait_alu depctr_vm_vsrc`, inserted by the rocisa hazard pass.

**Emission order alone is not the argument.** The scheduler does hoist commits above the next
pass's gather issue — verified in the generated assembly. Safety was checked by replaying the DS
queue against *both* counters over the emitted code; no `ValuC` commit lands with an in-flight
`ds_bpermute` still holding that register unread. **Re-check this if the hazard pass changes.**

### 6.4 Register pressure

`AccumShuffle=2` costs `VW*slotBatch` extra vgprs while `ValuC` is fully live. Whether that is
free depends on how much the pool has left at `endSummation`, which varies with MacroTile. On
MT128x256x256 it was free (`vgpr_count` 1020, 0 spills either way) because the pool reuses
registers already dead by then. **Check for spills on smaller tiles.**

Separately: the dense MT256x256 row in `accshuf_validate.yaml` is absent because that
MatrixInstruction overflows the VGPR budget there (1035 > 1024) at `VectorWidthA=1` with
`AccumShuffle=0` — i.e. with no shuffle emitted at all. That overflow is **not** caused by this
feature.

### 6.5 It grows the epilogue past `simm16`

The shuffle makes the epilogue long enough that the GSU branch overflows the `simm16` range of
`s_cbranch_scc1`. The long-branch condition in `globalWriteElements` was extended to cover it:

```python
if (... or kernel["StoreRemapVectorWidth"]
        or (kernel["SourceSwap"] and kernel.get("enableLDSTrA", False)
            and kernel["VectorWidthA"] > 1)):
```

Counter-intuitively **`VectorWidthA=2` is the worst case**, not 8: `StoreVectorWidth` follows
`VectorWidthA`, so a narrow VW means more and narrower store instructions, and the store side
costs more than the shuffle itself saves. MT256x256 / VWA2 overflows where VWA1 does not.

### 6.6 Silent drops — always check `Actual Solutions`

Two failure modes in this area produce only a **warning** and still let the run exit 0:

- `error: branch size exceeds simm16` (assembly step)
- `total VGPRS (N) overflowed max VGPRS (1024)` (codegen step)

Both drop the kernel and the benchmark reports PASS for whatever remains. Always compare:

```
# Actual Solutions: 44 / 72 after SolutionStructs    <- rejections, expected and reported
# Actual Solutions: 44 / 44 after KernelWriter       <- must match, or kernels vanished
```

### 6.7 Coverage gaps

Validated: 3064 runs over 82 kernels on gfx1250, all PASSED at `NumElementsToValidate: -1`
(full compare, no sampling) — sparse and dense, VWA 2/4/8, AccumShuffle 1 and 2, GSU 1 and 2,
FactorDim 0 and 1, over tile-aligned shapes, M-only and N-only macro-tile edges, a doubly ragged
shape, the 16x16 minimum and a K tail.

Not covered:

- **wave64** — rejected, never derived.
- **batch > 1** — independently broken under `TDMInst=3`, so it is deliberately absent from the
  test config rather than silently failing there.
- **f8 / non-f16 dtypes** — all validation runs were f16.
- **K > 512.**

---

## 7. Where to look when something breaks

| Symptom | Look at |
|---------|---------|
| Wrong M rows in D, only lane 0 correct | The shuffle is not running. Check `AccumShuffle` is 1 or 2, and the kernel name for `VWA` > 1. |
| Kernel name shows `VWA1` but yaml says 4 | `AccumShuffle: 0` clamped it. By design — see §5. |
| Fewer kernels benchmarked than generated | §6.6. Grep the log for `simm16` and `overflowed max VGPRS`. |
| Values wrong only at VW=8 | The group commit. Every destination reads all VW sources; in-place write corrupts the group. |
| `AccumShuffle=2` slower than 1 | Expected on some configurations — §6.2. Not a bug by itself. |
| Correct output but a perf regression after a rocisa change | §6.3. The hazard pass inserts the `depctr_vm_vsrc` that makes pipelining safe; if it changed, re-verify before trusting the schedule. |
