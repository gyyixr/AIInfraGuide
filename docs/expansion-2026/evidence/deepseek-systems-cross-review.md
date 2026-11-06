# DeepSeek systems cross-review, 2026-11-07

Scope: read-only review of module seven 6.1—6.5, 7.1, 7.2, after reading 5.1/5.2 (subsequently assigned to architecture reviewer). No target document edits.

## Confirmed correction

- 6.1 DeepGEMM, line 245: the arithmetic-intensity ratio immediately above is called a lower-bound model. Under the stated single-read/write idealization, denominator bytes are the minimum traffic estimate and the resulting arithmetic intensity is optimistic/an upper bound, not a lower bound. More actual traffic lowers arithmetic intensity. Clarify the cold-data/whole-operation assumptions and distinguish traffic lower bound from AI upper bound.

## Material boundary worth adding

- 6.5 DeepJIT: content-addressed key discussion omits in-process digest cache lifetime. Cached fixed README line 290 states post-hook hashes are cached per thread and changing the hook needs process restart. Line 346 states tracked header digests are cached for a runtime lifetime and changing tracked headers needs runtime recreation. Add this to edit/rebuild/debug advice; a live runtime is not guaranteed to notice file edits immediately.

## Checked without additional substantive issue

- DeepEP V2.5 versus V1/NVSHMEM boundary, GPU SM use, NCCL requirements.
- EPHandle expanded psum alignment formula, token versus expert-assignment counts.
- EventOverlap first wait consumes the post-wait hook, deferred return tuple, buffer lifetime.
- EPLB greedy replica example, mapping directions and LPLB early-research/old-integration boundary.
- DeepSelect finite-input scan/filter/compact oracle, expectation proof, sort and NaN backend limits.
- DeepJIT digest fields, atomic directory publication versus durability, multi-root and timeout statements.
- 3FS pending/committed behavior, relaxed reads, eventual active file length, close/fsync and recovery state distinctions.
- smallpond fixed Session Ray initialization, memory_allocator=system, by_rows, Hash partition signatures, write_parquet blocking versus lazy, task-level recovery documentation.

Evidence: .local/expansion-2026 cached fixed READMEs, EP source, 3FS design notes; fixed raw smallpond commit 52ecc5e45535c7448f848bcb45b0da00d9484f81 session.py/dataframe.py/docs/source/internals.rst fetched read-only. No GPU, RDMA, storage deployment, or recovery execution was attempted.
