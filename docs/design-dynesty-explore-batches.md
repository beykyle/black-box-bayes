# Design: full-range exploration batches for dynesty (`add_batch`)

Status: **proposal, not implemented.** Written against `dynesty 3.1.0` and
`src/black_box_bayes/cli.py` as of the `alpha_ca48` work.

## Why

`black-box-bayes` currently drives dynesty through a single `run_nested(...)` call. For
dynamic runs that means every batch after the initial one is allocated by dynesty's
*weight function*, which targets the likelihood range where adding points most improves
the posterior or the evidence (per `--dynesty-pfrac`). That is the right default and it
is not what you want when the problem is **mode discovery**.

dynesty's FAQ is explicit about the gap:

> if you have a really large number of modes, you can use `add_batch` functionality with
> `logl_bounds` of `(-inf, inf)`, to basically do repeated sampling of the posterior. In
> this case you have a good chance of discovering all the modes in your data.

There is no CLI flag for this today, and the `alpha_ca48` example is a concrete case
where it matters. Measured there (6 parameters, ~32 nats of information):

| nlive | ln Z (volume) | distinct `Vv` basins in the dead points |
|---|---|---|
| 500 | −308.64 | 1 |
| 1500 | −80.71 | 2 |
| 4000 | −29.48 | 3 |
| 12000 | −30.09 | 4 |

The modes only separate in the last ~200 nats of the compression, so the basin count was
still climbing at nlive = 12000 and the evidence swung 279 nats between the cheapest and
converged runs. Weight-function batches concentrate effort where the *current* run says
it is valuable, which by construction cannot be a mode the run has not found. Full-range
batches restart the compression from the prior and get an independent shot at every mode.

There is a second motivation. A frequent workflow is to use nested sampling as a cheap
mode-*cataloguing* pass — find the basins, then build per-mode priors and calibrate each
separately. The posterior is the wrong product for that (it correctly weights subdominant
modes out of existence: at nlive = 12000 the `alpha_ca48` posterior is ~100% one mode).
The **dead points** are the right product, and full-range batches are the cheapest way to
enrich them.

## What dynesty provides

`DynamicSampler.add_batch()` in 3.1.0:

```python
add_batch(nlive=500, dlogz=0.01, mode='weight', wt_function=None, wt_kwargs=None,
          maxiter=None, maxcall=None, logl_bounds=None, save_bounds=True,
          print_progress=True, print_func=None, stop_val=None, resume=False,
          checkpoint_file=None, checkpoint_every=None)
```

Relevant behaviour, read from the source rather than the docs:

- `mode` is one of `'auto' | 'weight' | 'full' | 'manual'`.
- `mode='full'` is the one we want. It leaves `logl_bounds = None`, which downstream means
  the whole range — the FAQ's `(-inf, inf)` without having to pass it.
- `mode='manual'` is the **only** mode that accepts explicit `logl_bounds`; passing them
  with any other mode raises `RuntimeError`. So do not implement this by passing
  `logl_bounds=(-inf, inf)` — use `mode='full'`.
- Batches are merged by `combine_runs()`, so `sampler.results` after N batches is a single
  consistent dynamic-nested-sampling result. `logz`, `logzerr`, `samples`, `logl` all
  reflect the combined run; no bookkeeping is needed on our side.
- The source notes that when `checkpoint_every` is passed explicitly, `add_batch` assumes
  it is "running externally" and manages its own timer rather than `run_nested`'s global
  one. Since we would be calling it externally, pass both `checkpoint_file` and
  `checkpoint_every`.

## Proposed CLI surface

Three flags, all no-ops unless `--dynesty-run dynamic`:

| flag | type | default | meaning |
|---|---|---|---|
| `--dynesty-explore-batches` | int | `0` | Number of **full-range** batches to append after `run_nested` returns. `0` preserves today's behaviour exactly. |
| `--dynesty-explore-nlive` | int | falls back to `--nlive-batch`, else dynesty's 500 | Live points per exploration batch. |
| `--dynesty-explore-maxcall` | int | `None` | Per-batch likelihood-call cap. Cheap insurance: a full-range batch redoes the whole prior-to-posterior compression, so it is not obviously bounded. |

Naming note: "explore" rather than "extra" or "full" because the flags describe intent.
`--dynesty-explore-batches 4` should read as "take four more independent looks at the
whole prior".

Rejected alternatives:

- **Overloading `--maxbatch`.** That already means "how many weight-function batches
  `run_nested` may take". Conflating the two would make it impossible to ask for
  *both* kinds, which is the normal case.
- **A `--dynesty-batch-mode {weight,full}` switch on `run_nested`.** `run_nested` does not
  expose `mode`; it always uses the weight function. Implementing this would mean
  reimplementing `run_nested`'s batch loop, which is a much larger change for no gain.
- **Exposing `mode='manual'` + `--dynesty-logl-bounds`.** More general, but the only
  documented use is the `(-inf, inf)` case, and manual bounds are easy to set
  incorrectly in a way that silently biases the evidence. Can be added later if wanted.

## Implementation sketch

All of this lives in `run_dynesty` (`src/black_box_bayes/cli.py`). The single insertion
point is between `sampler.run_nested(**run_kwargs)` and `dt = time() - t0`.

```python
    run_kwargs = _filter_kwargs_for_callable(sampler.run_nested, run_kwargs)
    sampler.run_nested(**run_kwargs)

    # --- NEW: full-range exploration batches -------------------------------------
    n_explore = getattr(args, "dynesty_explore_batches", 0) or 0
    if n_explore and dynesty_run == "dynamic":
        batch_nlive = args.dynesty_explore_nlive or args.nlive_batch
        print(
            f"Adding {n_explore} full-range exploration batch(es), "
            f"nlive={batch_nlive or 'dynesty default'} each. These re-sample the whole "
            "prior rather than the weight function's preferred slice, which is what "
            "gives previously-missed modes a chance to appear."
        )
        for i in range(n_explore):
            batch_kwargs = dict(
                mode="full",                 # NOT logl_bounds -- see design note
                nlive=batch_nlive,
                maxcall=args.dynesty_explore_maxcall,
                print_progress=args.dynesty_progress,
                checkpoint_file=str(checkpoint_path),
                checkpoint_every=args.dynesty_checkpoint_every,
            )
            batch_kwargs = {k: v for k, v in batch_kwargs.items() if v is not None}
            batch_kwargs = _filter_kwargs_for_callable(sampler.add_batch, batch_kwargs)
            sampler.add_batch(**batch_kwargs)
            logz = sampler.results["logz"][-1]
            print(f"  explore batch {i + 1}/{n_explore}: ncall={sampler.ncall} "
                  f"logz={logz:.3f}")
    # -------------------------------------------------------------------------------

    dt = time() - t0
```

Notes on the sketch:

- Reuse `_filter_kwargs_for_callable`, as the rest of the file does, so a dynesty version
  that renames or drops a kwarg degrades instead of crashing.
- Strip `None` values before filtering: `add_batch`'s defaults are meaningful (`nlive=500`),
  and passing `None` explicitly is not the same as omitting it.
- Print `logz` after each batch. If it moves by ≫ its own error bar, that batch found
  something, and that is exactly the signal a user needs.

### Argument parsing

Beside the other dynesty flags in `parse_args` (~line 940):

```python
    parser.add_argument("--dynesty-explore-batches", type=int, default=0,
                        help="Number of full-range (mode='full') batches to append after "
                             "the main dynamic run. Full-range batches re-sample the "
                             "entire prior instead of the weight function's preferred "
                             "likelihood slice, which is dynesty's recommended way to "
                             "discover modes a run has missed. Dynamic runs only.")
    parser.add_argument("--dynesty-explore-nlive", type=int, default=None,
                        help="Live points per exploration batch (default: --nlive-batch, "
                             "else dynesty's default).")
    parser.add_argument("--dynesty-explore-maxcall", type=int, default=None,
                        help="Per-batch likelihood-call cap for exploration batches.")
```

### Validation

In the existing positive-integer check block (~line 1010), add
`("--dynesty-explore-nlive", args.dynesty_explore_nlive)` and
`("--dynesty-explore-maxcall", args.dynesty_explore_maxcall)`;
`--dynesty-explore-batches` should be validated as **non-negative** (0 is the default and
must stay legal), so it belongs with `--burnin` / `--idata-discard` instead.

Add an explicit guard, because silently ignoring a flag the user set is the failure mode
this whole example has been documenting:

```python
    if args.dynesty_explore_batches and args.dynesty_run != "dynamic":
        parser.error("--dynesty-explore-batches requires --dynesty-run dynamic")
```

### Recorded metadata

`_dynesty_to_inferencedata` builds an `extra` dict of attrs. Add:

```python
        "dynesty_explore_batches": int(getattr(args, "dynesty_explore_batches", 0) or 0),
        "dynesty_explore_nlive": ...,
```

so a result file records whether exploration batches were used. Without this, two `.nc`
files with very different provenance are indistinguishable — and given that this feature
changes what the evidence converges to, that matters.

## Semantics and edge cases

**The evidence stays valid.** `combine_runs` merges batches into one dynamic-NS result, so
`logz` remains a proper estimator. But note that adding full-range batches can *change*
`logz` substantially — that is the point. A jump means the batch found mass the original
run missed, i.e. the original number was wrong. It is not a bug and should not be
smoothed over in the output.

**`logzerr` will not warn you.** Same lesson as elsewhere in this project: the quoted
error is the statistical error of the compression actually performed. In `alpha_ca48`,
dynesty reported ±0.28 on a value that was 279 nats wrong. Exploration batches are a
mitigation for a failure mode the error bar cannot see, so the docs should say plainly
that the way to gain confidence is *batch-to-batch stability of `logz`*, not a small
`logzerr`.

**Resume.** `run_nested`'s own resume is unchanged. Resuming *into* the exploration loop is
not handled by this sketch: a job that dies during batch 3 of 5 restarts the whole
exploration phase on resume. Given each batch checkpoints, a more careful design could
count completed batches from the restored sampler state (`sampler.batch`). I would leave
that out of v1 and document the limitation rather than get it subtly wrong.

**MPI.** Nothing special. `add_batch` uses the sampler's existing `pool`, and only the
master rank reaches this code (workers are inside `pool.wait()`).

**Interaction with `--dynesty-use-stop`.** Independent. `use_stop` governs when
`run_nested`'s weight-function batches stop; exploration batches run afterwards
unconditionally. Worth a sentence in the help text, since a user could reasonably expect
`use_stop` to cover both.

**Static runs.** `NestedSampler` has no `add_batch`. Hence the guard above.

## Testing

Mirror the existing patterns in `tests/test_cli_unit.py` (monkeypatched, no real sampling):

1. **Parsing/validation** — `--dynesty-explore-batches 3` parses; combining it with
   `--dynesty-run static` exits non-zero; negative values rejected; `0` accepted.
2. **Call plumbing** — monkeypatch a fake sampler exposing `run_nested` and `add_batch`,
   assert `add_batch` is called exactly N times with `mode="full"` and **without**
   `logl_bounds` (the `RuntimeError` guard in dynesty makes that combination fatal, so it
   is worth pinning).
3. **Kwarg filtering** — a fake `add_batch` whose signature lacks `checkpoint_every` must
   still be called successfully.
4. **Attrs** — `dynesty_explore_batches` appears in the written `.nc` attrs.

An end-to-end test against the toy config would be nice but should be gated behind
`BBB_RUN_SLOW` like `test_end_to_end_cli`, since a dynamic run with extra batches is not
fast.

## Documentation

- `README.md`: one row in the dynesty flag list, plus a short paragraph in the dynesty
  section explaining *when* to reach for it (multimodal, or modes suspected to separate
  only at high likelihood) and that the dead points, not the posterior, are what you mine
  for a mode catalogue.
- `examples/alpha_ca48/README.md`: this is the natural worked example. The basin table
  above already shows the problem; a follow-up row showing basin count with exploration
  batches would close the loop.

## Open questions for review

1. **Default `0` or something non-zero for dynamic runs?** I have proposed `0` (strictly
   opt-in, no behaviour change). The argument for a small non-zero default is that a user
   running dynamic NS on an unknown posterior probably *should* pay for one full-range
   batch. I lean opt-in, but it is your call.
2. **Should exploration batches run before the weight-function batches?** Discovering
   modes first would let the weight function allocate across all of them rather than
   optimizing within the subset found by the initial run. That is plausibly better and is
   a bigger change — `run_nested(maxbatch=0)`, then explore, then a second `run_nested`.
   Worth considering if v1 shows the ordering matters.
3. **Expose `mode='manual'` + explicit bounds?** Only if a concrete use case appears.
4. **Is per-batch `dlogz` worth exposing?** `add_batch` takes `dlogz=0.01`; the sketch
   leaves it at dynesty's default. Probably fine.
