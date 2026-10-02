`src/settings/` — Run configuration {#mftl_mod_settings}
======================================

**Public header:** `include/mftl_settings.h` · **Layer:** 4 ·
**Module group:** @ref mftl_settings

Every runtime option of the library in one struct, #MFTL_Settings, filled from a
settings file and the command line, with one set of defaults, one validation and
one reference. A flow solver keeps one file per simulation; the library's calls
take their parameter blocks straight out of the struct.

| File | Job | MPI? |
|---|---|---|
| `settings_keys.c` | **the key table**: every option, declared once — type, member, default, range, choices, one-line summary | no |
| `settings_text.c` | defaults, the parser, `MFTLSettingsSet`/`Get`, the writer, the combination checks | no |
| `settings_run.c` | loading on rank 0 and broadcasting, the process grid, the ensemble, resolving what depends on the mesh | yes |

The manual's *Settings reference* page and `mftl_settings template` both list
every key with its type, default, admissible values and meaning; both are
generated from `settings_keys.c`.

---

## One struct of the blocks the modules already take {#settings_shape}

#MFTL_Settings has one section per subsystem, and a section **embeds** the
parameter block its module already defines:

| Section | Holds | Passed as |
|---|---|---|
| `domain` | box, periodicity, process grid | read by `MFTLSettingsCreateEnsemble` |
| `sphere` | refinement level, radius, centre | to `MFTLGenerateSphere` |
| `remesh` | `every` + #MFTL_RemeshParameters | `&cfg.remesh.params` to `MFTLRemeshEnsemble` |
| `topology` | `every`, the resolution-relative factors + #MFTL_TopologyOptions | `&cfg.topology.options` to `MFTLEnsembleApplyTopologyChanges` |
| `contact` | `every` + #MFTL_BVHBuildParams | `&cfg.contact.params` to `MFTLEnsembleIntersectObjects` |
| `redistribute` | the two process-wide switches | applied by `MFTLSettingsCreateEnsemble` |
| `geometry` | the vertex-normal weighting | to `MFTLEnsembleUpdateVertexNormals` |
| `output` | `every`, file prefix + #MFTL_VtkOptions | `&cfg.output.vtk` to `MFTLWriteVtu` |

**No module reads the settings.** A function still takes exactly the block it
needs, so every module stays testable in isolation and nothing depends on hidden
state; the struct only gathers the blocks. The file's key names are the member
names, so `remesh.parallel_phase_budget` in a file is
`cfg.remesh.params.parallel_phase_budget` in C and its full rationale is on that
struct field.

The `every` keys are schedules the caller's loop reads through
`MFTLSettingsDue(cfg.remesh.every, step)`; the library has no timestep driver
and does not grow one here. They live in the file because a solver wants the
cadence of its MFTL calls next to the calls' parameters.

### Alternatives considered {#settings_alternatives}

- **A global key/value database that every module queries by name** (PETSc's
  options database, AMReX's `ParmParse`). Convenient, and it lets a solver add
  its own keys. It was not taken because a module reading options behind its
  caller's back is hidden global state: two ensembles in one process could not
  be configured differently, and whether every rank queried the same value
  would be invisible — for options that change the number of collectives
  (`skip_unchanged_bodies`, the remesh strategy) that is a deadlock without a
  diagnostic. The typed struct keeps both visible at the call site.
- **One file per module.** Scatters a simulation's configuration and multiplies
  the loading code; one file with sections gives the same separation.
- **YAML or JSON.** A C11 library with no dependency beyond MPI would have to
  embed a parser for either. The format here is a subset of TOML, which a person
  edits comfortably, which Python reads with the standard `tomllib` (post-
  processing and job scripts can read the same file), and whose subset is
  parsed in a few hundred lines.

What was kept from the database designs is their best property: **declare each
option once, with its type, default, range and documentation, and derive
everything else from that declaration** (deal.II's `ParameterHandler` works the
same way). Here the declaration is one `KEY(...)` line in `settings_keys.c`.

---

## The file {#settings_file}

```toml
# run.toml
[domain]
global_max = [2.0, 1.0, 1.0]
periodic   = [true, false, false]   # one value sets all three: periodic = false

[remesh]
every             = 1
target_length     = 0.0              # 0: hold the mesh's starting resolution
parallel_strategy = "dispersed"

[topology]
every = 5
```

`[section]` headers, `key = value`, `#` comments, integers, reals (`inf`
allowed where the range is unbounded, `nan` never), `true`/`false`, quoted
strings, and three-vectors as `[x, y, z]`. A keyword may be written with or
without quotes, and a three-vector without brackets (`2,2,1`) — both are there
for the command line, where `[` is a shell glob. A dotted key before the first
section header (`remesh.every = 2`) is TOML and is accepted. Keys left out keep
their current value.

**Everything that looks like a mistake is an error**, with the file, the line
and — for a typo — the key that was probably meant:

```
run.toml:7: unknown key remesh.target_lenght -- did you mean remesh.target_length?
run.toml:9: remesh.parallel_strategy: "dispersd" is not one of "auto", "spanning" or "dispersed" -- did you mean "dispersed"?
run.toml:12: remesh.smooth_relaxation: 1.5 is outside [0, 1]
run.toml:15: remesh.every is set twice (first on line 4)
run.toml:18: unknown key topology.parallel_phase_budget -- did you mean remesh.parallel_phase_budget?
```

A misspelled key that is silently left at its default is the one failure a
settings file must not have: the run proceeds on a value nobody chose. Every
problem in a file is reported in one pass, and **a text that has any problem
changes nothing** — it is applied to a copy that is committed only when every
line and the final combination check pass (`test_a_failed_parse_changes_nothing`).

The combination check covers what no single key's range can: a non-empty
domain box on every axis, a shared-file output in binary, explicit edge-length
bounds in order, a shrink slack of 0 or at least 1.

### The command line {#settings_cli}

`MFTLSettingsLoadArgs` takes every argument of the form `section.key=value`
(leading dashes ignored) and applies them after the file, one per line, so an
error names the argument. Everything else is left to the application;
`MFTLSettingsIsOverride` tells it which arguments to skip.

```bash
mpirun -np 64 ./solver run.toml remesh.every=2 output.prefix=out/run3 domain.cuts=0,0,1
```

### Written back exactly {#settings_write}

`MFTLSettingsWrite` writes a struct in the same format, in three forms: compact
(every key), `MFTL_SETTINGS_WRITE_CHANGED` (only what differs from the
defaults) and `MFTL_SETTINGS_WRITE_DOCUMENTED` (a comment before every key:
summary, type, range, choices, default — the template `mftl_settings template`
prints). Reals are written in the **shortest form that reads back to the same
double**, so writing and re-reading is the identity to the bit
(`test_round_trip_is_exact`, which includes `0.1 + 0.2`, a subnormal and a
string with quotes, a backslash and a `#`). Printing the resolved settings at
the start of a run therefore records a configuration that reproduces it.

---

## The call sequence {#settings_sequence}

```c
MFTL_Settings cfg;
MFTLSettingsDefaults(&cfg);
MFTLSettingsLoad(&cfg, "run.toml", MPI_COMM_WORLD);         // rank 0 reads, all parse
MFTLSettingsLoadArgs(&cfg, argc, argv, MPI_COMM_WORLD);     // section.key=value

MFTL_Ensemble *ens = MFTLSettingsCreateEnsemble(&cfg, MPI_COMM_WORLD);
/* ... add and generate the bodies ... */
MFTLSettingsResolve(&cfg, ens);                             // lengths from the mesh
if (rank == 0) MFTLSettingsWrite(&cfg, stdout, 0);          // this run's exact configuration

for (int step = 0; step < n_steps; ++step)
{
  /* move owned vertices, MFTLUpdateGhostPositions, geometry */
  if (MFTLSettingsDue(cfg.remesh.every, step)) MFTLRemeshEnsemble(ens, &cfg.remesh.params, &st);
  else                                         MFTLRedistribute(ens);
  if (MFTLSettingsDue(cfg.topology.every, step)) MFTLEnsembleApplyTopologyChanges(ens, &cfg.topology.options, &ts);
  if (MFTLSettingsDue(cfg.output.every, step))   MFTLWriteVtu(ens, name, &cfg.output.vtk);
}
```

`examples/settings_driven_run.c` is this program in full, with its settings in
`examples/settings/settings_driven_run.toml`; at the same parameters it
reproduces `examples/leveque_remeshing_parallel.c` exactly. A solver with its own
input format can skip the file entirely and forward keys one at a time with
`MFTLSettingsSet(&cfg, "remesh.every", "5", ...)` — the entry point for a
binding from another language as well, since it needs no struct layout.

### What depends on the mesh {#settings_resolve}

Some lengths are only meaningful relative to the mesh, and a settings file for
a family of runs should not hard-code them. A `0` means "derive it", and
`MFTLSettingsResolve` does, once the bodies exist:

| Key | `0` means |
|---|---|
| `remesh.target_length` $L$ | `MFTLEnsembleMeanEdgeLength` — hold the starting resolution |
| `remesh.min_edge_length`, `max_edge_length` | $\tfrac45 L$, $\tfrac43 L$ (Botsch & Kobbelt); squared bounds finalised |
| `topology.dx` | `dx_factor` $\times L$ (default $1.0\,L$) |
| `topology.delta_breakup` | `delta_breakup_factor` $\times L$ (default $1.5\,L$) |
| `topology.delta_coalescence` | `delta_coalescence_factor` $\times L$ (default $1.0\,L$) |

The factors are what the three open-surface examples hard-code (the liquid jet
uses 0.5 for coalescence). Resolved values are written into the
struct, so after the call the file's zeros are gone and the written
configuration holds the numbers the run actually used. The call is idempotent,
and it also points `topology.options.remesh` at `cfg.remesh.params` when
`topology.remesh_after` is set — the one pointer INTO the struct, which a copy
of the struct made afterwards still aims at the original.

Nothing to measure is an error only when something scheduled needs the length
(`remesh.every > 0`, or `topology.every > 0`); otherwise the zeros stay, and a
remesh attempted with them is refused loudly by the remesher's own band check.

### Two defaults differ from the module's, on purpose {#settings_defaults}

`MFTLSettingsDefaults` starts every module block from its module's own defaults
and then applies the table, and `test_defaults_follow_the_modules` asserts that
the two agree on every module key but one:

- **`remesh.parallel_phase_budget` is 4, not 32.** A settings file describes a
  simulation, i.e. a timestep loop, where 4 was measured best and the API
  default of 32 re-converges a good mesh every step at five times the cost and a
  worse volume (the table on #MFTL_RemeshParameters). 32 remains right for a
  one-shot remesh, which a caller who wants one sets.
- **The arguments `MFTLInitRemeshParams` has no default for** — `target_length`
  0 (derive), 3 iterations, triangle-based smoothing, and feature detection off
  at 180 degrees, which is right for a smooth interface.

---

## Collective state {#settings_collective}

Several options change how many collectives run or how long their messages
are — the remesh strategy, the phase budget, `skip_unchanged_bodies` — so the
struct **must be identical on every rank**, and a difference is a deadlock with
no diagnostic. Two mechanisms guarantee it:

- **Loading broadcasts the text, not the struct.** Rank 0 reads the file (the
  other ranks never touch the file system) and every rank parses the same
  bytes with the same parser. Every rank therefore reaches the same struct and
  the same verdict; no rank can take an error path the others skip. A missing
  file is broadcast as a length of −1 so that it, too, fails everywhere.
- **Agreement is verified, once.** `MFTLSettingsCreateEnsemble` and
  `MFTLSettingsResolve` hash every key's text form and reduce the hashes with
  `MPI_MIN` together with their bitwise complements — the minimum of $\neg h$ is
  the complement of the maximum of $h$ — so one `MPI_Allreduce` of $2K$
  integers yields both extremes, and a difference names its key:

  ```
  [MFTL] MFTLSettingsResolve: the ranks hold different settings (first difference: remesh.iterations).
  ```

  This catches a caller that changed a field from rank-local data after loading
  (`test_disagreeing_ranks_are_caught`, np = 2). It runs twice per job, not per
  timestep.

`MFTLSettingsCreateEnsemble` applies the two `[redistribute]` switches BEFORE the
ensemble exists, so that the first redistribution — the scatter of generated
bodies, where `shrink_slack` matters most — already runs under the configured
policy.

### The process grid {#settings_grid}

A zero in `domain.cuts` is chosen by `MFTLSettingsChooseCuts`: among the
factorisations of the rank count that keep the non-zero entries, the one whose
**narrowest rank box is widest**, then the one whose widest box is narrowest,
then more cuts on the lower axis. The narrowest width $h_\min$ is the quantity to
protect: it is what the spanning remesher's band condition $h > 4\delta$ is
taken against. On a cube this reproduces the table the LeVeque example spells
out by hand (2 → 2×1×1, 4 → 2×2×1, 6 → 3×2×1, 24 → 4×3×2, 32 → 4×4×2); on a
24×4×4 channel at 4 ranks it gives 4×1×1 where `MPI_Dims_create`, which ignores
the box, would give 2×2×1 and halve $h_\min$.

---

## Adding an option {#settings_adding}

1. Add the member to its module block (or to the section struct in
   `mftl_settings.h` for a run-level value), with its doc comment.
2. Add one `KEY(...)` line to `settings_keys.c`: section, name, type, member,
   default as file text, range, choices, unit, advanced flag, summary.
3. If the member belongs to #MFTL_RemeshParameters, also add it to
   `MFTL_REMESH_AGREED_FIELDS` in `src/remeshing/remesh_comm.c`.

Nothing else: the parser, the writer, `MFTLSettingsSet`, the tool and the
manual's reference page pick it up. The tests then hold the entry to its
member — `test_every_key_matches_its_member` compares the declared type with the
member's size (a `REAL` over an `int` would write eight bytes into four),
`test_every_default_is_admissible` parses every default and checks it against
its range, and `test_defaults_follow_the_modules` compares every module-block
default with the module's own. The table's format is also read by the
documentation build, so keep to the one-entry-per-macro form its header
describes.
