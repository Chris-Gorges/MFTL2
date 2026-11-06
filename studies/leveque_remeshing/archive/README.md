# Archived remeshing code

Code that the LeVeque study (`../REPORT.md`, `../PAPER_NOTES.md`) measured and that was
then removed from the library. Each patch applies with `git apply` to a checkout of
commit `bb3c785` (`src/`, `include/`, `examples/`, `tests/`, `CMakeLists.txt`), in this
order:

| patch | adds | needed for |
|---|---|---|
| `sizing_with_anisotropic_and_qem.patch` | the sizing modes as they were during the study: uniform, curvature (Dunyach et al. 2013), anisotropic (Jiao et al. 2010), quadric-error placement (`qem_placement`, the example's `-qem`), with their tests | every run of the study; the anisotropic and QEM rows |
| `sweep_orders.patch` (on top of the first) | Jiao et al.'s sweep order and length-sorted sweeps (`sweep_order`, `sorted_sweeps`; the example's `-order`, `-sorted`, `-smoothevery`) | REPORT sections 5 and 6 |
| `../followup_experiment.patch` (on top of the first only) | the follow-up's instrumentation: per-sweep volume budget, back-projections, Phong midpoints, Delaunay flips, all behind `MFTL_EXP_*` variables | REPORT section 7 |

`sweep_orders.patch` was rebuilt from the session record after the code had been
deleted: the driver from a full read of the file, the rest from the deletion edits.
Applying those deletions to it gives back the first patch's tree byte for byte, and three
study runs re-run with it (`uniform_r4_jiao`, `curvature_r4_jiao-sorted`,
`uniform_r4_mftl-sorted`) match the stored series in every column except the wall-clock
time. Its tests were not recovered; the code builds without warnings.

Build each variant in its own build directory, not in `build/`, so a running campaign
never has its binary replaced.

## Experiment switches against the CURRENT tree

These are not removed features but switches that existed only for one measurement. Each
applies with `git apply` to the working tree it was measured on (the placement and
projection work, then the correction module) and adds an environment variable or a
schedule that the library does not ship:

| patch | adds | needed for |
|---|---|---|
| `projection_schedule_experiment.patch` | `MFTL_EXP_PROJSWEEP`: Phong projection after every sweep, or after the last sweep of each iteration | PAPER_NOTES section 9.5 (`projEvery*`, `projS1`) |
| `ghost_normals_experiment.patch` | a ghost exchange of vertex normals before the paraboloid fits | PAPER_NOTES section 9.6 |
| `volume_area_step_experiment.patch` | `MFTL_EXP_VOLAREA=1`: `MFTLEnsembleCorrectVolumes()` divides by the surface AREA instead of the exact first variation -- mftl's `VolumeRecovery` with `iterations = 1` | `correction_study.py`, variant `vol_end_area1` (PAPER_NOTES section 10) |
| `volume_budget_experiment.patch` | `MFTL_EXP_VBUDGET=1` (per-sweep budget and the exact volume change of every collapse and flip, by bending angle), `MFTL_EXP_FLIPGUARD=τ`, `MFTL_EXP_FLIPCOMP=1` (exact flip compensation by a common shift). The volume-preserving collapse it first carried as `MFTL_EXP_VPCOLLAPSE=1` is now the library's `collapse_preserves_volume` (`-vpcollapse 1`), and the patch was rebased onto that tree; with the flag it reproduces the stored runs to every printed digit | `volume_ops_study.py`, PAPER_NOTES section 10.6 |
