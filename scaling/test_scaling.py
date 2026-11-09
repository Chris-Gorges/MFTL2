#!/usr/bin/env python3
"""Regression checks for submission/reporting; no Slurm installation required.

python3 scaling/test_scaling.py
python3 scaling/test_scaling.py --mpi   # also runs the built drivers on 1/2/4/8 ranks
"""

import csv
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
RUN_MPI = "--mpi" in sys.argv
if RUN_MPI:
    sys.argv.remove("--mpi")


def run(args, env=None, **kwargs):
    return subprocess.run(args, text=True, capture_output=True, timeout=120,
                          env={**os.environ, **(env or {})}, **kwargs)


def executable(path, contents):
    path.write_text(contents)
    path.chmod(0o755)


class Scripts(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="mftl-scaling-test-")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        shutil.copytree(ROOT / "scaling", self.root / "scaling")
        (self.root / "build").mkdir()
        (self.root / "build/CMakeCache.txt").write_text(
            "CMAKE_BUILD_TYPE:STRING=Release\nMFTL_COMM_PROFILE:BOOL=ON\n")
        executable(self.root / "build/strong_scaling", "#!/bin/bash\nexit 0\n")
        (self.root / "mockbin").mkdir()
        executable(self.root / "mockbin/sbatch", '''#!/usr/bin/env python3
import json, os, sys
with open(os.environ["SUBMIT_LOG"], "a") as f:
    f.write(json.dumps({"args": sys.argv[1:], "np": os.getenv("NRANKS")}) + "\\n")
if os.getenv("FAIL_RANK") == os.getenv("NRANKS", "collector"):
    sys.exit(1)
print(100 + int(os.getenv("NRANKS", "0")))
''')
        self.env = {
            "PATH": str(self.root / "mockbin") + os.pathsep + os.environ["PATH"],
            "SUBMIT_LOG": str(self.root / "submits.jsonl"), "RANKS": "1 72 1080",
            "OUT_DIR": str(self.root / "sweep"), "JOB_REPEATS": "1",
            "CORES_PER_NODE": "36", "MAX_NODES": "30", "RANKS_PER_NODE": "36",
            "NODES": "", "ALLOW_NO_PROFILE": "0",
            "EXCLUSIVE": "1", "MEMORY_PER_NODE": "0", "SBATCH_EXTRA": "",
            "BUILD_DIR": str(self.root / "build"),
            "SLURM_JOB_ID": "",
        }

    def launch(self, **env):
        return run(["bash", str(self.root / "scaling/launch_strong.sh")],
                   env={**self.env, **env})

    def test_submission_snapshot_and_placement(self):
        result = self.launch()
        self.assertEqual(result.returncode, 0, result.stderr)
        calls = [json.loads(s) for s in (self.root / "submits.jsonl").read_text().splitlines()]
        self.assertEqual(len(calls), 4)
        for call, nodes, rpn in zip(calls, (1, 2, 30), (1, 36, 36)):
            self.assertIn(f"--nodes={nodes}", call["args"])
            self.assertIn(f"--ntasks-per-node={rpn}", call["args"])
            self.assertIn("--mem=0", call["args"])
            self.assertTrue(call["args"][-1].endswith("snapshot/job_scaling.sh"))
        self.assertIn("--dependency=afterany:101:172:1180", calls[-1]["args"])
        original = self.root / "build/strong_scaling"
        original.write_text("rebuilt")
        self.assertNotEqual(original.read_text(), (self.root / "sweep/snapshot/strong_scaling").read_text())
        self.assertNotEqual(self.launch().returncode, 0)  # cannot overwrite results

    def test_no_silent_density_change(self):
        result = self.launch(RANKS="1080", RANKS_PER_NODE="32")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("skipped", (self.root / "sweep/points.tsv").read_text())
        self.assertFalse((self.root / "submits.jsonl").exists())

    def test_default_campaign_is_one_size_and_full_nodes(self):
        for mode, size in (("strong", "--spheres 110592"), ("weak", "--spheres-per-rank 1000")):
            executable(self.root / f"build/{mode}_scaling", "#!/bin/bash\nexit 0\n")
            env = {**self.env, "RANKS": "", "OUT_DIR": str(self.root / mode),
                   "RANKS_PER_NODE": "", "SPHERES": "", "PER_RANK": "", "TIME_LIMIT": "",
                   "MAX_NODES": "", "MEMORY_PER_NODE": ""}
            result = run(["bash", str(self.root / f"scaling/launch_{mode}.sh"), "--dry-run"], env=env)
            self.assertEqual(result.returncode, 0, result.stderr)
            with (self.root / mode / "points.tsv").open() as f:
                points = list(csv.DictReader(f, delimiter="\t"))
            eligible = [p for p in points if p["status"] == "dry-run"]
            self.assertEqual([int(p["np"]) for p in eligible], [36, 72, 144, 288])
            self.assertTrue(all(int(p["rpn"]) == 36 and int(p["np"]) == 36*int(p["nodes"]) for p in eligible))
            self.assertIn(size, (self.root / mode / "sweep.txt").read_text())
            self.assertEqual(result.stdout.count("--time=01:00:00"), 4)
            self.assertNotIn("--mem=", result.stdout)

    def test_node_cap_filters_without_changing_problem_or_density(self):
        for cap, expected in ((1, [1]), (3, [1, 2]), (8, [1, 2, 4, 8]),
                              (10, [1, 2, 4, 8]), (30, [1, 2, 4, 8, 16, 30])):
            out = self.root / f"cap-{cap}"
            result = self.launch(RANKS="", MAX_NODES=str(cap), OUT_DIR=str(out))
            self.assertEqual(result.returncode, 0, result.stderr)
            with (out / "points.tsv").open() as f:
                points = list(csv.DictReader(f, delimiter="\t"))
            submitted = [p for p in points if p["status"] == "submitted"]
            self.assertEqual([int(p["nodes"]) for p in submitted], expected)
            self.assertTrue(all(int(p["np"]) == 36*int(p["nodes"]) and p["rpn"] == "36" for p in submitted))
            config = (out / "sweep.txt").read_text()
            self.assertIn("--spheres 110592", config)
            self.assertIn("  ranks         " + " ".join(str(36*n) for n in expected) + "\n", config)
            calls = [json.loads(s) for s in (self.root / "submits.jsonl").read_text().splitlines()]
            dependency = "--dependency=afterany:" + ":".join(str(100+36*n) for n in expected)
            self.assertIn(dependency, calls[-1]["args"])
            result = run(["bash", str(self.root / "scaling/collect.sh"), str(out)])
            self.assertEqual(result.returncode, 0, result.stderr)
            missing = next(l for l in (out / "summary.txt").read_text().splitlines() if "no verified report" in l)
            for p in points:
                if p["status"] == "skipped":
                    self.assertNotIn(p["tag"], missing)

    def test_custom_nodes_follow_rank_density_and_ranks_override(self):
        result = self.launch(RANKS="", NODES="1 3 6", MAX_NODES="6", CORES_PER_NODE="24", RANKS_PER_NODE="24")
        self.assertEqual(result.returncode, 0, result.stderr)
        calls = [json.loads(s) for s in (self.root / "submits.jsonl").read_text().splitlines()]
        for call, n in zip(calls, (1, 3, 6)):
            self.assertIn(f"--nodes={n}", call["args"])
            self.assertIn(f"--ntasks={n*24}", call["args"])
            self.assertIn("--ntasks-per-node=24", call["args"])
        result = self.launch(RANKS="1 2", NODES="1 3 6", OUT_DIR=str(self.root / "explicit-ranks"))
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("  ranks         1 2\n", (self.root / "explicit-ranks/sweep.txt").read_text())

    def test_optional_memory_request(self):
        for i, value in enumerate(("default", "", "0", "100G", "64000")):
            result = self.launch(RANKS="36", MEMORY_PER_NODE=value, OUT_DIR=str(self.root / f"memory-{i}"))
            self.assertEqual(result.returncode, 0, result.stderr)
            calls = [json.loads(s) for s in (self.root / "submits.jsonl").read_text().splitlines()]
            args = calls[-2]["args"]
            expected = [] if value in ("default", "") else [f"--mem={value}"]
            self.assertEqual([a for a in args if a.startswith("--mem=")], expected)
            self.assertIn("--exclusive", args)
            self.assertFalse(any(a.startswith("--mem=") for a in calls[-1]["args"]))

    def test_unprofiled_build_is_refused_before_submission(self):
        (self.root / "build/CMakeCache.txt").write_text(
            "CMAKE_BUILD_TYPE:STRING=Release\nMFTL_COMM_PROFILE:BOOL=OFF\n")
        result = self.launch()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("--comm-profile", result.stderr)
        self.assertFalse((self.root / "submits.jsonl").exists())

    def test_submission_failure_and_missing_repeat(self):
        result = self.launch(RANKS="1 4", FAIL_RANK="4", JOB_REPEATS="2")
        self.assertNotEqual(result.returncode, 0)
        self.assertTrue((self.root / "sweep/reports/np00004_r2.failed").exists())
        result = run(["bash", str(self.root / "scaling/collect.sh"), str(self.root / "sweep")])
        self.assertEqual(result.returncode, 0, result.stderr)
        summary = (self.root / "sweep/summary.txt").read_text()
        self.assertIn("np00001_r2(job=101,status=submitted)", summary)
        self.assertIn("np00004_r2", summary)

    def test_invalid_rank_settings(self):
        for env in ({"RANKS": "0"}, {"RANKS": "1 1"}, {"RANKS_PER_NODE": "37"},
                    {"MAX_NODES": "0"}, {"JOB_REPEATS": "oops"},
                    {"RANKS": "", "NODES": "1 0"}, {"RANKS": "", "NODES": "1 1"},
                    {"RANKS": "", "NODES": "1 x"}, {"MEMORY_PER_NODE": "-1"},
                    {"MEMORY_PER_NODE": "100 G"}):
            self.assertNotEqual(self.launch(**env).returncode, 0, env)

    def test_collector_excludes_invalid_and_preserves_samples(self):
        sweep = self.root / "sweep"
        reports = sweep / "reports"
        reports.mkdir(parents=True)
        fields = ("mode=strong motion=flow np=4 nodes=2 rpn=2 bodies=8 refine=1 "
                  "tris=640 tris_rank=160 profiled=1 optimised=1 "
                  "repeats=1 requested_repeats=3 budget_stop=1 init_s=0.5 init_batches=8 init_batch_pool_mb=12")
        for i, t in enumerate((8, 4, 6, 0.001), 1):
            (reports / f"np00004_r{i}.txt").write_text(
                f"@summary {fields} ms_step={t} ms_median={t+1} ok={int(i != 4)}\n")
        # A valid-looking summary is still invalid if MPI returned a failure.
        (reports / "np00004_r5.txt").write_text(f"@summary {fields} ms_step=0.0001 ok=1\n")
        (reports / "np00004_r5.failed").write_text("launcher failed\n")
        (reports / "np00004_r6.txt").write_text("interrupted report\n")
        result = run(["bash", str(self.root / "scaling/collect.sh"), str(sweep)])
        self.assertEqual(result.returncode, 0, result.stderr)
        summary = (sweep / "summary.txt").read_text()
        self.assertIn("one node (P = 0)", summary)  # np <= 36 does not imply one node
        self.assertIn("EXCLUDED FROM TIMING TABLES", summary)
        self.assertIn("Soft runtime budget reduced", summary)
        row = summary.split("core-ms    node-ms\n")[1].splitlines()[0].split()
        self.assertEqual(row[:5], ["4", "3", "4.0000", "6.0000", "8.0000"])
        with (sweep / "samples.tsv").open() as f:
            rows = list(csv.DictReader(f, delimiter="\t"))
        self.assertEqual(len(rows), 3)
        self.assertEqual(sorted(float(r["ms_min"]) for r in rows), [4, 6, 8])
        self.assertTrue(all(r["budget_stop"] == "1" and r["repeats"] == "1" and
                            r["requested_repeats"] == "3" and r["init_batches"] == "8" for r in rows))

    def test_job_records_module_and_report_failures(self):
        sweep = self.root / "sweep"
        snapshot = sweep / "snapshot"
        snapshot.mkdir(parents=True)
        site = snapshot / "site.sh"
        site.write_text("site_modules() { return 7; }\nsite_runtime_env() { :; }\n")
        env = {**self.env, "MODE": "strong", "NRANKS": "1", "MFTL_ROOT": str(self.root)}
        job = ["bash", str(self.root / "scaling/job_scaling.sh")]
        result = run(job, env=env)
        self.assertEqual(result.returncode, 7, result.stderr)
        marker = sweep / "reports/np00001.failed"
        self.assertIn("exit=7", marker.read_text())
        site.write_text("site_modules() { :; }\nsite_runtime_env() { :; }\n"
                        "LAUNCHER=(bash -c 'shift; exec \"$@\"' --)\n")
        executable(snapshot / "strong_scaling", "#!/bin/bash\nexit 0\n")
        result = run(job, env=env)
        self.assertEqual(result.returncode, 2, result.stderr)  # success without a report
        executable(snapshot / "strong_scaling", '''#!/bin/bash
while [ "$#" -gt 0 ]; do
  if [ "$1" = --report ]; then echo '@summary np=1 ms_step=1 ok=1' >"$2"; break; fi
  shift
done
''')
        result = run(job, env=env)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertFalse(marker.exists())

    def test_remesh_launchers_submit_the_remesh_drivers(self):
        for mode in ("remesh_strong", "remesh_weak"):
            executable(self.root / f"build/{mode}_scaling", "#!/bin/bash\nexit 0\n")
        for launcher, regime, rpn in (("strong", "dispersed", 36), ("strong", "spanning", 36),
                                      ("weak", "dispersed", 36), ("weak", "spanning", 32)):
            out = self.root / f"{launcher}-{regime}"
            env = {**self.env, "RANKS": "", "NODES": "1 2", "MAX_NODES": "2", "OUT_DIR": str(out),
                   "RANKS_PER_NODE": "", "REGIME": regime}
            result = run(["bash", str(self.root / f"scaling/launch_remesh_{launcher}.sh"), "--dry-run"], env=env)
            self.assertEqual(result.returncode, 0, result.stderr)
            with (out / "points.tsv").open() as f:
                points = [p for p in csv.DictReader(f, delimiter="\t") if p["status"] == "dry-run"]
            self.assertEqual([int(p["np"]) for p in points], [rpn, 2 * rpn], (launcher, regime))
            self.assertIn(f"--regime {regime}", (out / "sweep.txt").read_text())
            self.assertIn(f"job-name=mftl-remesh_{regime}_{launcher}-np{rpn}", result.stdout)
            self.assertIn(f"MODE=remesh_{launcher}", result.stdout)
            for f in (f"remesh_{launcher}_scaling", "remesh_common.h", "collect_remesh.sh", f"remesh_{launcher}_scaling.c"):
                self.assertTrue((out / "snapshot" / f).exists(), f)
        # An explicit density still wins over the spanning default.
        out = self.root / "weak-spanning-36"
        env = {**self.env, "RANKS": "", "NODES": "1", "MAX_NODES": "1", "OUT_DIR": str(out),
               "RANKS_PER_NODE": "24", "REGIME": "spanning"}
        result = run(["bash", str(self.root / "scaling/launch_remesh_weak.sh"), "--dry-run"], env=env)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--ntasks=24", result.stdout)
        result = run(["bash", str(self.root / "scaling/launch_remesh_strong.sh"), "--dry-run"],
                     env={**self.env, "REGIME": "both", "OUT_DIR": str(self.root / "bad")})
        self.assertNotEqual(result.returncode, 0)

    def test_remesh_collector_compares_modes_and_digests(self):
        sweep = self.root / "sweep"
        reports = sweep / "reports"
        reports.mkdir(parents=True)
        base = ("mode=remesh_strong regime={r} spanning={m} np={p} nodes=1 rpn={p} cuts=1x1x1 bodies=8 refine=3 "
                "R_over_h=0.5 tris=1000 tris_rank=500 steps=10 ms_step={t} remesh=1 migrate=0.1 sweep=0.5 "
                "reference=0.1 projection=0.1 ids=0.1 phases=1.5 migrations=2.5 frozen_pct=4 imbalance=1 "
                "dv_pct=-0.01 dv_worst_pct=0.02 dist_rms_pct=0.001 dist_max_pct=0.01 qmin=0.9 qmean=0.99 "
                "oob_pct=0 not_converged=0 missed=0 digest={d} sizing=uniform proj_mode=phong ok={ok}")
        for p, t in ((1, 8.0), (2, 4.0), (4, 2.5)):
            (reports / f"np{p:05d}.txt").write_text(
                "@summary " + base.format(r="spanning", m="once", p=p, t=t, d="aa", ok=1) + "\n" +
                "@summary " + base.format(r="spanning", m="nested", p=p, t=1.5 * t, d="bb", ok=1) + "\n")
        (reports / "np00008.txt").write_text(
            "@summary " + base.format(r="spanning", m="once", p=8, t=1, d="aa", ok=1) + "\n" +
            "@summary " + base.format(r="spanning", m="nested", p=8, t=1, d="bb", ok=0) + "\n")
        result = run(["bash", str(self.root / "scaling/collect.sh"), str(sweep)])
        self.assertEqual(result.returncode, 0, result.stderr)
        summary = (sweep / "summary.txt").read_text()
        self.assertIn("== once ==", summary)
        self.assertIn("== nested ==", summary)
        self.assertIn("once against nested", summary)
        self.assertIn("EXCLUDED FROM THE TABLES (failed or incomplete reports): np00008", summary)
        ratio = [l.split() for l in summary.split("once against nested")[1].splitlines() if l.strip()[:1].isdigit()]
        self.assertEqual([r[0] for r in ratio], ["1", "2", "4"])
        self.assertTrue(all(abs(float(r[3]) - 1.5) < 1e-9 for r in ratio))
        speedup = [l.split() for l in summary.split("== once ==")[1].split("accuracy")[0].splitlines() if l.strip()[:1].isdigit()]
        self.assertEqual([r[6] for r in speedup], ["1.00", "2.00", "3.20"])
        with (sweep / "samples.tsv").open() as f:
            self.assertEqual(len(list(csv.DictReader(f, delimiter="\t"))), 6)
        # The dispersed digest check, both verdicts.
        for digests, verdict in ((("ab", "ab"), "IDENTICAL at every P"), (("ab", "cd"), "DIFFERS between rank counts")):
            sweep = self.root / f"dispersed-{digests[1]}"
            (sweep / "reports").mkdir(parents=True)
            for p, d in zip((1, 2), digests):
                (sweep / "reports" / f"np{p:05d}.txt").write_text(
                    "@summary " + base.format(r="dispersed", m="dispersed", p=p, t=2.0 / p, d=d, ok=1) + "\n")
            result = run(["bash", str(self.root / "scaling/collect_remesh.sh"), str(sweep)])
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn(f"DISPERSED DIGEST {verdict}", (sweep / "summary.txt").read_text())

    def test_job_rejects_a_report_with_one_failed_mode(self):
        sweep = self.root / "sweep"
        snapshot = sweep / "snapshot"
        snapshot.mkdir(parents=True)
        (snapshot / "site.sh").write_text("site_modules() { :; }\nsite_runtime_env() { :; }\n"
                                         "LAUNCHER=(bash -c 'shift; exec \"$@\"' --)\n")
        env = {**self.env, "MODE": "remesh_strong", "NRANKS": "1", "MFTL_ROOT": str(self.root)}
        for lines, rc in ((["ok=1", "ok=0"], 2), (["ok=1", "ok=1"], 0)):
            body = "\\n".join(f"@summary np=1 ms_step=1 {x}" for x in lines)
            executable(snapshot / "remesh_strong_scaling", f'''#!/bin/bash
while [ "$#" -gt 0 ]; do
  if [ "$1" = --report ]; then printf '{body}\\n' >"$2"; break; fi
  shift
done
''')
            result = run(["bash", str(self.root / "scaling/job_scaling.sh")], env=env)
            self.assertEqual(result.returncode, rc, result.stdout + result.stderr)


@unittest.skipUnless(RUN_MPI, "pass --mpi to run MPI driver checks")
class Drivers(unittest.TestCase):
    def test_mpi_drivers(self):
        with tempfile.TemporaryDirectory(prefix="mftl-scaling-mpi-") as folder:
            folder = Path(folder)
            for mode, count in (("strong", ["--spheres", "64"]),
                                ("weak", ["--spheres-per-rank", "8", "--box", "2"])):
                for p in (1, 2, 4, 8):
                    report = folder / f"{mode}-{p}.txt"
                    cmd = ["mpirun", "-n", str(p), str(ROOT / "build" / f"{mode}_scaling"),
                           *count, "--refine", "2", "--steps", "40", "--cycles", "2",
                           "--warmup", "5", "--repeat", "2", "--init-batch", "7", "--verify-init",
                           "--max-seconds", "0", "--report", str(report)]
                    result = run(cmd)
                    self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                    self.assertIn("steps=80", report.read_text())
                    self.assertIn(" ok=1", report.read_text())
                    self.assertEqual(report.read_text().count("@repeat"), 2)

    def test_batch_sizes_preserve_initial_mesh(self):
        with tempfile.TemporaryDirectory(prefix="mftl-scaling-batches-") as folder:
            for mode, count in (("strong", ["--spheres", "125"]),
                                ("weak", ["--spheres-per-rank", "8", "--box", "2"])):
                signatures = []
                for batch in (1, 7, 100, 10000):
                    report = Path(folder) / f"{mode}-{batch}.txt"
                    result = run(["mpirun", "-n", "4", str(ROOT / "build" / f"{mode}_scaling"),
                                  *count, "--refine", "1", "--motion", "drift", "--jitter", "0.3",
                                  "--steps", "12", "--repeat", "1", "--init-batch", str(batch),
                                  "--verify-init", "--report", str(report)])
                    self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                    fields = dict(w.split("=", 1) for w in next(l for l in report.read_text().splitlines()
                                                               if l.startswith("@summary")).split()[1:])
                    signatures.append(fields["init_signature"])
                    self.assertEqual(int(fields["init_batches"]), (int(fields["bodies"])+batch-1)//batch)
                    self.assertLess(float(fields["init_batch_pool_mb"]), 0.03*min(batch, int(fields["bodies"])))
                self.assertEqual(len(set(signatures)), 1)

    def test_invalid_cli_and_report_path(self):
        for args in (["--steps", "abc"], ["--phi", "nan"], ["--box", "inf"],
                     ["--init-batch", "0"], ["--max-seconds", "-1"],
                     ["--box", "1e200"], ["--box", "1e-200"],
                     ["--cuts", "2147483647", "2147483647", "2147483647"],
                     ["--refine", "8", "--spheres-per-rank", "2000"],
                     ["--cycles", "2147483647"],
                     ["--report", "/nonexistent-mftl-directory/report.txt"]):
            result = run(["mpirun", "-n", "2", str(ROOT / "build/weak_scaling"), *args])
            self.assertNotEqual(result.returncode, 0, args)

    def test_exhausted_budget_does_not_report_a_timing(self):
        with tempfile.TemporaryDirectory(prefix="mftl-scaling-budget-") as folder:
            report = Path(folder) / "report.txt"
            result = run(["mpirun", "-n", "2", str(ROOT / "build/strong_scaling"),
                          "--spheres", "8", "--refine", "0", "--max-seconds", "1", "--report", str(report)])
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("setup exhausted the soft run budget", result.stderr)
            self.assertNotIn("@summary", report.read_text())

    def test_non_neighbour_motion_is_flagged(self):
        with tempfile.TemporaryDirectory(prefix="mftl-scaling-route-") as folder:
            report = Path(folder) / "report.txt"
            result = run(["mpirun", "-n", "8", str(ROOT / "build/strong_scaling"),
                          "--spheres", "512", "--refine", "0", "--cuts", "8", "1", "1",
                          "--steps", "1", "--warmup", "0", "--repeat", "2", "--report", str(report)])
            self.assertNotEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertIn("global route           FAILED", report.read_text())
            self.assertIn(" ok=0", report.read_text())

    def test_remesh_drivers(self):
        """Dispersed: one digest at every P. Spanning: both modes in one job. Weak spanning: grid rule."""
        with tempfile.TemporaryDirectory(prefix="mftl-scaling-remesh-") as folder:
            folder = Path(folder)
            digests = []
            for p in (1, 2, 4):
                report = folder / f"dispersed-{p}.txt"
                result = run(["mpirun", "-n", str(p), str(ROOT / "build/remesh_strong_scaling"),
                              "--regime", "dispersed", "--per-axis", "2", "--refine", "2", "--steps", "10",
                              "--trajectories", "1", "--warmup", "0", "--report", str(report)])
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                line = next(l for l in report.read_text().splitlines() if l.startswith("@summary"))
                fields = dict(w.split("=", 1) for w in line.split()[1:])
                self.assertEqual(fields["ok"], "1")
                self.assertEqual(fields["euler"], "16")
                digests.append(fields["digest"])
            self.assertEqual(len(set(digests)), 1, digests)

            report = folder / "spanning.txt"
            result = run(["mpirun", "-n", "2", str(ROOT / "build/remesh_strong_scaling"),
                          "--regime", "spanning", "--refine", "2", "--steps", "10", "--trajectories", "1",
                          "--warmup", "0", "--report", str(report)])
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            modes = [dict(w.split("=", 1) for w in l.split()[1:])["spanning"]
                     for l in report.read_text().splitlines() if l.startswith("@summary")]
            self.assertEqual(modes, ["once", "nested"])

            result = run(["mpirun", "-n", "2", str(ROOT / "build/remesh_weak_scaling"),
                          "--regime", "spanning", "--refine", "1", "--steps", "4"])
            self.assertEqual(result.returncode, 2)
            self.assertIn("divisible by --boxes-per-body 2", result.stderr)
            for args in (["--regime", "both"], ["--radius", "0.6"], ["--steps", "1"],
                         ["--spanning-modes", "twice"], ["--sizing", "anisotropic"]):
                result = run(["mpirun", "-n", "1", str(ROOT / "build/remesh_strong_scaling"), *args])
                self.assertNotEqual(result.returncode, 0, args)


if __name__ == "__main__":
    unittest.main()
