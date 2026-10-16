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
                   "RANKS_PER_NODE": "", "SPHERES": "", "PER_RANK": "", "TIME_LIMIT": ""}
            result = run(["bash", str(self.root / f"scaling/launch_{mode}.sh"), "--dry-run"], env=env)
            self.assertEqual(result.returncode, 0, result.stderr)
            with (self.root / mode / "points.tsv").open() as f:
                points = list(csv.DictReader(f, delimiter="\t"))
            self.assertEqual([int(p["np"]) for p in points], [36, 72, 144, 288, 576, 1080])
            self.assertTrue(all(int(p["rpn"]) == 36 and int(p["np"]) == 36*int(p["nodes"]) for p in points))
            self.assertIn(size, (self.root / mode / "sweep.txt").read_text())
            self.assertEqual(result.stdout.count("--time=01:00:00"), 6)

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
                    {"MAX_NODES": "0"}, {"JOB_REPEATS": "oops"}):
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


if __name__ == "__main__":
    unittest.main()
