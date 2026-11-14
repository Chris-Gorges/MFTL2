"""Check destructive handoff boundaries and optionally exercise real MPI output.

python3 tools/media/test_duct_stream.py
python3 tools/media/test_duct_stream.py --binary BUILD/droplets_in_duct_flow
"""

import argparse
import os
from pathlib import Path
import re
import struct
import subprocess
import sys
import tempfile
import time
import unittest
import zlib

from duct_gif import atomic_text, finish_frame, mesh_files, stop_process, stream_frames, supervise

def png_chunk(kind, data):
    return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data))


PNG = (b"\x89PNG\r\n\x1a\n" + png_chunk(b"IHDR", struct.pack(">IIBBBBB", 1, 1, 8, 6, 0, 0, 0))
       + png_chunk(b"IDAT", zlib.compress(b"\0\0\0\0\0")) + png_chunk(b"IEND", b""))
BINARY = None


class HandoffTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.work = Path(self.tmp.name)
        (self.work / "mesh").mkdir()
        (self.work / "frames").mkdir()
        self.pvtu = self.work / "mesh/duct_0000.pvtu"
        self.pieces = [self.work / f"mesh/duct_0000_{r:05d}.vtu" for r in range(2)]
        for path in self.pieces:
            path.write_bytes(b"<VTKFile>complete</VTKFile>\n")
        self.pvtu.write_text('<VTKFile><PUnstructuredGrid>' + ''.join(
            f'<Piece Source="{p.name}"/>' for p in self.pieces) + '</PUnstructuredGrid></VTKFile>')
        self.job = {"out": str(self.work / "frames"),
                    "stream": {"steps": [0], "mesh_dir": str(self.work / "mesh"), "ranks": 2, "timeout": 0.1}}
        self.png = self.work / "frames/frame_0000.png"

    def test_incomplete_png_preserves_every_mesh_piece(self):
        self.png.write_bytes(PNG[:-8])
        with self.assertRaises((RuntimeError, OSError)):
            finish_frame(self.job, self.pvtu, self.png, 10)
        self.assertTrue(all(p.exists() for p in self.pieces + [self.pvtu]))
        self.assertFalse(self.pvtu.with_suffix(".done").exists())

    def test_deletion_is_confined_to_one_completed_frame(self):
        unrelated = self.work / "mesh/duct_9999_00000.vtu"
        unrelated.write_bytes(b"retain")
        self.png.write_bytes(PNG)
        finish_frame(self.job, self.pvtu, self.png, 10)
        self.assertTrue(self.png.exists())
        self.assertTrue(self.pvtu.with_suffix(".done").exists())
        self.assertTrue(unrelated.exists())
        self.assertFalse(any(p.exists() for p in self.pieces + [self.pvtu]))

    def test_complete_but_corrupt_png_preserves_mesh(self):
        corrupt = bytearray(PNG)
        corrupt[45] ^= 1
        self.png.write_bytes(corrupt)
        with self.assertRaisesRegex(RuntimeError, "Corrupt PNG"):
            finish_frame(self.job, self.pvtu, self.png, 10)
        self.assertTrue(all(p.exists() for p in self.pieces + [self.pvtu]))
        self.assertFalse(self.pvtu.with_suffix(".done").exists())

    def test_index_cannot_request_deletion_outside_frame(self):
        self.pvtu.write_text('<VTKFile><Piece Source="../simulation.log"/></VTKFile>')
        with self.assertRaises(RuntimeError):
            mesh_files(self.pvtu, 2)

    def test_index_alone_is_not_a_completion_signal(self):
        with self.assertRaisesRegex(RuntimeError, "Timed out"):
            next(stream_frames(self.job))
        atomic_text(self.pvtu.with_suffix(".ready"), "0 2\n")
        self.assertEqual(next(stream_frames(self.job)), str(self.pvtu))

    def test_marker_with_unfinished_piece_is_rejected(self):
        atomic_text(self.pvtu.with_suffix(".ready"), "0 2\n")
        self.pieces[1].write_bytes(b"still writing")
        with self.assertRaisesRegex(RuntimeError, "Incomplete VTU"):
            next(stream_frames(self.job))

    def test_failed_renderer_stops_simulation(self):
        sim = self.work / "simulate.py"
        sim.write_text("import os,time\nfrom pathlib import Path\n"
                       "Path('../sim.pid').write_text(str(os.getpid()))\ntime.sleep(30)\n")
        render = self.work / "render.py"
        render.write_text("import time,sys\nfrom pathlib import Path\n"
                          f"root=Path({str(self.work)!r})\n(root/'renderer.ready').touch()\n"
                          "while not (root/'sim.pid').exists(): time.sleep(.02)\nsys.exit(7)\n")
        with self.assertRaisesRegex(RuntimeError, "Renderer failed"):
            supervise([sys.executable, str(sim)], [sys.executable, str(render)], self.work, 5, 1, 1)
        pid = int((self.work / "sim.pid").read_text())
        with self.assertRaises(ProcessLookupError):
            os.kill(pid, 0)


class MPIHandoffTests(unittest.TestCase):
    def setUp(self):
        if BINARY is None:
            self.skipTest("pass --binary to exercise the actual two-rank writer")
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.work = Path(self.tmp.name)
        (self.work / "mesh").mkdir()
        (self.work / "frames").mkdir()
        self.command = [os.environ.get("MPIEXEC", "mpirun"), "-n", "2", BINARY,
                        "--drops", "40", "--refine", "3", "--steps", "4", "--time", str(4/240),
                        "--report", "2", "--verify"]

    def test_one_frame_bound_and_numerical_equivalence(self):
        reference = subprocess.run(self.command, cwd=self.work, capture_output=True, text=True, timeout=60, check=True)
        job = {"out": str(self.work / "frames"),
               "stream": {"steps": [0, 2, 4], "mesh_dir": str(self.work / "mesh"), "ranks": 2, "timeout": 15}}
        with (self.work / "stream.log").open("w") as log:
            proc = subprocess.Popen(self.command + ["--vtk", "2", "--vtk-stream", "--vtk-timeout", "15"],
                                    cwd=self.work / "mesh", stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            try:
                for i, pvtu in enumerate(stream_frames(job)):
                    time.sleep(0.3)  # deliberately slow consumer: the producer must not advance
                    self.assertIsNone(proc.poll())
                    self.assertEqual(len(list((self.work / "mesh").glob("*.pvtu"))), 1)
                    self.assertEqual(len(list((self.work / "mesh").glob("*.vtu"))), 2)
                    png = self.work / f"frames/frame_{i:04d}.png"
                    png.write_bytes(PNG)
                    finish_frame(job, pvtu, png, 0)
                self.assertEqual(proc.wait(timeout=15), 0)
            finally:
                stop_process(proc)
        streamed = (self.work / "stream.log").read_text()
        self.assertIn("verify: ok", streamed)
        pattern = r"^\s+(\d+)\s+([\d.]+)\s+(\d+)\s+([\d.]+)\s+[\d.]+$"
        self.assertEqual(re.findall(pattern, reference.stdout, re.M), re.findall(pattern, streamed, re.M))
        self.assertEqual(list((self.work / "mesh").iterdir()), [])

    def test_lost_consumer_times_out_and_preserves_frame(self):
        result = subprocess.run(self.command + ["--vtk", "2", "--vtk-stream", "--vtk-timeout", "0.3"],
                                cwd=self.work / "mesh", capture_output=True, text=True, timeout=20)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("did not acknowledge", result.stderr)
        self.assertEqual(len(list((self.work / "mesh").glob("*.pvtu"))), 1)
        self.assertEqual(len(list((self.work / "mesh").glob("*.vtu"))), 2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--binary", type=Path)
    args, rest = parser.parse_known_args()
    if args.binary:
        BINARY = str(args.binary.resolve())
    unittest.main(argv=[sys.argv[0]] + rest)
