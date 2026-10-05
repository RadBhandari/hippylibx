# --------------------------------------------------------------------------bc-
# Copyright (C) 2024 The University of Texas at Austin
#
# This file is part of the hIPPYlibx library. For more information and source
# code availability see https://hippylib.github.io.
#
# SPDX-License-Identifier: GPL-2.0-only
# --------------------------------------------------------------------------ec-


from pathlib import Path
import csv
from dolfinx.io import XDMFFile
import dolfinx as dlx
import numpy as np
from dolfinx.io import VTXWriter
import adios4dolfinx as a4d

class Callback:
    def __init__(
        self,
        run_dir,
        filename,
        comm,
        save_m=False,
        m_filename=None,
    ):
        self.comm = comm
        self.rank = comm.rank

        self.path = Path(run_dir) / filename

        self.save_m = save_m

        if m_filename is None:
            stem = self.path.stem
            self.m_path = self.path.parent / f"{stem}_m_history.xdmf"
        else:
            self.m_path = Path(run_dir) / m_filename

        self.history = []

        self.file = None
        self.writer = None
        self.fieldnames = None

        self.m_mesh_written = False
        self.m_iter = 0

        if self.rank == 0:
            self.path.parent.mkdir(parents=True, exist_ok=True)

        # Make sure directory exists before all ranks try to open XDMF
        self.comm.Barrier()

        if self.rank == 0:
            self.file = open(self.path, "w", newline="")

        self.m_vtx = None  # init lazily on first call
        self.m_output = None

        self.checkpoint_path = self.path.parent / f"{self.path.stem}_checkpoint.bp"
        self._mesh_written = False

        self._checkpoint_iter = 0

    def __call__(self, info):
        entry = dict(info)

        # m should be a dolfinx.fem.Function
        m = entry.pop("m", None)

        self.history.append(entry)

        # -------------------------
        # 1. Write scalar history (rank 0 only )
        # -------------------------
        if self.rank == 0:
            if self.writer is None:
                self.fieldnames = list(entry.keys())
                self.writer = csv.DictWriter(
                    self.file,
                    fieldnames=self.fieldnames,
                )
                self.writer.writeheader()

            self.writer.writerow(entry)
            self.file.flush()

        # -------------------------
        # 2. Write m checkpoint via adios4dolfinx (ALL ranks -- collective)
        # -------------------------
        if m is not None:
            if not self._mesh_written:
                a4d.write_mesh(self.checkpoint_path, m.function_space.mesh)
                self._mesh_written = True

            a4d.write_function(
                self.checkpoint_path,
                m,
                time=float(self._checkpoint_iter),
                name="m_map",
            )
            self._checkpoint_iter += 1

        # -------------------------
        # 3. Write m to VTX/.bp for visualization (ALL ranks -- collective)
        # -------------------------
        if self.save_m and m is not None:

            # The VTXWriter must remain attached to one persistent Function.
            if self.m_output is None:
                self.m_output = dlx.fem.Function(
                    m.function_space,
                    name=m.name,
                )

                self.m_vtx = VTXWriter(
                    self.comm,
                    str(self.m_path.with_suffix(".bp")),
                    [self.m_output],
                )

            # Copy the current callback value into the Function
            # registered with the writer.
            self.m_output.x.array[:] = m.x.array
            self.m_output.x.scatter_forward()

            # Write the updated persistent Function.
            self.m_vtx.write(float(self.m_iter))
            self.m_iter += 1

    def close(self):
        if self.rank == 0 and self.file is not None:
            self.file.close()
            self.file = None

        if self.m_vtx is not None:
            self.m_vtx.close()