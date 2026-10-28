# --------------------------------------------------------------------------bc-
# Copyright (C) 2024 The University of Texas at Austin
#
# This file is part of the hIPPYlibx library. For more information and source
# code availability see https://hippylib.github.io.
#
# SPDX-License-Identifier: GPL-2.0-only
# --------------------------------------------------------------------------ec-

import os

import dolfinx as dlx

from ..modeling.variables import STATE, PARAMETER, ADJOINT


class CheckpointCallback:
    """
    Concrete callback implementation for saving checkpoints during Newton iterations.

    Interface:
        callback(it, x, **kwargs)

    where:
        it   : current Newton iteration index
        x    : current iterate, e.g. [u, m, p]
        kwargs: optional diagnostics such as cost, misfit, reg, gradnorm, alpha, cg_it, etc.

    The callback converts vectors to dolfinx.Function objects internally if needed.
    """

    def __init__(self, model, output_dir, save_every=1):
        self.model = model
        self.output_dir = output_dir
        self.save_every = save_every
        os.makedirs(self.output_dir, exist_ok=True)

    def _as_function(self, vec, Vh):
        if vec is None:
            return None
        func = dlx.fem.Function(Vh)
        func.x.array[:] = vec.array
        func.x.scatter_forward()
        return func

    def __call__(self, it, x, **kwargs):
        if it % self.save_every != 0:
            return

        u = self._as_function(x[STATE], self.model.Vh_u)
        m = self._as_function(x[PARAMETER], self.model.prior.Vh)
        p = self._as_function(x[ADJOINT], self.model.Vh_p)

        checkpoint_file = os.path.join(self.output_dir, f"checkpoint_it{it}.h5")
        print(f"Saved checkpoint at iteration {it}: {checkpoint_file}")
