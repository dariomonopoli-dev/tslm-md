# SPDX-FileCopyrightText: 2025 Stanford University, ETH Zurich, and the project authors (see CONTRIBUTORS.md)
# SPDX-FileCopyrightText: 2025 This source file is part of the OpenTSLM open-source project.
#
# SPDX-License-Identifier: MIT

# Lazy export: importing `opentslm` (e.g. for a submodule) used to eagerly pull
# in OpenTSLM -> OpenTSLMFlamingo -> open_flamingo -> open_clip -> torchvision.
# That chain fails on environments with a torch/torchvision mismatch (e.g.
# SageMaker Studio). SP-only users never touch Flamingo, so defer the import
# until OpenTSLM is actually accessed.

__all__ = ["OpenTSLM"]


def __getattr__(name):
    if name == "OpenTSLM":
        from opentslm.model.llm.OpenTSLM import OpenTSLM
        return OpenTSLM
    raise AttributeError(f"module 'opentslm' has no attribute {name!r}")