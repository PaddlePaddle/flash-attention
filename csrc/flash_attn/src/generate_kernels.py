# Copied from Driss Guessous's PR in PyTorch: https://github.com/pytorch/pytorch/pull/105602

# This file is run to generate the kernel instantiations for the flash_attn kernels
# They are written to several files in order to speed up compilation

import argparse
import itertools
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

DTYPE_MAP = {
    "fp16": "cutlass::half_t",
    "bf16": "cutlass::bfloat16_t",
}

SM = [80]  # Sm80 kernels support up to
HEAD_DIMENSIONS = [32, 64, 96, 128, 160, 192, 224, 256]
IS_CAUSAL = ["false", "true"]
IS_DENSEMASK = ["false", "true"]
IS_FLASHMASK = ["false", "true"]
KERNEL_IMPL_TEMPLATE_FWD = """#include "flash_fwd_launch_template.h"

template<>
void run_mha_fwd_<{DTYPE}, {HEAD_DIM}, {IS_CAUSAL}, {IS_DENSEMASK}, {IS_FLASHMASK}>(Flash_fwd_params &params, cudaStream_t stream) {{
    run_mha_fwd_hdim{HEAD_DIM}<{DTYPE}, {IS_CAUSAL}, {IS_DENSEMASK}, {IS_FLASHMASK}>(params, stream);
}}
"""

KERNEL_IMPL_TEMPLATE_BWD = """#include "flash_bwd_launch_template.h"

template<>
void run_mha_bwd_<{DTYPE}, {HEAD_DIM}, {IS_CAUSAL}, {IS_DENSEMASK}, {IS_FLASHMASK}>(Flash_bwd_params &params, cudaStream_t stream, const bool configure) {{
    run_mha_bwd_hdim{HEAD_DIM}<{DTYPE}, {IS_CAUSAL}, {IS_DENSEMASK}, {IS_FLASHMASK}>(params, stream, configure);
}}
"""


@dataclass
class Kernel:
    sm: int
    dtype: str
    head_dim: int
    is_causal: bool
    is_densemask: bool
    is_flashmask: bool
    direction: str

    @property
    def template(self) -> str:
        if self.direction == "fwd":
            return KERNEL_IMPL_TEMPLATE_FWD.format(
                DTYPE=DTYPE_MAP[self.dtype], HEAD_DIM=self.head_dim, IS_CAUSAL=self.is_causal, IS_DENSEMASK=self.is_densemask, IS_FLASHMASK=self.is_flashmask
            )
        elif self.direction == "bwd":
            return KERNEL_IMPL_TEMPLATE_BWD.format(
                DTYPE=DTYPE_MAP[self.dtype], HEAD_DIM=self.head_dim, IS_CAUSAL=self.is_causal, IS_DENSEMASK=self.is_densemask, IS_FLASHMASK=self.is_flashmask
            )
        else:
            raise ValueError(f'direction: `{self.direction}` is not supported now!')

    @property
    def filename(self) -> str:
        switch_list = []
        if self.is_causal == 'true':
            switch_list.append('causal')
        if self.is_densemask == 'true':
            switch_list.append('densemask')
        if self.is_flashmask == 'true':
            switch_list.append('flashmask')
        switch_filename = '_'.join(switch_list)
        if len(switch_list) > 0:
            switch_filename = '_' + switch_filename
        
        return f"flash_{self.direction}_hdim{self.head_dim}_{self.dtype}{switch_filename}_sm{self.sm}.cu"


def get_all_kernels() -> List[Kernel]:
    for direction in ["fwd", "bwd"]:
        for dtype, head_dim, is_causal, is_densemask, is_flashmask, sm in itertools.product(DTYPE_MAP.keys(), HEAD_DIMENSIONS, IS_CAUSAL, IS_DENSEMASK, IS_FLASHMASK, SM):
            # if is_causal == 'true' and is_densemask == 'true':
            #     continue
            if is_densemask == 'true' and is_flashmask == 'true':
                continue
            yield Kernel(sm=sm, dtype=dtype, head_dim=head_dim, is_causal=is_causal, is_densemask=is_densemask, is_flashmask=is_flashmask, direction=direction)


def write_kernel(kernel: Kernel, autogen_dir: Path) -> None:
    prelude = """// Copyright (c) 2024, Tri Dao.
// Splitting the different head dimensions to different files to speed up compilation.
// This file is auto-generated. See "generate_kernels.py"\n
"""
    (autogen_dir / kernel.filename).write_text(prelude + kernel.template)


def main(output_dir: Optional[str]) -> None:
    if output_dir is None:
        output_dir = Path(__file__).parent
    else:
        output_dir = Path(output_dir)

    for kernel in get_all_kernels():
        write_kernel(kernel, output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="generate_kernels",
        description="Generate the flash_attention kernels template instantiations",
    )
    # Set an optional output directory
    parser.add_argument(
        "-o",
        "--output_dir",
        required=False,
        help="Where to generate the kernels "
        " will default to the current directory ",
    )
    args = parser.parse_args()
    main(args.output_dir)

