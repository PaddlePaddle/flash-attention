# Copied from Driss Guessous's PR in PyTorch: https://github.com/pytorch/pytorch/pull/105602

# This file is run to generate the kernel instantiations for the flash_attn kernels
# They are written to several files in order to speed up compilation

import argparse
import itertools
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

FWD_DTYPE_MAP = {
    "fp16": "cutlass::half_t",
    "bf16": "cutlass::bfloat16_t",
    "e4m3": "cutlass::float_e4m3_t",
}
BWD_DTYPE_MAP = {
    "fp16": "cutlass::half_t",
    "bf16": "cutlass::bfloat16_t",
}

SM = [90]  # Sm90 kernels support up to
FWD_HEAD_DIMENSIONS = [64, 128, 256]
FWD_GQA_HEADS = [1, 2, 4, 8, 16, 32]
BWD_HEAD_DIMENSIONS = [64, 96, 128]
BWD_GQA_HEADS = [1]

KERNEL_IMPL_TEMPLATE_FWD = """#include "flash_fwd_launch_template.h"

template<>
void run_mha_fwd{GQA}_<{DTYPE}, {HEAD_DIM}{GQA_HEAD}>(Flash_fwd_params &params, cudaStream_t stream) {{
    run_mha_fwd_hdim{HEAD_DIM}{FP8}{GQA}<{DTYPE}{GQA_HEAD}>(params, stream);
}}
"""

KERNEL_IMPL_TEMPLATE_BWD = """#include "flash_bwd_launch_template.h"

template<>
void run_mha_bwd_<{DTYPE}, {HEAD_DIM}>(Flash_bwd_params &params, cudaStream_t stream) {{
    run_mha_bwd_hdim{HEAD_DIM}<{DTYPE}>(params, stream);
}}
"""


@dataclass
class Kernel:
    sm: int
    dtype: str
    head_dim: int
    gqa_head: int
    direction: str

    @property
    def template(self) -> str:
        
        GQA_HEAD = ''
        GQA = ''
        if self.gqa_head > 1:
            GQA_HEAD = f', {self.gqa_head}'
            GQA = '_gqa'

        FP8 = ''
        if self.dtype == 'e4m3':
            FP8 = '_fp8'

        if self.direction == "fwd":
            return KERNEL_IMPL_TEMPLATE_FWD.format(
                DTYPE=FWD_DTYPE_MAP[self.dtype], HEAD_DIM=self.head_dim, GQA=GQA, GQA_HEAD=GQA_HEAD, FP8=FP8 
            )
        elif self.direction == "bwd":
            return KERNEL_IMPL_TEMPLATE_BWD.format(
                DTYPE=BWD_DTYPE_MAP[self.dtype], HEAD_DIM=self.head_dim 
            )
        else:
            raise ValueError(f'direction: `{self.direction}` is not supported now!')

    @property
    def filename(self) -> str:
        switch_list = []
        if self.gqa_head > 1:
            switch_list.append(f'gqa{self.gqa_head}')
        switch_filename = '_'.join(switch_list)
        if len(switch_list) > 0:
            switch_filename = '_' + switch_filename
        
        return f"flash_{self.direction}_hdim{self.head_dim}_{self.dtype}{switch_filename}_sm{self.sm}.cu"


def get_all_fwd_kernels() -> List[Kernel]:
    for dtype, head_dim, gqa_head, sm in itertools.product(FWD_DTYPE_MAP.keys(), FWD_HEAD_DIMENSIONS, FWD_GQA_HEADS, SM):
        yield Kernel(sm=sm, dtype=dtype, head_dim=head_dim, gqa_head=gqa_head, direction='fwd')

def get_all_bwd_kernels() -> List[Kernel]:
    for dtype, head_dim, gqa_head, sm in itertools.product(BWD_DTYPE_MAP.keys(), BWD_HEAD_DIMENSIONS, BWD_GQA_HEADS, SM):
        yield Kernel(sm=sm, dtype=dtype, head_dim=head_dim, gqa_head=gqa_head, direction='bwd')


def write_kernel(kernel: Kernel, autogen_dir: Path) -> None:
    prelude = """// Copyright (c) 2024, Jay Shah, Ganesh Bikshandi, Ying Zhang, Vijay Thakkar, Pradeep Ramani, Tri Dao.
// Splitting the different head dimensions to different files to speed up compilation.
// This file is auto-generated. See "generate_kernels.py"\n
"""
    (autogen_dir / kernel.filename).write_text(prelude + kernel.template)


def main(output_dir: Optional[str]) -> None:
    if output_dir is None:
        output_dir = Path(__file__).parent
    else:
        output_dir = Path(output_dir)

    for kernel in get_all_fwd_kernels():
        write_kernel(kernel, output_dir)

    for kernel in get_all_bwd_kernels():
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

