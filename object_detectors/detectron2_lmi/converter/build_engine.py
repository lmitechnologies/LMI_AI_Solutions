#
# SPDX-FileCopyrightText: Copyright (c) 1993-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import os
import sys
import logging
import argparse

import numpy as np
import tensorrt as trt
from cuda import cudart


logging.basicConfig(level=logging.INFO)
logging.getLogger("EngineBuilder").setLevel(logging.INFO)
log = logging.getLogger("EngineBuilder")

sys.path.insert(1, os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir))
import detectron2_lmi.utils.common_runtime as common



class EngineBuilder:
    """
    Parses an ONNX graph and builds a TensorRT engine from it.
    """

    def __init__(self, verbose=False, workspace=8):
        """
        :param verbose: If enabled, a higher verbosity level will be set on the TensorRT logger.
        :param workspace: Max memory workspace to allow, in Gb.
        """
        self.trt_logger = trt.Logger(trt.Logger.INFO)
        if verbose:
            self.trt_logger.min_severity = trt.Logger.Severity.VERBOSE

        trt.init_libnvinfer_plugins(self.trt_logger, namespace="")

        self.builder = trt.Builder(self.trt_logger)
        self.config = self.builder.create_builder_config()
        self.config.set_memory_pool_limit(
            trt.MemoryPoolType.WORKSPACE, workspace * (2**30)
        )

        self.batch_size = None
        self.network = None
        self.parser = None

    def create_network(self, onnx_path):
        """
        Parse the ONNX graph and create the corresponding TensorRT network definition.
        :param onnx_path: The path to the ONNX graph to load.
        """
        self.network = self.builder.create_network(0)
        self.parser = trt.OnnxParser(self.network, self.trt_logger)

        onnx_path = os.path.realpath(onnx_path)
        with open(onnx_path, "rb") as f:
            if not self.parser.parse(f.read()):
                log.error("Failed to load ONNX file: {}".format(onnx_path))
                for error in range(self.parser.num_errors):
                    log.error(self.parser.get_error(error))
                sys.exit(1)

        inputs = [self.network.get_input(i) for i in range(self.network.num_inputs)]
        outputs = [self.network.get_output(i) for i in range(self.network.num_outputs)]

        log.info("Network Description")
        for input in inputs:
            self.batch_size = input.shape[0]
            log.info(
                "Input '{}' with shape {} and dtype {}".format(
                    input.name, input.shape, input.dtype
                )
            )
        for output in outputs:
            log.info(
                "Output '{}' with shape {} and dtype {}".format(
                    output.name, output.shape, output.dtype
                )
            )
        assert self.batch_size > 0

    def create_engine(
        self,
        engine_path,
        precision,
        config_file,
        calib_input=None,
        calib_cache=None,
        calib_num_images=5000,
        calib_batch_size=8,
    ):
        """
        Build the TensorRT engine and serialize it to disk.
        :param engine_path: The path where to serialize the engine to.
        :param precision: The datatype to use for the engine, either 'fp32', 'fp16', 'int8'.
        :param calib_input: The path to a directory holding the calibration images.
        :param calib_cache: The path where to write the calibration cache to, or if it already exists, load it from.
        :param calib_num_images: The maximum number of images to use for calibration.
        :param calib_batch_size: The batch size to use for the calibration process.
        """
        engine_path = os.path.realpath(engine_path)
        engine_dir = os.path.dirname(engine_path)
        os.makedirs(engine_dir, exist_ok=True)
        log.info("Building {} Engine in {}".format(precision, engine_path))

        inputs = [self.network.get_input(i) for i in range(self.network.num_inputs)]

        if precision in ["fp16", "int8"]:
            if not self.builder.platform_has_fast_fp16:
                log.warning("FP16 is not supported natively on this platform/device")
            self.config.set_flag(trt.BuilderFlag.FP16)

        engine_bytes = self.builder.build_serialized_network(self.network, self.config)
        if engine_bytes is None:
            log.error("Failed to create engine")
            sys.exit(1)

        with open(engine_path, "wb") as f:
            log.info("Serializing engine to file: {:}".format(engine_path))
            f.write(engine_bytes)


def main(args):
    builder = EngineBuilder(args.verbose, args.workspace)
    builder.create_network(args.onnx)
    builder.create_engine(
        args.engine,
        args.precision,
        args.det2_config,
        args.calib_input,
        args.calib_cache,
        args.calib_num_images,
        args.calib_batch_size,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--onnx", help="The input ONNX model file to load")
    parser.add_argument("-e", "--engine", help="The output path for the TRT engine")
    parser.add_argument(
        "-c",
        "--det2_config",
        default=None,
        help="The Detectron 2 config file (.yaml) for the model",
        type=str,
    )
    parser.add_argument(
        "-p",
        "--precision",
        default="fp16",
        choices=["fp32", "fp16", "int8"],
        help="The precision mode to build in, either fp32/fp16/int8, default: 'fp16'",
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Enable more verbose log output"
    )
    parser.add_argument(
        "-w",
        "--workspace",
        default=1,
        type=int,
        help="The max memory workspace size to allow in Gb, " "default: 1",
    )
    parser.add_argument(
        "--calib_input", help="The directory holding images to use for calibration"
    )
    parser.add_argument(
        "--calib_cache",
        default="./calibration.cache",
        help="The file path for INT8 calibration cache to use, default: ./calibration.cache",
    )
    parser.add_argument(
        "--calib_num_images",
        default=5000,
        type=int,
        help="The maximum number of images to use for calibration, default: 5000",
    )
    parser.add_argument(
        "--calib_batch_size",
        default=8,
        type=int,
        help="The batch size for the calibration process, default: 8",
    )
    args = parser.parse_args()
    if not all([args.onnx, args.engine]):
        parser.print_help()
        log.error("These arguments are required: --onnx and --engine")
        sys.exit(1)
    if args.precision in ["int8"] and not (
        args.calib_input or os.path.exists(args.calib_cache)
    ):
        parser.print_help()
        log.error(
            "When building in int8 precision, --calib_input or an existing --calib_cache file is required"
        )
        sys.exit(1)
    main(args)