#!/usr/bin/env python3
"""Generate tiny deterministic ONNX models for testing.

Requires: pip install onnx numpy
No runtime dependencies (no onnxruntime needed to generate).

Models generated:
  - add_floats.onnx: f(A[2,3], B[2,3]) -> C[2,3] = A + B  (float32)
  - identity_float.onnx: f(X[1,4]) -> Y[1,4] = X  (float32)
  - dynamic_batch.onnx: f(X[N,3]) -> Y[N,3] = X  (float32, dynamic first dim)
  - int64_add.onnx: f(A[2], B[2]) -> C[2] = A + B  (int64)
"""

from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper


OUTPUT_DIR = Path(__file__).resolve().parent


def make_add_floats():
    A = helper.make_tensor_value_info("A", TensorProto.FLOAT, [2, 3])
    B = helper.make_tensor_value_info("B", TensorProto.FLOAT, [2, 3])
    C = helper.make_tensor_value_info("C", TensorProto.FLOAT, [2, 3])

    node = helper.make_node("Add", inputs=["A", "B"], outputs=["C"])
    graph = helper.make_graph([node], "add_graph", [A, B], [C])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.producer_name = "gonx-test-gen"
    onnx.checker.check_model(model)
    onnx.save(model, OUTPUT_DIR / "add_floats.onnx")
    print("  Created add_floats.onnx")


def make_identity_float():
    X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 4])
    Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, 4])

    node = helper.make_node("Identity", inputs=["X"], outputs=["Y"])
    graph = helper.make_graph([node], "identity_graph", [X], [Y])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.producer_name = "gonx-test-gen"
    onnx.checker.check_model(model)
    onnx.save(model, OUTPUT_DIR / "identity_float.onnx")
    print("  Created identity_float.onnx")


def make_dynamic_batch():
    X = helper.make_tensor_value_info("X", TensorProto.FLOAT, ["N", 3])
    Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, ["N", 3])

    node = helper.make_node("Identity", inputs=["X"], outputs=["Y"])
    graph = helper.make_graph([node], "dynamic_graph", [X], [Y])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.producer_name = "gonx-test-gen"
    onnx.checker.check_model(model)
    onnx.save(model, OUTPUT_DIR / "dynamic_batch.onnx")
    print("  Created dynamic_batch.onnx")


def make_int64_add():
    A = helper.make_tensor_value_info("A", TensorProto.INT64, [2])
    B = helper.make_tensor_value_info("B", TensorProto.INT64, [2])
    C = helper.make_tensor_value_info("C", TensorProto.INT64, [2])

    node = helper.make_node("Add", inputs=["A", "B"], outputs=["C"])
    graph = helper.make_graph([node], "int64_add_graph", [A, B], [C])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.producer_name = "gonx-test-gen"
    onnx.checker.check_model(model)
    onnx.save(model, OUTPUT_DIR / "int64_add.onnx")
    print("  Created int64_add.onnx")


if __name__ == "__main__":
    print("Generating test ONNX models...")
    make_add_floats()
    make_identity_float()
    make_dynamic_batch()
    make_int64_add()
    print("Done.")
