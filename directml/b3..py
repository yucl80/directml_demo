import onnxruntime as ort
from transformers import AutoTokenizer
import numpy as np

print(ort.get_device())
print(ort.get_available_providers())

sess_opt = ort.SessionOptions()
sess_opt.intra_op_num_threads = 3
sess_opt.execution_mode = ort.ExecutionMode.ORT_PARALLEL
sess_opt.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
sess_opt.add_session_config_entry("session.intra_op.allow_spinning", "1")
sess_opt.add_session_config_entry("session.intra_op_thread_affinities", "1;2")

tokenizer = AutoTokenizer.from_pretrained("D:\\llm\\bge-m3")
ort_session = ort.InferenceSession(
    path_or_bytes="D:\\llm\\bge-m3\\model.onnx",
    providers=ort.get_available_providers(),
    sess_options=sess_opt,
)
import time

outputs = any

start_time = time.time()
for i in range(1, 200):
    inputs = tokenizer(
        "BGE M3 is an embedding model supporting dense retrieval, lexical matching and multi-vector interaction.",
        padding="longest",
        return_tensors="np",
    )
    inputs = {key: value.astype(np.int64) for key, value in inputs.items()}
    inputs_onnx = {k: ort.OrtValue.ortvalue_from_numpy(v) for k, v in inputs.items()}
    outputs = ort_session.run(None, inputs_onnx)
end_time = time.time()
print("Time taken: ", end_time - start_time)
print(outputs)

from optimum.onnxruntime import ORTModelForCustomTasks

sess_opt = ort.SessionOptions()
sess_opt.intra_op_num_threads = 12
sess_opt.execution_mode = ort.ExecutionMode.ORT_PARALLEL
sess_opt.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

ort_session = ort.InferenceSession(
    path_or_bytes="D:\\llm\\bge-m3\\model.onnx",
    providers=["CPUExecutionProvider"],
    sess_options=sess_opt,
)

start_time = time.time()
for i in range(1, 200):
    inputs = tokenizer(
        "BGE M3 is an embedding model supporting dense retrieval, lexical matching and multi-vector interaction.",
        padding="longest",
        return_tensors="np",
    )
    inputs = {key: value.astype(np.int64) for key, value in inputs.items()}
    inputs_onnx = {k: ort.OrtValue.ortvalue_from_numpy(v) for k, v in inputs.items()}
    outputs = ort_session.run(None, inputs_onnx)

end_time = time.time()
print("Time taken: ", end_time - start_time)
print(outputs)
