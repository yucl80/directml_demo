
import onnxruntime as ort
from transformers import AutoTokenizer
import numpy as np

print(ort.get_device())
print(ort.get_available_providers())

tokenizer = AutoTokenizer.from_pretrained("D:\\llm\\bge-m3")
ort_session = ort.InferenceSession(
    path_or_bytes="D:\\llm\\bge-m3\\model.onnx",providers=ort.get_available_providers()
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

import os

os.environ["TF_ENABLE_ONEDNN_OPTS"]="0"

from optimum.onnxruntime import ORTModelForCustomTasks



local_onnx_model = ORTModelForCustomTasks.from_pretrained(
    model_id="D:\\llm\\bge-m3", provider="CPUExecutionProvider"
)


start_time = time.time()
for i in range(1, 200):
    inputs = tokenizer(
        [
            "BGE M3 is an embedding model supporting dense retrieval",
            "lexical matching and multi-vector interaction.",
        ],
        padding="longest",
        return_tensors="np",
    )
    inputs = {key: value.astype(np.int64) for key, value in inputs.items()}
    outputs = local_onnx_model.forward(**inputs)
end_time = time.time()
print("Time taken: ", end_time - start_time)
print(outputs)
