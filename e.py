import psutil
import onnxruntime
import numpy
import os
import time


device_name = "CPU-DML"
max_seq_length = 128
doc_stride = 128
max_query_length = 64

sess_options = onnxruntime.SessionOptions()

# Optional: store the optimized graph and view it using Netron to verify that model is fully optimized.
# Note that this will increase session creation time so enable it for debugging only.
sess_options.optimized_model_filepath = os.path.join(
    "D:\\llm\\tmp", "optimized_model_{}.onnx".format(device_name)
)

# Please change the value according to best setting in Performance Test Tool result.
sess_options.intra_op_num_threads = psutil.cpu_count(logical=True)

session = onnxruntime.InferenceSession(
    "D:\\llm\\bge-m3",
    sess_options,
    providers=["DmlExecutionProvider", "CPUExecutionProvider"],
)
total_samples = 1000
dataset = [1, 2, 3]
latency = []
for i in range(total_samples):
    data = dataset[i]
    ort_inputs = {
        "input_ids": data[0].cpu().reshape(1, max_seq_length).numpy(),
        "input_mask": data[1].cpu().reshape(1, max_seq_length).numpy(),
        "segment_ids": data[2].cpu().reshape(1, max_seq_length).numpy(),
    }
    start = time.time()
    ort_outputs = session.run(None, ort_inputs)
    latency.append(time.time() - start)

print(
    "OnnxRuntime {} Inference time = {} ms".format(
        device_name, format(sum(latency) * 1000 / len(latency), ".2f")
    )
)
