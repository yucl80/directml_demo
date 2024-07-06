from benchmark_helper import Precision
from transformers import AutoTokenizer, GPT2Config
import numpy
import time
import torch
import os
import onnxruntime
import pandas as pd
from gpt2_helper import Gpt2Helper 

tokenizer = AutoTokenizer.from_pretrained('gpt2-xl')
device='cuda'

class ONNX_Generative_Model():
    
    def __init__(self, onnx_model_path,
                 model_class_name,
                 config,
                 device='cuda',
                 is_float16=True, 
                ):
        self.onnx_model_path = onnx_model_path
        self.model_class_name = model_class_name
        self.config = config
        self.device = device
        self.float_type = torch.float16 if is_float16 else torch.float32
        self.is_float16 = is_float16
        self.batch_size = 1
        
        #shapes for IO binding
        self.empty_past_sequence_length=0
        self.empty_past_shape = [2, 1, self.config.num_attention_heads, self.empty_past_sequence_length, 
                            int(self.config.hidden_size / self.config.num_attention_heads)]
        self.max_output_shapes = Gpt2Helper.get_output_shapes(1, 1024, 1024, self.config, self.model_class_name)
        
        #CUDA session with ONNX model
        self.sess_options = onnxruntime.SessionOptions()
        self.session = onnxruntime.InferenceSession(self.onnx_model_path, self.sess_options, 
                            providers=['CUDAExecutionProvider' if device=='cuda' else 'CPUExecutionProvider'])

        
    
    
    
    def get_real_inputs(self, 
                        input_ids,
                        past,
                        sequence_length,
                        past_sequence_length):

        """ Create real inputs for GPT2 model.
        Returns torch tensors of input_ids, position_ids, 
        attention_mask and a list of past state tensors.
        """

        past_sequence_length = past[0].shape[3]
        total_sequence_length = input_ids.shape[1]
        
        print('past_sequence_length', past_sequence_length)
        print('total_sequence_length', total_sequence_length)
        
        attention_mask = torch.ones([self.batch_size, total_sequence_length], 
                                    dtype=self.float_type, device=self.device)

        # Deduce position_ids from attention mask
        position_ids = (attention_mask.long().cumsum(-1) - 1)
        position_ids.masked_fill_(position_ids < 0, 0)
#        position_ids = position_ids[:, past_sequence_length:]
        position_ids = position_ids[:, :]
        
        print('input_ids.shape', input_ids.shape)
        print('position_ids.shape', position_ids.shape)
        print('attention_mask.shape', attention_mask.shape)
        print('past[0].shape', past[0].shape)

        return input_ids, position_ids, attention_mask, past
    

    
    def __call__(self, input_ids, past=None):

        if past:
            past_sequence_length = past[0].shape[3]
            print('past_sequence_length', past_sequence_length)
            print('past_shape', past[0].shape)
        else:
            past_sequence_length = self.empty_past_sequence_length
            past = [torch.empty(self.empty_past_shape, dtype=self.float_type, 
                                device=self.device) for _ in range(self.config.n_layer)]

        sequence_length = len(input_ids[0])


        #put everything together for ONNX input
        onnx_inputs = self.get_real_inputs(input_ids, past, sequence_length, past_sequence_length)
        output_shapes = Gpt2Helper.get_output_shapes(1, past_sequence_length, sequence_length, 
                                                     self.config, self.model_class_name)
        output_buffers = Gpt2Helper.get_output_buffers(self.max_output_shapes, self.device, Precision.FLOAT16)



        ort_io_outputs = Gpt2Helper.onnxruntime_inference_with_binded_io(
                                self.session,
                                onnx_inputs,
                                output_buffers,
                                output_shapes,
                                total_runs=0,
                                return_numpy=False,
                                include_copy_output_latency=True)

        return ort_io_outputs
        

model = ONNX_Generative_Model('gpt2_xl_onnx_past/_past_fp16.onnx',
                             model_class_name='GPT2LMHeadModel',
                             config=GPT2Config.from_pretrained('gpt2-xl'),
                             device=device,
                             is_float16=True,)