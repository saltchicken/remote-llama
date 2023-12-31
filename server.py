import asyncio, logging, queue, threading, configparser

import grpc
from proto.proto_pb2 import LlamaReply
from proto.proto_pb2 import LlamaRequest
from proto.proto_pb2_grpc import LlamaCallbackServicer
from proto.proto_pb2_grpc import add_LlamaCallbackServicer_to_server

from langchain.llms import LlamaCpp
from langchain import PromptTemplate, LLMChain
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.base import BaseCallbackHandler
from langchain.memory import ConversationBufferMemory
from langchain.schema.output import LLMResult
from typing import Any


class MyCustomHandler(BaseCallbackHandler):
    def on_llm_new_token(self, token: str, **kwargs) -> None:
        q.put(token)
    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> Any:
        q.put('[[[END]]]')

class Llama():
    def __init__(self):

## Llama 2 template
#         self.template = """<s>[INST] <<SYS>>
# You are a helpful, respectful and honest assistant. Always answer as helpfully as possible. Answer as breifly as possible.

# If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.

# Here is a history of our conversation:
# {chat_history}
# <</SYS>>

# {question} [/INST]
# """

## Alpaca Convention template
        self.template = """
### Instruction: Here is a history of our conversation: {chat_history}
        
### Input: {question}
        
### Response:"""
        self.prompt = PromptTemplate(template=self.template, input_variables=["chat_history", "question"])
        callback_manager = CallbackManager([MyCustomHandler()])
        self.memory = ConversationBufferMemory(memory_key="chat_history")
        n_gpu_layers = 100  # Change this value based on your model and your GPU VRAM pool.
        n_batch = 512  # Should be between 1 and n_ctx, consider the amount of VRAM in your GPU.

        self.llm = LlamaCpp(
            model_path=config["DEFAULT"]["ModelPath"],
            n_gpu_layers=n_gpu_layers,
            n_batch=n_batch,
            input={"temperature": 0.75, "max_length": 2000, "top_p": 1},
            n_ctx=2048,
            callback_manager=callback_manager,
            verbose=True,
        )
        self.chain = LLMChain(llm=self.llm, prompt=self.prompt, memory=self.memory)
        
    def run(self, request):
        response = self.chain.run(request)
        return response
        
config = configparser.ConfigParser()
config.read("default.ini")
q = queue.Queue()
llama = Llama()

def producer(request):
    llama.run(request)

class LlamaCallback(LlamaCallbackServicer):
    async def llamaAsk(self, request: LlamaRequest, context: grpc.aio.ServicerContext) -> LlamaReply:
        logging.info("Serving llama request %s", request)
        producer_thread = threading.Thread(target=producer, args=(request.prompt,))
        producer_thread.start()
        while True:
            item = q.get()
            if item == '[[[END]]]':
                break
            else:
                yield LlamaReply(answer=f"{item}")

async def serve() -> None:
    server = grpc.aio.server()
    add_LlamaCallbackServicer_to_server(LlamaCallback(), server)
    listen_addr = "[::]:50051"
    server.add_insecure_port(listen_addr)
    logging.info("Starting server on %s", listen_addr)
    await server.start()
    await server.wait_for_termination()

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(serve())
    
    
    
    
