import argparse
import json
import time
import logging
from threading import Thread

import uvicorn

from fastapi import FastAPI
from fastapi.responses import JSONResponse, StreamingResponse

import openvino as ov
import openvino_genai as ovgenai

from transformers import AutoTokenizer

from openai.types.chat.chat_completion_chunk import (
        ChatCompletionChunk,
        Choice,
        ChoiceDelta,
    )

from iterable_streamer import IterableStreamer


logger = logging.getLogger(__name__)


class OVServer:
    def __init__(
            self,
            model_path,
            port=8000,
            host="localhost",
            device="GPU",
            log_level="info",
        ):
        self.model_name_or_path = model_path
        self.port = port
        self.host = host
        self.log_level = log_level
        self.device = device
        scheduler_config = ovgenai.SchedulerConfig()
        scheduler_config.dynamic_split_fuse = False
        scheduler_config.max_num_batched_tokens = 4096
        # scheduler_config.num_kv_blocks = 4096 // 16
        self.model = ovgenai.LLMPipeline(
            self.model_name_or_path,
            self.device,
            scheduler_config=scheduler_config,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)

    @staticmethod
    def add_server_arguments(parser):
        parser.add_argument("--model_path", type=str, default=None, required=True, help="Path to the model directory or model name")
        parser.add_argument("--port", type=int, default=8000, help="Port to run the server on")
        parser.add_argument("--host", type=str, default="localhost", help="Host to run the server on")

    @staticmethod
    def get_clean_messages(messages):
        clean_messages = []
        for message in messages:
            parsed_message = {"role": message["role"], "content": []}
            if isinstance(message["content"], str):
                parsed_content = message["content"]
            elif isinstance(message["content"], list):
                parsed_content = []
                for content in message["content"]:
                    if content["type"] == "text":
                        parsed_content.append(content["text"])
                parsed_content = " ".join(parsed_content)
            parsed_message["content"] = parsed_content
            clean_messages.append(parsed_message)
        return clean_messages

    def run(self):
        app = FastAPI()

        @app.post("/v1/chat/completions")
        def chat_completions(request: dict):
            output = self.process_chat_completions(request)
            return StreamingResponse(output, media_type="text/event-stream")

        @app.options("/v1/models")
        @app.get("/v1/models")
        def get_all_models():
            return JSONResponse(content={"object": "list", "data": [self.model_name_or_path]})

        uvicorn.run(app, host=self.host, port=self.port, log_level=self.log_level)

    def process_chat_completions(self, request: dict):
        messages = request.get("messages", [])
        assert messages[-1]["role"] == "user", "The last message must be from the user."
        clean_messages = self.get_clean_messages(messages)

        inputs = ov.Tensor(self.tokenizer.apply_chat_template(
            clean_messages,
            add_generation_prompt=True,
            tools=request.get("tools"),
            tokenize=True,
            return_tensors='np',
            **request.get("chat_template_kwargs", {}),
        ))

        request_id = request.get("request_id", "req_0")

        generation_streamer = IterableStreamer(self.tokenizer)
        generation_config = self.create_generation_config_from_request(request, self.model.get_generation_config())

        def stream_chat_completion(streamer, _request_id):
            thread = Thread(target=self.model.generate, args=(inputs, generation_config, streamer))
            results = ""

            try:
                thread.start()
                yield self.build_chat_completion_chunk(_request_id, role="assistant", model=self.model_name_or_path)
                for result in streamer:
                    results += result
                    if result != "":
                        yield self.build_chat_completion_chunk(_request_id, content=result, model=self.model_name_or_path)
                # TODO add usage stats if requested
                yield self.build_chat_completion_chunk(_request_id, finish_reason="stop", model=self.model_name_or_path)

                thread.join()
            except Exception as e:
                logger.error(str(e))
                yield f'data: {{"error": "{str(e)}"}}'
        
        return stream_chat_completion(generation_streamer, request_id)

    def build_chat_completion_chunk(
            self,
            request_id="",
            content=None,
            model=None,
            role=None,
            finish_reason=None,
        ):
        chunk = ChatCompletionChunk(
            id=request_id,
            created=int(time.time()),
            model=model,
            choices=[
                Choice(
                    delta=ChoiceDelta(
                        role=role,
                        content=content
                    ),
                    index=0,
                    finish_reason=finish_reason,
                )
            ],
            system_fingerprint="",
            object="chat.completion.chunk",
        )
        return f"data: {chunk.model_dump_json(exclude_none=True)}\n\n"

    def create_generation_config_from_request(self, req, model_generation_config):
        if req.get("generation_config") is not None:
            generation_config = ovgenai.GenerationConfig(**json.loads(req["generation_config"]))
        else:
            generation_config = model_generation_config
        # Response-specific parameters
        if req.get("max_output_tokens") is not None:
            generation_config.max_new_tokens = int(req["max_output_tokens"])

        # Completion-specific parameters
        if req.get("max_tokens") is not None:
            generation_config.max_new_tokens = int(req["max_tokens"])
        if req.get("frequency_penalty") is not None:
            generation_config.repetition_penalty = float(req["frequency_penalty"])
        # if req.get("logit_bias") is not None:
        #     generation_config.sequence_bias = req["logit_bias"]
        if req.get("stop") is not None:
            generation_config.stop_strings = set(req["stop"])
        if req.get("temperature") is not None:
            generation_config.temperature = float(req["temperature"])
            if float(req["temperature"]) == 0.0:
                generation_config.do_sample = False
        if req.get("top_p") is not None:
            generation_config.top_p = float(req["top_p"])        
        return generation_config

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="OV Server")
    OVServer.add_server_arguments(parser)
    args = parser.parse_args()
    logging.basicConfig(
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
        level=logging.INFO
    )
    server = OVServer(
        **vars(args)
    )
    server.run()