import argparse
import json
import time
import logging
import datetime
from threading import Thread

import uvicorn

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse

import openvino as ov
import openvino_genai as ovgenai

from transformers import AutoTokenizer

from openai.types.chat.chat_completion_chunk import (
        ChatCompletionChunk,
        Choice,
        ChoiceDelta,
        CompletionUsage
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
        logger.info(f"Initializing OV Server with model: {model_path}, device: {device}, host: {host}, port: {port}")
        
        self.model_path = model_path
        self.port = port
        self.host = host
        self.log_level = log_level
        self.device = device
        
        try:
            scheduler_config = ovgenai.SchedulerConfig()
            scheduler_config.dynamic_split_fuse = False
            scheduler_config.max_num_batched_tokens = 4096 * 4
            # scheduler_config.enable_prefix_caching = True
            # scheduler_config.num_kv_blocks = 4096 // 16
            logger.info(f"Loading LLM pipeline from {self.model_path} on {self.device}")
            self.model = ovgenai.LLMPipeline(
                self.model_path,
                self.device,
                scheduler_config=scheduler_config,
            )
            
            logger.info(f"Loading tokenizer from {model_path}")
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            
            logger.info("OV Server initialization completed successfully")
        except Exception as e:
            logger.error(f"Failed to initialize OV Server: {str(e)}")
            raise

    @staticmethod
    def add_server_arguments(parser):
        parser.add_argument("--model_path", type=str, default=None, required=True, help="Path to the model directory or model name")
        parser.add_argument("--port", type=int, default=8000, help="Port to run the server on")
        parser.add_argument("--host", type=str, default="localhost", help="Host to run the server on")
        parser.add_argument("--device", type=str, default="GPU", help="Device to run inference on (CPU, GPU)")
        parser.add_argument("--log_level", type=str, default="info", help="Logging level")

    @staticmethod
    def get_clean_messages(messages):
        logger.debug(f"Processing {len(messages)} messages for chat template")
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
        logger.debug("Message processing completed")
        return clean_messages
    
    def get_models(self):
        # Right now the server only handles a single model
        return [
            {
                "id": self.model_path,
                "object": "model",
                "created": datetime.datetime.now().timestamp(),
                "owned_by": "user",
            },
        ]

    def run(self):
        logger.info(f"Starting FastAPI server on {self.host}:{self.port}")
        app = FastAPI()

        @app.post("/v1/chat/completions")
        def chat_completions(request: dict):
            logger.info("Received chat completion request")
            try:
                # Server only handles streaming for now, reject queries that are not streaming
                if not request.get("stream"):
                    logger.error("Non-streaming requests are not supported")
                    raise HTTPException(status_code=422, detail="Non-streaming requests are not supported")

                output = self.process_chat_completions(request)
                return StreamingResponse(output, media_type="text/event-stream")
            except Exception as e:
                logger.error(f"Error processing chat completion: {str(e)}")
                raise

        @app.options("/v1/models")
        @app.get("/v1/models")
        def get_all_models():
            logger.debug("Received models list request")
            return JSONResponse(content={"object": "list", "data": self.get_models()})

        logger.info("Starting uvicorn server...")
        uvicorn.run(app, host=self.host, port=self.port, log_level=self.log_level)

    def process_chat_completions(self, request: dict):
        request_id = request.get("request_id", "req_0")
        
        messages = request.get("messages", [])
        if not messages or messages[-1]["role"] != "user":
            logger.error("Invalid request: Last message must be from the user")
            raise HTTPException(status_code=422, detail="The last message must be from the user.")
        
        logger.debug(f"Request contains {len(messages)} messages")
        clean_messages = self.get_clean_messages(messages)

        try:
            logger.debug("Applying chat template and tokenizing input")
            inputs = ov.Tensor(self.tokenizer.apply_chat_template(
                clean_messages,
                add_generation_prompt=True,
                tools=request.get("tools"),
                tokenize=True,
                return_tensors='np',
                **request.get("chat_template_kwargs", {}),
            ))
            
            logger.debug(f"Input tensor shape: {inputs.shape}")
        except Exception as e:
            logger.error(f"Failed to process input for request {request_id}: {str(e)}")
            raise

        generation_streamer = IterableStreamer(self.tokenizer)
        generation_config = self.create_generation_config_from_request(request, self.model.get_generation_config())

        def stream_chat_completion(streamer, _request_id):
            thread = Thread(target=self.model.generate, args=(inputs, generation_config, streamer))
            results = ""

            try:
                thread.start()
                yield self.build_chat_completion_chunk(_request_id, role="assistant", model=self.model_path)
                
                token_count = 0
                for result in streamer:
                    results += result
                    if result != "":
                        token_count += 1  # Might not be accurate. We assume the streamer outputs all
                        yield self.build_chat_completion_chunk(_request_id, content=result, model=self.model_path)

                logger.info(f"Generation completed for request {_request_id}. Prompt: {inputs.shape[1]} tokens, Generated: {token_count} tokens")
                usage = None
                if request.get("stream_options", {}).get("include_usage"):
                    usage = {
                        "prompt_tokens": inputs.shape[1],
                        "completion_tokens": token_count,
                        "total_tokens": inputs.shape[1] + token_count
                    }
                yield self.build_chat_completion_chunk(_request_id, finish_reason="stop", model=self.model_path, usage=usage)

                thread.join()
            except Exception as e:
                logger.error(f"Error during generation for request {_request_id}: {str(e)}")
                yield f'data: {{"error": "{str(e)}"}}'
        
        return stream_chat_completion(generation_streamer, request_id)

    def build_chat_completion_chunk(
            self,
            request_id="",
            content=None,
            model=None,
            role=None,
            finish_reason=None,
            usage=None,
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
            usage=CompletionUsage(**usage) if usage is not None else None,
        )
        return f"data: {chunk.model_dump_json(exclude_none=True)}\n\n"

    def create_generation_config_from_request(self, req, model_generation_config):
        try:
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
            
        except Exception as e:
            logger.error(f"Error creating generation config: {str(e)}")
            logger.warning("Falling back to default generation config")
            return model_generation_config

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="OV Server")
    OVServer.add_server_arguments(parser)
    args = parser.parse_args()
    
    # Configure logging
    logging.basicConfig(
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
        level=logging.INFO
    )
    
    logger.info("Starting OpenVINO GenAI Server")
    logger.info(f"Arguments: {vars(args)}")
    
    try:
        server = OVServer(**vars(args))
        server.run()
    except KeyboardInterrupt:
        logger.info("Server shutdown requested by user")
    except Exception as e:
        logger.error(f"Server failed to start: {str(e)}")
        raise