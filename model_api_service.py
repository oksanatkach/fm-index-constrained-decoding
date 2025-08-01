from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import vllm
import uvicorn
import argparse
import sys
from typing import Optional


class QuestionRequest(BaseModel):
    question: str
    max_tokens: Optional[int] = 100
    min_tokens: Optional[int] = 50
    temperature: Optional[float] = 0.0
    n: Optional[int] = 1
    top_p: Optional[float] = 1.0
    stop_tokens: Optional[list] = None


class AnswerResponse(BaseModel):
    answer: str


class VLLMService:
    def __init__(self, model_path: str, prompt_file: str):
        print(f"Loading model from {model_path}...")
        self.model = vllm.LLM(model=model_path,
                              enable_prefix_caching=True,
                              task="generate",
                              max_model_len=32768,
                              **{"compilation_config": {
                                  "level": 1,
                                  "use_inductor": False,
                                  "use_cudagraph": False,
                                  "backend": "eager"
                              }}
                              )

        try:
            with open(prompt_file, 'r') as fh:
                self.prompt = fh.read().strip()
            print(f"Loaded prompt from {prompt_file}")
        except FileNotFoundError:
            print(f"Error: Prompt file '{prompt_file}' not found.")
            sys.exit(1)

        print("Model loaded successfully!")

    def generate_answer(self, question: str, max_tokens: int = 100,
                        min_tokens: int = 50, temperature: float = 0.0,
                        stop_tokens: list = None) -> str:
        # Default stop tokens for Qwen models
        if stop_tokens is None:
            stop_tokens = ["<|endoftext|>", "<|im_end|>", "\n\n", "</think>"]

        output = self.model.generate(
            prompts=f"{self.prompt} {question}",
            sampling_params=vllm.SamplingParams(
                max_tokens=max_tokens,
                min_tokens=min_tokens,
                temperature=temperature,
                n=1,
                top_p=1.0,
                top_k=0,
                min_p=0.0,
                stop=stop_tokens
            )
        )
        return output[0].outputs[0].text

    def generate_answer_custom(self, question: str, max_tokens: int = 100, n: int = 1, top_p: float = 1.0,
                   min_tokens: int = 50, temperature: float = 0.0,
                   stop_tokens: list = None) -> str:
        # Default stop tokens for Qwen models
        if stop_tokens is None:
            stop_tokens = ["<|endoftext|>", "<|im_end|>", "\n\n", "</think>"]

        output = self.model.chat(
            messages=[{"role": "user", "content": question}],
            sampling_params=vllm.SamplingParams(
                max_tokens=max_tokens,
                min_tokens=min_tokens,
                temperature=temperature,
                n=n,
                top_p=top_p,
                top_k=0,
                min_p=0.0,
                stop=stop_tokens
            ),
            chat_template_kwargs={"enable_thinking": False}
        )
        return output[0].outputs[0].text

    def chat(self, question: str, max_tokens: int = 100, n: int = 1, top_p: float = 1.0,
                   min_tokens: int = 50, temperature: float = 0.0,
                   stop_tokens: list = None) -> str:
        # Default stop tokens for Qwen models
        if stop_tokens is None:
            stop_tokens = ["<|endoftext|>", "<|im_end|>", "\n\n", "</think>"]

        output = self.model.chat(
            messages=[{"role": "user", "content": f"{self.prompt} {question}"}],
            sampling_params=vllm.SamplingParams(
                max_tokens=max_tokens,
                min_tokens=min_tokens,
                temperature=temperature,
                n=n,
                top_p=top_p,
                top_k=0,
                min_p=0.0,
#                stop=stop_tokens
            ),
            chat_template_kwargs={"enable_thinking": False}
        )
        return output[0].outputs[0].text



# Global service instance
service = None

app = FastAPI(title="VLLM Question Answering API", version="1.0.0")


@app.on_event("startup")
async def startup_event():
    global service
    # These will be set by the main function
    model_path = getattr(app.state, 'model_path', "/home/qwen3-8b")
    prompt_file = getattr(app.state, 'prompt_file', 'PAQ_prompt_exp_2.txt')
    service = VLLMService(model_path, prompt_file)


@app.get("/")
async def root():
    return {"message": "VLLM Question Answering API is running"}


@app.get("/health")
async def health_check():
    return {"status": "healthy", "model_loaded": service is not None}


@app.post("/ask", response_model=AnswerResponse)
async def ask_question(request: QuestionRequest):
    if service is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet")

    try:
        answer = service.generate_answer(
            question=request.question,
            max_tokens=request.max_tokens,
            min_tokens=request.min_tokens,
            temperature=request.temperature,
            stop_tokens=request.stop_tokens
        )
        print(answer)
        return AnswerResponse(answer=answer)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating answer: {str(e)}")


@app.post("/ask_custom", response_model=AnswerResponse)
async def ask_question_custom(request: QuestionRequest):
    if service is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet")

    try:
        answer = service.generate_answer_custom(
            question=request.question,
            max_tokens=request.max_tokens,
            min_tokens=request.min_tokens,
            temperature=request.temperature,
            stop_tokens=request.stop_tokens
        )
        return AnswerResponse(answer=answer)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating answer: {str(e)}")


@app.post("/chat", response_model=AnswerResponse)
async def ask_question_custom(request: QuestionRequest):
    if service is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet")

    try:
        answer = service.chat(
            question=request.question,
            n=request.n,
            max_tokens=request.max_tokens,
            min_tokens=request.min_tokens,
            temperature=request.temperature,
            stop_tokens=request.stop_tokens
        )
        return AnswerResponse(answer=answer)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating answer: {str(e)}")


def main():
    parser = argparse.ArgumentParser(description='Start VLLM API server')
    parser.add_argument('--model', default="/home/qwen3-8b", help='Path to the model')
    parser.add_argument('--prompt-file', default='PAQ_prompt_exp_2.txt', help='Path to the prompt file')
    parser.add_argument('--host', default="127.0.0.1", help='Host to bind to')
    parser.add_argument('--port', type=int, default=8001, help='Port to bind to')

    args = parser.parse_args()

    # Store config in app state for startup event
    app.state.model_path = args.model
    app.state.prompt_file = args.prompt_file

    print(f"Starting VLLM API server on {args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
