import torch
import logging
import os
import threading
import asyncio
import sys
import gc
import openai
import re
import json
from typing import AsyncGenerator, Optional, Dict, Any

from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
try:
    from transformers import BitsAndBytesConfig
    try:
        import bitsandbytes  # noqa: F401
        BNB_AVAILABLE = True
    except Exception:
        BNB_AVAILABLE = False
except Exception:
    BitsAndBytesConfig = None
    BNB_AVAILABLE = False
from app.core.config import settings
from app.core.models.vram_manager import vram_manager

# [V8.1] 引入 vLLM 支持
try:
    from vllm import AsyncLLMEngine, AsyncEngineArgs, SamplingParams
    VLLM_AVAILABLE = True
except ImportError:
    VLLM_AVAILABLE = False

logger = logging.getLogger(__name__)

class LocalSLMService:
    """
    单引擎本地模型服务 (Rebuilt for Pain Point #14)
    支持 vLLM 和 Transformers 的流式输出。
    """
    _instance = None
    _model = None
    _tokenizer = None
    _engine = None  # vLLM Engine
    _api_client = None # External vLLM API Client
    _engine_type = 'transformers' # 'vllm', 'transformers', or 'api'
    _load_lock = threading.Lock()
    _semaphore = None

    # [Chinese Environment] Ensure activation keys are in Chinese
    KEY_MEDICAL = "你是一个专业的医疗助手，请先进行逻辑思考再回答。"
    KEY_JSON = "你是一个严格遵循 JSON 格式的医疗接口。"
    KEY_TRIAGE_INTENT = (
        "你是一名专业的急诊分诊护士。你的任务是分析患者的主诉，"
        "并将其归类为以下四种意图之一：CRISIS, GREETING, VAGUE_SYMPTOM, COMPLEX_SYMPTOM。"
        "输出严格 JSON 格式。"
    )

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            with cls._load_lock:
                if cls._instance is None:
                    cls._instance = super(LocalSLMService, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        if hasattr(self, '_initialized') and self._initialized:
            return
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self._initialized = True
        
        # Initialize model_name with default
        self.model_name = os.path.join(settings.PROJECT_ROOT, "models", "Qwen3-0.6B-DPO-v11_2-Clean")
        
        # [VRAM] Register to VRAM Manager
        vram_manager.register_model("local_slm", self)

    def unload(self):
        """Unload model from VRAM (Called by VRAMManager)."""
        with self._load_lock:
            logger.info("🔻 [LocalSLM] Unloading model...")
            if self._model is not None:
                del self._model
                self._model = None
            
            if self._engine is not None:
                del self._engine
                self._engine = None
            
            if self._tokenizer is not None:
                del self._tokenizer
                self._tokenizer = None
            
            # Reset types but keep API client if it exists (it doesn't use VRAM)
            if self._engine_type != 'api':
                self._engine_type = None
            
            gc.collect()
            torch.cuda.empty_cache()
            logger.info("✅ [LocalSLM] Model unloaded successfully.")

    def is_loaded(self) -> bool:
        return self._model is not None or self._engine is not None or (self._engine_type == 'api' and self._api_client is not None)

    def _get_semaphore(self):
        if LocalSLMService._semaphore is None:
            LocalSLMService._semaphore = asyncio.Semaphore(5)
        return LocalSLMService._semaphore

    def _load_model(self):
        with self._load_lock:
            # [Security] Check global switch
            if not settings.ENABLE_LOCAL_FALLBACK:
                logger.warning("🛑 [LocalSLM] Local fallback is disabled by configuration. Aborting load.")
                raise RuntimeError("Local SLM is disabled by configuration (ENABLE_LOCAL_FALLBACK=False).")

            if self.is_loaded():
                vram_manager.mark_used("local_slm")
                return
            
            # [VRAM] 推理资源预编排
            vram_manager.orchestrate_pre_inference(required_mb=1500)

            # 1. Try External API
            try:
                if os.getenv("SKIP_API_CHECK") != "true":
                    test_client = openai.OpenAI(base_url=settings.LOCAL_SLM_URL, api_key="empty", timeout=2.0)
                    test_client.models.list()
                    self._api_client = openai.AsyncOpenAI(base_url=settings.LOCAL_SLM_URL, api_key="empty", timeout=10.0)
                    self._engine_type = 'api'
                    logger.info("✅ [LocalSLM] Connected to external vLLM API.")
                    return
            except Exception:
                pass

            # 2. Local Model Path
            self.model_name = os.path.join(settings.PROJECT_ROOT, "models", "Qwen3-0.6B-DPO-v11_2-Clean")
            if not os.path.exists(self.model_name):
                 # Fallback for dev environment if model missing
                 self.model_name = "Qwen/Qwen2.5-0.5B-Instruct" 

            # 3. Try vLLM
            if VLLM_AVAILABLE and self.device == "cuda":
                try:
                    logger.info(f"🚀 [LocalSLM] Initializing vLLM Engine with {self.model_name}...")
                    
                    # [Optim] Dynamic Quantization Configuration
                    quant_config = settings.LOCAL_SLM_QUANTIZATION
                    if quant_config and quant_config.lower() == "none":
                        quant_config = None
                        
                    engine_args = AsyncEngineArgs(
                        model=self.model_name,
                        trust_remote_code=True,
                        gpu_memory_utilization=0.15,
                        max_model_len=2048,
                        dtype="float16",
                        quantization=quant_config, # [Optim] Enable Int8/AWQ
                        enforce_eager=True,
                    )
                    self._engine = AsyncLLMEngine.from_engine_args(engine_args)
                    self._engine_type = 'vllm'
                    self._tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
                    logger.info("✅ [LocalSLM] vLLM engine loaded.")
                    return
                except Exception as ve:
                    logger.warning(f"⚠️ [LocalSLM] vLLM load failed: {ve}")

            # 4. Fallback Transformers
            logger.info(f"🚀 [LocalSLM] Loading Transformers engine: {self.model_name}")
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
            quant_mode = (settings.LOCAL_SLM_QUANTIZATION or "").strip().lower()
            model_kwargs = {
                "trust_remote_code": True,
            }

            # [Fix] Enable int8/int4 quantization in Transformers fallback (bitsandbytes).
            if self.device == "cuda" and quant_mode in {"int8", "8bit", "int4", "4bit"} and BNB_AVAILABLE:
                if quant_mode in {"int4", "4bit"}:
                    model_kwargs["quantization_config"] = BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_quant_type="nf4",
                        bnb_4bit_compute_dtype=torch.float16,
                    )
                    logger.info("⚙️ [LocalSLM] Transformers 4-bit quantization enabled (bitsandbytes).")
                else:
                    model_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
                    logger.info("⚙️ [LocalSLM] Transformers 8-bit quantization enabled (bitsandbytes).")
                model_kwargs["device_map"] = "auto"
            else:
                if self.device == "cuda" and quant_mode in {"int8", "8bit", "int4", "4bit"} and not BNB_AVAILABLE:
                    logger.warning("⚠️ [LocalSLM] bitsandbytes is unavailable. Falling back to float16 Transformers.")
                model_kwargs["torch_dtype"] = torch.float16 if self.device == "cuda" else torch.float32
                model_kwargs["device_map"] = self.device

            try:
                self._model = AutoModelForCausalLM.from_pretrained(self.model_name, **model_kwargs)
            except Exception as te:
                # [Resilience] If bitsandbytes path is unavailable/incompatible, fallback to float16.
                err_lower = str(te).lower()
                quant_requested = quant_mode in {"int8", "8bit", "int4", "4bit"}
                bnb_error = "bitsandbytes" in err_lower or "8-bit quantization" in err_lower or "4-bit quantization" in err_lower
                if quant_requested and bnb_error:
                    logger.warning(f"⚠️ [LocalSLM] Quantized Transformers load failed, fallback to float16: {te}")
                    fallback_kwargs = {
                        "trust_remote_code": True,
                        "torch_dtype": torch.float16 if self.device == "cuda" else torch.float32,
                        "device_map": self.device,
                    }
                    self._model = AutoModelForCausalLM.from_pretrained(self.model_name, **fallback_kwargs)
                else:
                    raise
            self._engine_type = 'transformers'
            logger.info("✅ [LocalSLM] Transformers engine loaded.")

    async def generate_stream(self, prompt: str, **kwargs) -> AsyncGenerator[str, None]:
        """
        [V12.0] Native Async Generator for Streaming (Pain Point #14)
        """
        # [VRAM] Mark used
        vram_manager.mark_used("local_slm")

        if not self.is_loaded():
            await asyncio.to_thread(self._load_model)

        system_prompt = kwargs.get("system_prompt", self.KEY_MEDICAL)
        max_tokens = kwargs.get("max_new_tokens", 512)
        temperature = kwargs.get("temperature", 0.3)
        
        # --- API Mode ---
        if self._engine_type == 'api':
            async with self._get_semaphore():
                try:
                    # [V8.2] 兼容 llama-server 的 /v1/chat/completions
                    # llama-server 不需要手动拼接 ChatML，它会自动处理 messages
                    stream = await self._api_client.chat.completions.create(
                        model=settings.LOCAL_SLM_MODEL, # 使用配置中的模型名
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": prompt}
                        ],
                        temperature=temperature,
                        max_tokens=max_tokens,
                        stream=True,
                        extra_body={"repetition_penalty": 1.2} # [Optim] Increase repetition penalty
                    )
                    async for chunk in stream:
                        content = chunk.choices[0].delta.content or ""
                        if content:
                            yield content
                except Exception as e:
                    logger.error(f"API Stream Error: {e}")
                    yield f"[Error: {e}]"
            return

        # --- vLLM Mode ---
        if self._engine_type == 'vllm':
            async with self._get_semaphore():
                import uuid
                request_id = str(uuid.uuid4())
                sampling_params = SamplingParams(
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=0.9,
                    repetition_penalty=1.2, # [Optim] Increase repetition penalty
                    stop=["<|im_end|>", "<|endoftext|>"]
                )
                
                results_generator = self._engine.generate(prompt, sampling_params, request_id)
                
                previous_text = ""
                async for request_output in results_generator:
                    current_text = request_output.outputs[0].text
                    delta = current_text[len(previous_text):]
                    if delta:
                        yield delta
                    previous_text = current_text
            return

        # --- Transformers Mode (with TextIteratorStreamer) ---
        if self._engine_type == 'transformers':
            async with self._get_semaphore():
                inputs = self._tokenizer(prompt, return_tensors="pt").to(self.device)
                streamer = TextIteratorStreamer(self._tokenizer, skip_prompt=True, skip_special_tokens=True)
                do_sample = temperature is None or temperature > 0
                
                gen_kwargs = {
                    "input_ids": inputs.input_ids,
                    "attention_mask": inputs.attention_mask,
                    "max_new_tokens": max_tokens,
                    "do_sample": do_sample,
                    "repetition_penalty": 1.2, # [Optim] Increase repetition penalty
                    "streamer": streamer,
                    "pad_token_id": self._tokenizer.eos_token_id
                }
                if do_sample:
                    gen_kwargs["temperature"] = temperature

                thread = threading.Thread(target=self._model.generate, kwargs=gen_kwargs)
                thread.start()

                loop = asyncio.get_running_loop()
                queue = asyncio.Queue()
                
                def _consume_stream():
                    try:
                        for text in streamer:
                            if not loop.is_closed():
                                loop.call_soon_threadsafe(queue.put_nowait, text)
                            else:
                                break
                    except Exception as e:
                        # Ignore loop closed errors during shutdown
                        if "Event loop is closed" not in str(e):
                            logger.error(f"Stream consumption error: {e}")
                    finally:
                        try:
                            if not loop.is_closed():
                                loop.call_soon_threadsafe(queue.put_nowait, None)
                        except RuntimeError:
                            pass

                threading.Thread(target=_consume_stream).start()

                while True:
                    token = await queue.get()
                    if token is None:
                        break
                    yield token

    async def generate_response_async(self, prompt: str, **kwargs) -> str:
        """Non-streaming wrapper"""
        response = ""
        async for token in self.generate_stream(prompt, **kwargs):
            response += token
            if kwargs.get("stream_callback"):
                await kwargs["stream_callback"](token)
        return response.strip()

    async def generate_batch_async(self, prompts: list[str], **kwargs) -> list[str]:
        """
        Batch generation using async gather.
        """
        tasks = []
        for prompt in prompts:
            tasks.append(self.generate_response_async(prompt, **kwargs))
        return await asyncio.gather(*tasks)

    async def constrained_classify(self, query: str, categories: list[str], reasoning: bool = False) -> str:
        """
        Classify query into one of the categories.
        """
        if not categories:
            return "OTHER"

        normalized = [str(c).strip().upper() for c in categories if str(c).strip()]
        if not normalized:
            return "OTHER"

        categories_str = ", ".join(normalized)
        query_text = (query or "").strip()
        signals = self._intent_signals(query_text)

        # Align with latest triage/intent corpora: Chinese emergency triage + strict JSON output.
        if {"CRISIS", "GREETING", "VAGUE_SYMPTOM", "COMPLEX_SYMPTOM"}.issubset(set(normalized)):
            # Deterministic fast-path:
            # pure greeting (and no symptom evidence) should not enter model ambiguity.
            if signals["greeting_score"] > 0 and signals["symptom_score"] == 0:
                return "GREETING"

            system_prompt = self.KEY_TRIAGE_INTENT
            prompt = f"患者主诉：{query}\n请输出 JSON："
            max_new_tokens = 32
        else:
            # Keep legacy generic classification compatibility.
            system_prompt = self.KEY_JSON
            prompt = (
                f"请将用户输入分类到以下标签之一：{categories_str}\n"
                f"用户输入：{query}\n"
                "只输出 JSON，例如 {\"intent\":\"LABEL\"}"
            )
            max_new_tokens = 32

        response = await self.generate_response_async(
            prompt,
            system_prompt=system_prompt,
            max_new_tokens=max_new_tokens,
            temperature=0.0,
        )

        candidate = self._extract_category_from_text(response, normalized)
        if candidate:
            # Post-correction for the most common confusion pair GREETING <-> VAGUE_SYMPTOM.
            if {"GREETING", "VAGUE_SYMPTOM", "COMPLEX_SYMPTOM"}.issubset(set(normalized)):
                corrected = self._resolve_greeting_symptom_conflict(
                    candidate=candidate,
                    signals=signals,
                    categories=normalized,
                )
                if corrected:
                    return corrected
            return candidate

        # Heuristic fallback should not collapse to categories[0] (e.g., all -> CRISIS).
        fallback = self._heuristic_intent_fallback(query_text, normalized)
        return fallback

    def _intent_signals(self, query: str) -> Dict[str, int]:
        text = (query or "").strip().lower()
        if not text:
            return {
                "greeting_score": 0,
                "symptom_score": 0,
                "detail_score": 0,
            }

        greeting_keywords = [
            "你好", "您好", "在吗", "嗨", "哈喽", "hello", "hi", "早上好", "晚上好",
            "中午好", "你是谁", "能听到吗"
        ]
        symptom_keywords = [
            "痛", "疼", "痒", "晕", "恶心", "呕吐", "发烧", "咳", "咳嗽", "流鼻涕", "鼻塞",
            "头痛", "头晕", "胸闷", "胸痛", "气短", "呼吸", "腹痛", "肚子", "腹泻", "拉肚子",
            "便血", "血压", "心慌", "心悸", "乏力", "不舒服", "难受", "过敏", "红疹", "炎症"
        ]
        detail_keywords = [
            "天", "周", "月", "小时", "持续", "反复", "加重", "一直", "突然", "夜里",
            "饭后", "体温", "度", "分钟", "昨晚", "今天"
        ]

        greeting_score = sum(1 for kw in greeting_keywords if kw in text)
        symptom_score = sum(1 for kw in symptom_keywords if kw in text)
        detail_score = sum(1 for kw in detail_keywords if kw in text)
        return {
            "greeting_score": greeting_score,
            "symptom_score": symptom_score,
            "detail_score": detail_score,
        }

    def _resolve_greeting_symptom_conflict(
        self,
        candidate: str,
        signals: Dict[str, int],
        categories: list[str],
    ) -> Optional[str]:
        greeting_score = int(signals.get("greeting_score", 0))
        symptom_score = int(signals.get("symptom_score", 0))
        detail_score = int(signals.get("detail_score", 0))

        # If symptom evidence exists, don't keep GREETING.
        if candidate == "GREETING" and symptom_score > 0:
            if detail_score > 0 and "COMPLEX_SYMPTOM" in categories:
                return "COMPLEX_SYMPTOM"
            if "VAGUE_SYMPTOM" in categories:
                return "VAGUE_SYMPTOM"

        # If no symptom evidence and clear greeting evidence, prefer GREETING.
        if candidate in {"VAGUE_SYMPTOM", "COMPLEX_SYMPTOM"} and symptom_score == 0 and greeting_score > 0:
            if "GREETING" in categories:
                return "GREETING"
        return None

    def _extract_category_from_text(self, text: str, categories: list[str]) -> Optional[str]:
        raw = (text or "").strip()
        if not raw:
            return None

        # 1) Try JSON block first.
        m = re.search(r"\{.*\}", raw, re.DOTALL)
        if m:
            try:
                data = json.loads(m.group(0))
                for k in ("intent", "category", "label", "class"):
                    v = data.get(k)
                    if isinstance(v, str):
                        vv = v.strip().upper()
                        if vv in categories:
                            return vv
            except Exception:
                pass

        # 2) Try XML-style tags.
        m = re.search(r"<category>\s*([A-Z_]+)\s*</category>", raw.upper())
        if m and m.group(1) in categories:
            return m.group(1)

        # 3) Exact label mention.
        up = raw.upper()
        for c in categories:
            if re.search(rf"\b{re.escape(c)}\b", up):
                return c
        return None

    def _heuristic_intent_fallback(self, query: str, categories: list[str]) -> str:
        text = (query or "").strip().lower()
        signals = self._intent_signals(text)

        def has(label: str) -> bool:
            return label in categories

        # Priority: any symptom evidence should avoid GREETING fallback.
        if signals["symptom_score"] > 0:
            if signals["detail_score"] > 0 and has("COMPLEX_SYMPTOM"):
                return "COMPLEX_SYMPTOM"
            if has("VAGUE_SYMPTOM"):
                return "VAGUE_SYMPTOM"
            if has("MEDICAL_CONSULT"):
                return "MEDICAL_CONSULT"

        if signals["greeting_score"] > 0 and signals["symptom_score"] == 0 and has("GREETING"):
            return "GREETING"

        crisis_kws = ["救命", "不想活", "自杀", "胸痛", "昏迷", "呼吸困难", "大出血", "120"]
        greeting_kws = ["你好", "您好", "hello", "hi", "在吗", "你是谁"]
        vague_kws = ["不舒服", "难受", "有点", "怎么办", "咋办", "不太好"]

        if has("CRISIS") and any(k in text for k in crisis_kws):
            return "CRISIS"
        if has("GREETING") and any(k in text for k in greeting_kws):
            return "GREETING"
        if has("VAGUE_SYMPTOM") and (len(text) <= 6 or any(k in text for k in vague_kws)):
            return "VAGUE_SYMPTOM"
        if has("COMPLEX_SYMPTOM"):
            return "COMPLEX_SYMPTOM"
        if has("MEDICAL_CONSULT"):
            return "MEDICAL_CONSULT"
        return categories[0]

local_slm = LocalSLMService()
