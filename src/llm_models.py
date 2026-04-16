"""
Auxiliary functions for using LMMs with LangChain
"""

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from os.path import join

import dotenv
import os

from utils import MAIN_PATH

dotenv.load_dotenv(join(MAIN_PATH, "..", ".env"))

if "GOOGLE_API_KEY" not in os.environ:
    raise Exception("Missing GOOGLE_API_KEY on .env file.")


if "GROQ_API_KEY" not in os.environ:
    raise Exception("Missing GROQ_API_KEY on .env file.")

if "NVIDIA_API_KEY" not in os.environ:
    raise Exception("Missing NVIDIA_API_KEY on .env file.")


# https://github.com/cheahjs/free-llm-api-resources?tab=readme-ov-file
class Providers:
    GOOGLE = ChatGoogleGenerativeAI
    GROQ = ChatGroq
    NVIDIA = ChatNVIDIA


class GoogleModels:
    # https://docs.cloud.google.com/vertex-ai/generative-ai/docs/models/gemini/2-5-flash
    # RDP = 20
    GEMINI_2_5_FLASH = "gemini-2.5-flash"

    GEMMA_4_26B_A4B_IT = "gemma-4-26b-a4b-it"

    GEMMA_4_31B_IT = "gemma-4-31b-it"

    GEMINI_3_1_FLASH_LITE_PREVIEW = "gemini-3.1-flash-lite-preview"


class GroqModels:
    # https://console.groq.com/docs/rate-limits#rate-limits

    # https://console.groq.com/docs/compound/systems/compound
    # RDP = 250
    GROQ_COMPOUND = "groq/compound"

    # https://console.groq.com/docs/model/openai/gpt-oss-120b
    # RDP = 1K

    # Structure Outputs Models (All 1K Requisitions Per Day)
    OPENAI_GPT_OSS_120B = "openai/gpt-oss-120b"

    OPENAI_GPT_OSS_20B = "openai/gpt-oss-20b"

    OPENAI_GPT_OSS_SAFEGUARD_20B = "openai/gpt-oss-safeguard-20b"

    META_LLAMA_LLAMA_4_SCOUT_17B_16E_INSTRUCT = "meta-llama/llama-4-scout-17b-16e-instruct"

    QWEN_QWEN3_32B = "qwen/qwen3-32b"

    # --------------------------------------------------------------------------


    # RDP = 1K
    LLAMA_3_3_70B_VERSATILE = "llama-3.3-70b-versatile"

    # RDP = 14.4k
    LLAMA_3_1_8B_INSTANT = "llama-3.1-8b-instant"



class NvidiaModels:
    # https://build.nvidia.com/models

    DEEPSEEK_AI_DEEPSEEK_V3_2 = "deepseek-ai/deepseek-v3.2"

    DEEPSEEK_AI_DEEPSEEK_V3_1 = "deepseek-ai/deepseek-v3.1"

    MOONSHOTAI_KIMI_K2_INSTRUCT = "moonshotai/kimi-k2-instruct"

    Z_AI_GLM4_7 = "z-ai/glm4.7"

    MICROSOFT_PHI_3_5_MINI_INSTRUCT = "microsoft/phi-3.5-mini-instruct"

    NVIDIA_MISTRAL_NEMO_MINITRON_8B_BASE = "nvidia/mistral-nemo-minitron-8b-base"

    GOOGLE_GEMMA_2_27B_IT = "google/gemma-2-27b-it"

    MINIMAXAI_MINIMAX_M2_7 = "minimaxai/minimax-m2.7"

    META_LLAMA_GUARD_4_12B = "meta/llama-guard-4-12b"



def get_model(provider, model):
    return provider(
        model=model,
        temperature=0.4,
    )
