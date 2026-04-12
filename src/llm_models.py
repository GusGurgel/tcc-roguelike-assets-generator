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


class GroqModels:
    # https://console.groq.com/docs/rate-limits#rate-limits

    # https://console.groq.com/docs/compound/systems/compound
    # RDP = 250
    GROQ_COMPOUND = "groq/compound"

    # https://console.groq.com/docs/model/openai/gpt-oss-120b
    # RDP = 1K
    OPENAI_GPT_OSS_120B = "openai/gpt-oss-120b"

    OPENAI_GPT_OSS_20B = "openai/gpt-oss-20b"

    # LLAMA_3_1_8B_INSTANT = "llama-3.1-8b-instant"

    META_LLAMA_LLAMA_4_MAVERICK_17B_128E_INSTRUCT = (
        "meta-llama/llama-4-maverick-17b-128e-instruct"
    )


class NvidiaModels:
    # https://build.nvidia.com/models

    DEEPSEEK_V3_2 = "deepseek-ai/deepseek-v3.2"


def get_model(provider, model):
    return provider(
        model=model,
        temperature=0.4,
    )
