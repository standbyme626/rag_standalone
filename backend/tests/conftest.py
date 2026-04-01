import os
import sys
from pathlib import Path

# Set required env vars BEFORE any app imports to avoid Settings validation errors
os.environ.setdefault("OPENAI_MODEL_NAME", "gpt-3.5-turbo")
os.environ.setdefault("MILVUS_HOST", "localhost")
os.environ.setdefault("MILVUS_PORT", "19530")
os.environ.setdefault("REDIS_HOST", "localhost")
os.environ.setdefault("REDIS_PORT", "6379")
os.environ.setdefault("DATABASE_URL", "sqlite:///./test.db")
os.environ.setdefault("ENABLE_HIERARCHICAL_INDEX", "false")
os.environ.setdefault("LANGFUSE_PUBLIC_KEY", "test-key")
os.environ.setdefault("LANGFUSE_SECRET_KEY", "test-secret")
os.environ.setdefault("LANGFUSE_HOST", "https://cloud.langfuse.com")

sys.path.insert(0, str(Path(__file__).parent.parent))
