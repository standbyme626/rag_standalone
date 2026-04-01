"""
重排器工厂 — 统一多后端 reranker 接口

支持的 backend:
  - qwen: Qwen CausalLM reranker（现有，支持 PyTorch/ONNX/SiliconFlow）
  - cross_encoder: SentenceTransformers CrossEncoder (新增)

用法:
    from app.rag.reranker import create_reranker
    reranker = create_reranker()  # 根据配置自动选择后端
"""

from typing import Optional

# Expose the original QwenReranker for backward compatibility
from app.rag.reranker.qwen_adapter import QwenReranker

__all__ = ["create_reranker", "QwenReranker"]


def create_reranker(
    backend: Optional[str] = None,
    model_path: Optional[str] = None,
    device: Optional[str] = None,
    cross_encoder_model: Optional[str] = None,
    batch_size: Optional[int] = None,
):
    """
    根据配置创建重排器实例

    Args:
        backend: 后端类型（None 时从配置读取）
            - "qwen" / "cross_encoder"
        model_path: Qwen 模型路径
        device: 设备类型
        cross_encoder_model: CrossEncoder 模型名称
        batch_size: 批处理大小

    Returns:
        BaseReranker 实例或 None（当所有后端都不可用时）
    """
    if backend is None or model_path is None or device is None \
            or cross_encoder_model is None or batch_size is None:
        from app.core.config import settings
        backend = backend or settings.RERANKER_BACKEND
        model_path = model_path or settings.RERANKER_MODEL_PATH
        device = device or settings.RERANKER_DEVICE
        cross_encoder_model = cross_encoder_model \
            or settings.RERANKER_CROSS_ENCODER_MODEL
        batch_size = batch_size or settings.RERANKER_BATCH_SIZE

    backend = backend.lower().strip()

    if backend == "cross_encoder":
        try:
            from app.rag.reranker.cross_encoder import CrossEncoderReranker
            print(f"[Reranker] Using CrossEncoder backend: {cross_encoder_model}")
            return CrossEncoderReranker(
                model_name=cross_encoder_model,
                device=device if device != "auto" else None,
                batch_size=batch_size,
            )
        except ImportError as e:
            print(f"[Reranker] CrossEncoder import failed, falling back to Qwen: {e}")
            return QwenReranker(model_path)

    return QwenReranker(model_path)
