import torch
import logging
import gc
import psutil
import os
import asyncio
import time
from typing import Optional, Dict, Any
from app.core.config import settings

logger = logging.getLogger(__name__)

class VRAMManager:
    """
    [Optimization Plan 2] 推理资源与显存编排管理器 (VRAM & Orchestration)
    目标：统一管理 GPU 显存，实现抢占式清理与热驻留策略。
    """
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(VRAMManager, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.total_vram = 0
        if torch.cuda.is_available():
            self.total_vram = torch.cuda.get_device_properties(0).total_memory / (1024**2) # MB
        
        # 预设阈值
        self.critical_threshold = 0.9 # 90% 触发强制清理
        self.warning_threshold = 0.7  # 70% 触发预警清理
        
        # Model Registry & LRU
        self.registered_models: Dict[str, Any] = {} # name -> model_instance
        self.model_access_history: Dict[str, float] = {} # name -> timestamp
        self._async_semaphore = asyncio.Semaphore(1) # Limit concurrent heavy inference

        self._initialized = True
        logger.info(f"🚀 [VRAMManager] Initialized on {self.device}. Total VRAM: {self.total_vram:.2f} MB")

    def register_model(self, name: str, instance: Any):
        """Register a model instance for management."""
        self.registered_models[name] = instance
        self.model_access_history[name] = time.time()
        logger.info(f"📝 [VRAMManager] Model Registered: {name}")

    def mark_used(self, name: str):
        """Mark a model as recently used (for LRU tracking)."""
        if name in self.registered_models:
            self.model_access_history[name] = time.time()

    def unload_model(self, name: str):
        """Unload a specific model."""
        if name in self.registered_models:
            logger.info(f"🔻 [VRAMManager] Unloading Model: {name}")
            instance = self.registered_models[name]
            if hasattr(instance, 'unload'):
                try:
                    instance.unload()
                except Exception as e:
                    logger.error(f"❌ [VRAMManager] Error unloading {name}: {e}")
            
            # Remove from registry? No, we keep it registered so we can reload it?
            # Actually, the instance persists (singleton), but its internal model is None.
            # So we don't remove it from registered_models, just update its status if needed.
            # But for LRU purposes, we might want to know it's unloaded.
            # Let's assume the instance handles the "reloading" when called next time.
            
            self.clear_cache(force=True)

    def unload_lru(self):
        """Unload the least recently used model."""
        if not self.registered_models:
            return
        
        # Find LRU model that is currently loaded (we need a way to check if loaded)
        # For now, we assume all registered models might be loaded.
        # Ideally, we should check `instance.is_loaded()` if available.
        
        sorted_models = sorted(self.model_access_history.items(), key=lambda x: x[1])
        
        for name, _ in sorted_models:
            instance = self.registered_models[name]
            # Check if model is actually loaded to avoid useless unload calls
            is_loaded = True
            if hasattr(instance, 'is_loaded'):
                if callable(instance.is_loaded):
                    is_loaded = instance.is_loaded()
                else:
                    is_loaded = instance.is_loaded
            
            if is_loaded:
                logger.warning(f"📉 [VRAMManager] Triggering LRU Unload for: {name}")
                self.unload_model(name)
                return

    async def acquire_inference_permit(self):
        """Acquire permission to run inference (concurrency control)."""
        await self._async_semaphore.acquire()
        
    def release_inference_permit(self):
        self._async_semaphore.release()

    def get_allocated_vram(self) -> float:
        """获取已分配显存 (MB)"""
        if not torch.cuda.is_available():
            return 0.0
        return torch.cuda.memory_allocated() / (1024**2)

    def get_reserved_vram(self) -> float:
        """获取已预留显存 (MB)"""
        if not torch.cuda.is_available():
            return 0.0
        return torch.cuda.memory_reserved() / (1024**2)

    def get_vram_utilization(self) -> float:
        """获取显存利用率 (0.0 - 1.0)"""
        if self.total_vram == 0:
            return 0.0
        return self.get_allocated_vram() / self.total_vram

    def clear_cache(self, force: bool = False):
        """
        执行显存清理逻辑 (同步版)。
        """
        if not torch.cuda.is_available():
            return

        utilization = self.get_vram_utilization()
        
        # [Fix] 优化清理逻辑：只有在显存占用较高时才触发清理，避免过度同步
        # torch.cuda.empty_cache() 是一个同步操作，频繁调用会显著降低性能
        if force or utilization > 0.7: # 提高阈值到 70%
            try:
                # 记录时间以监控开销
                start_t = time.time()
                before = self.get_allocated_vram()
                
                # gc.collect() 只有在 force=True 时才执行完全 GC
                if force:
                    gc.collect()
                
                torch.cuda.empty_cache()
                
                # 移除非必要的 ipc_collect，除非明确需要
                if force:
                    try:
                        torch.cuda.ipc_collect()
                    except:
                        pass
                        
                after = self.get_allocated_vram()
                duration = time.time() - start_t
                
                # 仅在清理耗时较长或释放量较大时打印日志
                if duration > 0.1 or (before - after > 100): 
                    logger.info(f"🧹 [VRAMManager] Cache Cleared in {duration:.3f}s. VRAM: {before:.2f}MB -> {after:.2f}MB (Util: {utilization:.2%})")
            except Exception as e:
                logger.error(f"❌ [VRAMManager] Clear Cache Failed: {str(e)}")

    async def clear_cache_async(self, force: bool = False):
        """
        执行显存清理逻辑 (异步版)，防止阻塞事件循环。
        """
        if not torch.cuda.is_available():
            return
        
        # 将阻塞的清理操作放到线程池中执行
        await asyncio.to_thread(self.clear_cache, force=force)

    def orchestrate_pre_inference(self, required_mb: float = 500):
        """
        [Preemptive Orchestration] 推理前编排。
        如果可用显存不足以承载本次推理，则执行激进清理。
        """
        if not torch.cuda.is_available():
            return

        available_vram = self.total_vram - self.get_allocated_vram()
        if available_vram < required_mb:
            logger.info(f"⚡ [VRAMManager] Low available VRAM ({available_vram:.2f}MB). Orchestrating for {required_mb}MB request...")
            
            # 1. First try simple cache clear
            self.clear_cache(force=True)
            available_vram = self.total_vram - self.get_allocated_vram()
            
            # 2. If still not enough, try LRU unload
            if available_vram < required_mb:
                logger.warning(f"⚡ [VRAMManager] Still low VRAM ({available_vram:.2f}MB). Attempting LRU Unload...")
                self.unload_lru()
                
    def get_status_report(self) -> dict:
        """获取详细状态报告"""
        return {
            "device": str(self.device),
            "total_mb": self.total_vram,
            "allocated_mb": self.get_allocated_vram(),
            "reserved_mb": self.get_reserved_vram(),
            "utilization": self.get_vram_utilization(),
            "process_ram_mb": psutil.Process(os.getpid()).memory_info().rss / (1024**2),
            "registered_models": list(self.registered_models.keys())
        }

    async def orchestrate_pre_inference_async(self, required_mb: float = 500):
        """
        [Preemptive Orchestration] 推理前编排 (异步版)。
        如果可用显存不足以承载本次推理，则执行激进清理。
        """
        if not torch.cuda.is_available():
            return

        available_vram = self.total_vram - self.get_allocated_vram()
        if available_vram < required_mb:
            logger.info(f"⚡ [VRAMManager] Low available VRAM ({available_vram:.2f}MB). Orchestrating for {required_mb}MB request...")
            await self.clear_cache_async(force=True)
            
            # Check again and maybe unload LRU (requires sync call or async wrapper)
            available_vram = self.total_vram - self.get_allocated_vram()
            if available_vram < required_mb:
                 # Run LRU unload in thread pool
                 await asyncio.to_thread(self.unload_lru)

def vram_auto_clear(force: bool = False):
    """
    [Decorator] 自动显存清理装饰器。
    在函数执行前进行显存状态检查并尝试清理。
    """
    def decorator(func):
        import functools
        if asyncio.iscoroutinefunction(func):
            @functools.wraps(func)
            async def async_wrapper(*args, **kwargs):
                # 使用异步清理防止卡死
                await vram_manager.clear_cache_async(force=force)
                return await func(*args, **kwargs)
            return async_wrapper
        else:
            @functools.wraps(func)
            def sync_wrapper(*args, **kwargs):
                vram_manager.clear_cache(force=force)
                return func(*args, **kwargs)
            return sync_wrapper
    return decorator

vram_manager = VRAMManager()
