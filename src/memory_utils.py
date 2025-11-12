import torch
import gc
import logging

logger = logging.getLogger(__name__)


def clear_gpu_memory():
    """
    Clear GPU memory and run garbage collection.
    """
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    gc.collect()


def get_gpu_memory_info():
    """
    Get GPU memory information.
    
    Returns:
        Dictionary containing GPU memory info
    """
    if not torch.cuda.is_available():
        return {"available": False}
    
    device = torch.cuda.current_device()
    total_memory = torch.cuda.get_device_properties(device).total_memory
    allocated_memory = torch.cuda.memory_allocated(device)
    cached_memory = torch.cuda.memory_reserved(device)
    free_memory = total_memory - allocated_memory
    
    return {
        "available": True,
        "device": device,
        "total_memory_gb": total_memory / (1024**3),
        "allocated_memory_gb": allocated_memory / (1024**3),
        "cached_memory_gb": cached_memory / (1024**3),
        "free_memory_gb": free_memory / (1024**3),
        "memory_usage_percent": (allocated_memory / total_memory) * 100
    }


def log_gpu_memory_usage(stage: str = ""):
    """
    Log current GPU memory usage.
    
    Args:
        stage: Description of the current stage
    """
    memory_info = get_gpu_memory_info()
    if memory_info["available"]:
        logger.info(
            f"GPU Memory Usage {stage}: "
            f"Allocated: {memory_info['allocated_memory_gb']:.2f}GB, "
            f"Cached: {memory_info['cached_memory_gb']:.2f}GB, "
            f"Free: {memory_info['free_memory_gb']:.2f}GB, "
            f"Usage: {memory_info['memory_usage_percent']:.1f}%"
        )
    else:
        logger.info(f"GPU not available {stage}")


def memory_efficient_batch_processing(
    model, 
    batch_x, 
    batch_y, 
    device, 
    criterion=None, 
    optimizer=None,
    training=True
):
    """
    Memory-efficient batch processing with automatic memory cleanup.
    
    Args:
        model: PyTorch model
        batch_x: Input batch
        batch_y: Target batch
        device: Device to use
        criterion: Loss function (required for training)
        optimizer: Optimizer (required for training)
        training: Whether this is training or evaluation
        
    Returns:
        Tuple of (outputs, loss) or just outputs for evaluation
    """
    # Move batch to GPU
    batch_x = batch_x.to(device, non_blocking=True)
    batch_y = batch_y.to(device, non_blocking=True)
    
    if training:
        # Training mode
        model.train()
        optimizer.zero_grad()
        outputs = model(batch_x)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        
        # Clear GPU memory
        del batch_x, batch_y, outputs, loss
        torch.cuda.empty_cache()
        
        return None, None  # Loss will be handled separately
    else:
        # Evaluation mode
        model.eval()
        with torch.no_grad():
            outputs = model(batch_x)
            if criterion is not None:
                loss = criterion(outputs, batch_y)
            else:
                loss = None
        
        # Move outputs to CPU before clearing GPU memory
        outputs_cpu = outputs.cpu()
        if loss is not None:
            loss_cpu = loss.cpu()
        else:
            loss_cpu = None
        
        # Clear GPU memory
        del batch_x, batch_y, outputs
        if loss is not None:
            del loss
        torch.cuda.empty_cache()
        
        return outputs_cpu, loss_cpu


def monitor_memory_usage(func):
    """
    Decorator to monitor memory usage during function execution.
    
    Args:
        func: Function to monitor
        
    Returns:
        Wrapped function
    """
    def wrapper(*args, **kwargs):
        log_gpu_memory_usage("before function execution")
        try:
            result = func(*args, **kwargs)
            log_gpu_memory_usage("after function execution")
            return result
        except Exception as e:
            log_gpu_memory_usage("after function error")
            raise e
        finally:
            clear_gpu_memory()
    
    return wrapper
