import os
import logging

def setup_logger(name, timestamp, log_dir= "inference_logs"):
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, f"{name}_{timestamp}.log")
    
    logging.basicConfig(
        filename=log_path,
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        filemode='w'
    )
    return logging.getLogger(name), log_path


def get_latest_model(model_type: str, model_dir: str = "models"):
    pointer_file = os.path.join(model_dir, f"latest_{model_type}_model.txt")
    
    if os.path.exists(pointer_file):
        with open(pointer_file, "r") as f:
            model_path = f.read().strip()
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model path in {pointer_file} does not exist: {model_path}")
        return model_path
    else:
        # Fallback: load latest-looking .pkl in model_type directory
        model_type_dir = os.path.join(model_dir, model_type)
        if not os.path.exists(model_type_dir):
            raise FileNotFoundError(f"No pointer and no model directory found: {model_type_dir}")
        
        candidates = sorted(
            [f for f in os.listdir(model_type_dir) if f.endswith(".pkl")],
            reverse=True
        )
        if not candidates:
            raise FileNotFoundError(f"No model files found in {model_type_dir}")
        
        fallback_model = os.path.join(model_type_dir, candidates[0])
        logging.warning(f"No pointer found. Using fallback model: {fallback_model}")
        return fallback_model