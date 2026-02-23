# Import all model modules to trigger registration
import backend.model_zoo.sklearn_models  # noqa: F401
import backend.model_zoo.xgboost_models  # noqa: F401
import backend.model_zoo.lightgbm_models  # noqa: F401

# PyTorch and TensorFlow are optional (heavy dependencies)
try:
    import backend.model_zoo.pytorch_models  # noqa: F401
except ImportError:
    pass

try:
    import backend.model_zoo.tensorflow_models  # noqa: F401
except ImportError:
    pass
