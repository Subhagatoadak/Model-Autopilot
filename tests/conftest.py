import os
import pytest

# Set test environment
os.environ["OPENAI_API_KEY"] = "test-key"
os.environ["DB_PATH"] = "data/test_events.db"
