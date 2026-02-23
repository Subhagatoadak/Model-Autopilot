import io
import os
import pytest
from httpx import AsyncClient, ASGITransport

# Ensure model zoo is populated
import backend.model_zoo.sklearn_models
import backend.model_zoo.xgboost_models
import backend.model_zoo.lightgbm_models

from backend.database import init_db
from backend.main import app


@pytest.fixture(autouse=True)
async def setup_db():
    """Initialize the database before each test."""
    # Clean up test DB
    from backend.database import _db_path
    path = _db_path()
    if os.path.exists(path):
        os.remove(path)
    await init_db()
    yield
    if os.path.exists(path):
        os.remove(path)


@pytest.fixture
async def client():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


@pytest.mark.asyncio
async def test_health(client):
    response = await client.get("/api/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


@pytest.mark.asyncio
async def test_list_datasets_empty(client):
    response = await client.get("/api/datasets")
    assert response.status_code == 200
    assert isinstance(response.json(), list)
    assert len(response.json()) == 0


@pytest.mark.asyncio
async def test_model_zoo(client):
    response = await client.get("/api/models/zoo")
    assert response.status_code == 200
    models = response.json()
    assert len(models) > 0
    names = [m["name"] for m in models]
    assert "random_forest" in names


@pytest.mark.asyncio
async def test_model_zoo_filtered(client):
    response = await client.get("/api/models/zoo", params={"task_type": "classification"})
    assert response.status_code == 200
    models = response.json()
    for m in models:
        assert "classification" in m["task_types"]


@pytest.mark.asyncio
async def test_upload_csv(client):
    csv_content = b"a,b,target\n1,2,0\n3,4,1\n5,6,0\n7,8,1\n"
    response = await client.post(
        "/api/datasets/upload",
        files={"file": ("test.csv", io.BytesIO(csv_content), "text/csv")},
        data={"name": "test_dataset", "target_col": "target"},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["name"] == "test_dataset"
    assert data["num_rows"] == 4
    assert data["num_cols"] == 3
    assert data["target_col"] == "target"


@pytest.mark.asyncio
async def test_upload_non_csv(client):
    response = await client.post(
        "/api/datasets/upload",
        files={"file": ("test.txt", io.BytesIO(b"not csv"), "text/plain")},
        data={"name": "bad"},
    )
    assert response.status_code == 400


@pytest.mark.asyncio
async def test_list_experiments_empty(client):
    response = await client.get("/api/experiments")
    assert response.status_code == 200
    assert isinstance(response.json(), list)
    assert len(response.json()) == 0
