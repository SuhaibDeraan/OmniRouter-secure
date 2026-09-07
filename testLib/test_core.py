import os

import pytest
from fastapi.testclient import TestClient
from serverRouter.router import app
from .test_utils import test_logger

# The integration test suite authenticates against a live OmniRouter key.
# The key must never be committed — supply it via the TEST_OMNI_API_KEY
# environment variable (see testLib/TESTING.md). Tests are skipped when unset.
TEST_OMNI_API_KEY = os.environ.get("TEST_OMNI_API_KEY")

class BaseTest:
    def setup_method(self):
        if not TEST_OMNI_API_KEY:
            pytest.skip("TEST_OMNI_API_KEY is not set; skipping integration test")
        self.client = TestClient(app)
        self.logger = test_logger
        self.client.headers = {
            "Authorization": f"Bearer {TEST_OMNI_API_KEY}"
        }

class TestBasicEndpoints(BaseTest):
    def test_root(self):
        self.logger.info("Test Root")
        response = self.client.get("/")
        self.logger.debug(f"Test Root: {response.json()}")
        assert response.status_code == 200
        assert response.json() == {"message": "Welcome to OmniLLM!"}
        self.logger.info("Root endpoint test completed successfully")

class TestModelEndpoints(BaseTest):
    def test_list_models(self):
        self.logger.info("Test List Models")
        response = self.client.get("/v1/models")
        self.logger.debug(f"Test List Models: {response.json()}")
        assert response.status_code == 200
        assert "models" in response.json()
        self.logger.info("List models endpoint test completed successfully")
