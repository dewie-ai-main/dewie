# E2E test suite — tests the full request/response cycle through the FastAPI app.
#
# All tests use TestClient (sync) backed by mocked storage (fakeredis + AsyncMock).
# No live services (DB, Redis, LLM) are required.
#
# Run with:
#   pytest tests/e2e/ -v -s
