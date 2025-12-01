from fastapi.testclient import TestClient


def test_read_root_status_code(client: TestClient):
    """
    Tests the read_root response status_code
    """

    resp = client.get("/")
    expected_status_code = 200

    assert resp.status_code == expected_status_code


def test_read_root_content(client: TestClient):
    """
    Tests the read_root response message contents
    """

    resp = client.get("/")
    expected_message = {"message": "Hello from FastAPI!"}

    assert resp.json() == expected_message
