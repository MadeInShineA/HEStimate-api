import importlib
import sys
import types
import pytest
from fastapi.testclient import TestClient


def load_app_with_fake_deepface(
    monkeypatch, faces=None, verify_impl=None, boom_in_extract=False
):
    '''
    Inject a fake `deepface` module BEFORE importing main.py so tests run
    without the real DeepFace dependency.
    '''
    fake_mod = types.ModuleType('deepface')

    class _DeepFace:
        @staticmethod
        def extract_faces(img_path, anti_spoofing):
            if boom_in_extract:
                raise RuntimeError('unexpected')
            assert anti_spoofing is True
            assert isinstance(img_path, str)
            return [{'is_real': True}] if faces is None else faces

        @staticmethod
        def verify(img1_path, image2):
            if verify_impl is not None:
                return verify_impl(img1_path, image2)
            return {'verified': True}

    fake_mod.DeepFace = _DeepFace
    monkeypatch.setitem(sys.modules, 'deepface', fake_mod)

    import main

    importlib.reload(main)
    return main.app, main


# ---------- root
def test_root_returns_welcome(monkeypatch):
    app, _ = load_app_with_fake_deepface(monkeypatch)
    client = TestClient(app)

    resp = client.get('/')
    assert resp.status_code == 200
    assert resp.json() == {
        'message': 'Welcome to the HEStimate API, please check /docs for the doc'
    }


# ---------- /verify (JSON body: {'image': '...'} )
@pytest.mark.parametrize(
    'faces, expected_status, expected_json',
    [
        (
            [{'is_real': True}, {'is_real': True}],
            400,
            {'detail': 'Please take a picture with exactly 1 person'},
        ),
        ([], 400, {'detail': 'Please take a picture with exactly 1 person'}),
        (
            [{'is_real': False}],
            400,
            {'detail': 'Spoofing detected. Please provide a real face image.'},
        ),
        (
            [{'is_real': None}],
            400,
            {'detail': 'Anti-spoofing check unavailable for this image.'},
        ),
        (
            [{'is_real': True}],
            200,
            {'success': True, 'message': 'Face verified and appears real'},
        ),
    ],
)
def test_verify_all_responses(monkeypatch, faces, expected_status, expected_json):
    app, _ = load_app_with_fake_deepface(monkeypatch, faces=faces)
    client = TestClient(app)

    resp = client.post('/verify', json={'image': 'tests/images/simon.jpg'})
    assert resp.status_code == expected_status
    assert resp.json() == expected_json


def test_verify_unexpected_exception_returns_500(monkeypatch):
    app, _ = load_app_with_fake_deepface(monkeypatch, boom_in_extract=True)
    # Important: don't re-raise server exceptions; let us assert the 500 response instead
    client = TestClient(app, raise_server_exceptions=False)

    resp = client.post('/verify', json={'image': 'tests/images/simon.jpg'})
    assert resp.status_code == 500
    # Response body can vary (detail, stacktrace in debug, etc.), so we don't assert JSON here.


# ---------- /compare (JSON body: {'image1': '...', 'image2': '...'} )
def test_compare_returns_true(monkeypatch):
    def fake_verify(img1_path, image2):
        return {'verified': True}

    app, _ = load_app_with_fake_deepface(monkeypatch, verify_impl=fake_verify)
    client = TestClient(app)

    resp = client.post(
        '/compare', json={'image1': 'tests/images/simon.jpg', 'image2': 'tests/images/olivier.jpg'}
    )
    assert resp.status_code == 200
    assert resp.json() is True


def test_compare_returns_false(monkeypatch):
    def fake_verify(img1_path, image2):
        return {'verified': False}

    app, _ = load_app_with_fake_deepface(monkeypatch, verify_impl=fake_verify)
    client = TestClient(app)

    resp = client.post(
        '/compare', json={'image1': 'tests/images/simon.jpg', 'image2': 'tests/images/olivier2.jpg'}
    )
    assert resp.status_code == 200
    assert resp.json() is False


def test_compare_handles_error(monkeypatch):
    def fake_verify(img1_path, image2):
        raise ValueError('bad input')

    app, _ = load_app_with_fake_deepface(monkeypatch, verify_impl=fake_verify)
    client = TestClient(app)

    resp = client.post(
        '/compare', json={'image1': 'tests/images/simon.jpg', 'image2': 'tests/images/olivier.jpg'}
    )
    assert resp.status_code == 400
    assert resp.json() == {'detail': 'Comparison failed: bad input'}
