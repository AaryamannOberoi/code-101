"""Shape tests against mock_api.json, one match round-trip, invalid input returns 422.

If the data files are present the pipeline is loaded (or built once, about 3 minutes) first,
so the shapes are checked on live numbers.
"""
import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app import main

MOCK = json.loads((Path(__file__).resolve().parents[1] / 'app' / 'mock_api.json').read_text(encoding='utf-8'))
REQ = MOCK['POST /api/match']['request']
EM, EN = chr(0x2014), chr(0x2013)  # the two long dash characters
# dict keys that are data values (roles, test names) rather than schema keys
DATA_KEYED = {'top_skills', 'role_area_profile', 'wilcoxon_p'}


@pytest.fixture(scope='module')
def client():
    main.start_pipeline(blocking=True)
    with TestClient(main.app) as c:
        yield c


def same_shape(mock, live, path='$'):
    if isinstance(mock, bool):
        assert isinstance(live, bool), f'{path}: expected bool, got {type(live).__name__}'
    elif isinstance(mock, (int, float)):
        assert isinstance(live, (int, float)) and not isinstance(live, bool), f'{path}: expected number, got {live!r}'
    elif isinstance(mock, str):
        assert isinstance(live, str), f'{path}: expected str, got {type(live).__name__}'
    elif isinstance(mock, list):
        assert isinstance(live, list), f'{path}: expected list'
        if mock and live:
            for i, item in enumerate(live):
                same_shape(mock[min(i, len(mock) - 1)], item, f'{path}[{i}]')
    elif isinstance(mock, dict):
        assert isinstance(live, dict), f'{path}: expected object'
        missing = set(mock) - set(live)
        assert not missing, f'{path}: missing keys {sorted(missing)}'
        for k in mock:
            same_shape(mock[k], live[k], f'{path}.{k}')
        if path.split('.')[-1] in DATA_KEYED:
            assert set(live) == set(mock), f'{path}: keys differ'


@pytest.mark.parametrize('route', ['/api/meta', '/api/insights', '/api/evaluation', '/api/transparency'])
def test_get_shapes(client, route):
    r = client.get(route)
    assert r.status_code == 200
    same_shape(MOCK[f'GET {route}'], r.json(), route)


def test_health(client):
    body = client.get('/api/health').json()
    assert {'status', 'pipeline_ready', 'built_at'} <= set(body)


def test_match_round_trip(client):
    r = client.post('/api/match', json=REQ)
    assert r.status_code == 200, r.text
    body = r.json()
    same_shape(MOCK['POST /api/match']['response'], body, 'match')
    members = body['group']['members']
    assert sum(m['is_user'] for m in members) == 1
    assert sum(m['accountability_lead'] for m in members) == 1
    assert 3 <= len(body['group']['explanation']) <= 5
    assert [w['week'] for w in body['plan']['weeks']] == list(range(1, 9))


def test_match_does_not_mutate_pool(client):
    if not client.get('/api/health').json()['pipeline_ready']:
        pytest.skip('pipeline not built (data files missing)')
    before = client.get('/api/pool').json()
    client.post('/api/match', json=REQ)
    assert client.get('/api/pool').json() == before


@pytest.mark.parametrize('patch', [
    {'target_role': 'Astronaut'},
    {'hours_per_week': 0},
    {'hours_per_week': 41},
    {'skills': {**REQ['skills'], 'coding': 6}},
    {'skills': {**REQ['skills'], 'coding': 0.5}},
    {'tipi': [4] * 9},
    {'tipi': [4] * 9 + [8]},
    {'tipi': [4] * 9 + [3.5]},
])
def test_invalid_input_returns_422(client, patch):
    r = client.post('/api/match', json={**REQ, **patch})
    assert r.status_code == 422
    assert 'message' in r.json()


def test_no_long_dashes_in_responses(client):
    for route in ['/api/meta', '/api/insights', '/api/evaluation', '/api/transparency']:
        text = client.get(route).text
        assert EM not in text and EN not in text, route
    assert EM not in client.post('/api/match', json=REQ).text
