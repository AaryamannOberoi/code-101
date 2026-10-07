"""SkillCircle API. Serves mock_api.json until the pipeline is ready, then live numbers."""
import threading
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse

from . import service
from .pipeline import runner
from .schemas import MatchRequest, MatchResponse

STATE = {'state': None, 'building': False, 'error': None}
_lock = threading.Lock()


def _build_in_background():
    try:
        STATE['state'] = runner.load_or_build(force=True)
    except Exception as e:  # keep serving the mock, report the error on /api/health
        STATE['error'] = repr(e)
    finally:
        STATE['building'] = False


def start_pipeline(blocking=False):
    """Load the cache (fast) or build the pipeline. Non-blocking builds run in a thread."""
    with _lock:
        if STATE['state'] is not None or STATE['building'] or not runner.data_present():
            return
        cached = runner.load_or_build(log=lambda *_: None) if blocking else _try_cache()
        if cached is not None:
            STATE['state'] = cached
            return
        STATE['building'] = True
    threading.Thread(target=_build_in_background, daemon=True).start()


def _try_cache():
    import pickle
    try:
        with open(runner.CACHE, 'rb') as f:
            st = pickle.load(f)
        return st if st.get('fingerprint') == runner._fingerprint() else None
    except Exception:
        return None


@asynccontextmanager
async def lifespan(app):
    start_pipeline()
    yield


app = FastAPI(title='SkillCircle API', version='1.0', lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=['http://localhost:3000', 'http://127.0.0.1:3000'],
                   allow_methods=['*'], allow_headers=['*'])


@app.exception_handler(RequestValidationError)
async def validation_handler(request: Request, exc: RequestValidationError):
    msgs = []
    for e in exc.errors():
        loc = '.'.join(str(x) for x in e['loc'] if x != 'body')
        msgs.append(f'{loc}: {e["msg"]}')
    return JSONResponse(status_code=422, content={'message': 'Invalid input. ' + '; '.join(msgs),
                                                  'detail': [{k: v for k, v in e.items() if k != 'ctx'} for e in exc.errors()]})


def _live():
    return STATE['state']


def _get(view, mock_key):
    st = _live()
    return view(st) if st is not None else service.MOCK[mock_key]


@app.get('/api/health')
def health():
    st = _live()
    return {'status': 'ok', 'pipeline_ready': st is not None, 'built_at': st['built_at'] if st else None,
            'mode': 'live' if st else 'mock', 'building': STATE['building'],
            'missing_data_files': runner.missing_files(), 'error': STATE['error']}


@app.get('/api/meta')
def meta():
    return _get(service.meta, 'GET /api/meta')


@app.get('/api/insights')
def insights():
    return _get(service.insights, 'GET /api/insights')


@app.get('/api/evaluation')
def evaluation():
    return _get(service.evaluation, 'GET /api/evaluation')


@app.get('/api/transparency')
def transparency():
    return _get(service.transparency, 'GET /api/transparency')


@app.post('/api/match', response_model=MatchResponse)
def match(req: MatchRequest):
    st = _live()
    if st is None:
        return service.MOCK['POST /api/match']['response']
    return service.match(st, req.model_dump())


@app.get('/api/pool')
def pool(role: Optional[str] = Query(None, description='One of the 6 core roles')):
    if role is not None and role not in service.ROLES:
        raise HTTPException(422, f'role must be one of: {", ".join(service.ROLES)}')
    st = _live()
    if st is None:
        raise HTTPException(503, 'Pipeline not ready yet: the simulated pool is built from the data files')
    return service.pool_list(st, role)


@app.get('/api/export')
def export():
    path = runner.OUT_DIR / 'results.json'
    if _live() is None or not path.exists():
        raise HTTPException(503, 'Pipeline not ready yet: results.json has not been built')
    return FileResponse(path, media_type='application/json', filename='results.json')
