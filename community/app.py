"""Community Cloud and portable CPU entrypoint for the complete Large application."""
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ['SCHOLARRAG_CLOUD'] = '1'
for name, relative in {'PIP_CACHE_DIR': '.cache/pip', 'HF_HOME': '.cache/huggingface',
                       'TORCH_HOME': '.cache/torch', 'TMPDIR': '.cache/tmp'}.items():
    path = ROOT / relative
    path.mkdir(parents=True, exist_ok=True)
    os.environ[name] = str(path)
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ['HF_HUB_OFFLINE'] = '1'

import streamlit as st
from src.runtime_download import ensure_runtime

try:
    # Top-level Streamlit secrets also become environment variables. Access secrets
    # before creating the client; never copy them to a file or print them.
    try:
        for key in ('DEEPSEEK_API_KEY', 'DEEPSEEK_BASE_URL', 'DEEPSEEK_MODEL',
                    'DEEPSEEK_MAX_TOKENS', 'DEEPSEEK_TEMPERATURE', 'DEEPSEEK_TIMEOUT'):
            if key in st.secrets:
                os.environ[key] = str(st.secrets[key])
    except FileNotFoundError:
        pass

    @st.cache_resource(show_spinner=False)
    def prepare_runtime():
        message = st.empty()
        try:
            ensure_runtime(progress=message.info)
        finally:
            message.empty()

    prepare_runtime()
except Exception as exc:
    st.error(f'Full runtime preparation failed ({type(exc).__name__}).')
    st.info('Confirm that runtime-v1 Release assets exist and are publicly downloadable. '
            'An incomplete download will not be used. Refresh to retry.')
    st.stop()

# Import only after the download UI; retain original navigation, prompts and answers.
import importlib.util
spec = importlib.util.spec_from_file_location('scholarrag_web', ROOT / 'app.py')
web = importlib.util.module_from_spec(spec)
spec.loader.exec_module(web)
web.render_app()
