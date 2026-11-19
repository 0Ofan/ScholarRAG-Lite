"""One real Streamlit Search & Answer integration check without saving answer text."""
import json
from pathlib import Path

from streamlit.testing.v1 import AppTest

from src.rag_service import validate_citations

ROOT = Path(__file__).resolve().parents[1]
app = AppTest.from_file(str(ROOT / 'app.py'), default_timeout=120).run()
app.text_input[0].set_value('What evidence links COVID-19 infection to respiratory symptoms?')
app.button[0].click().run()
if 'last_answer' not in app.session_state:
    print('widgets:', [(x.label, x.disabled) for x in app.button])
    print('exceptions:', [str(x.message) for x in app.exception])
    raise RuntimeError('Search & Answer form did not submit')
state = app.session_state['last_answer']
generation = state['generation']
result = {
    'version': 'phase35-streamlit-ui-smoke-v1',
    'real_api': generation is not None,
    'page_exceptions': len(app.exception),
    'answer_characters': len(state['answer']),
    'evidence_count': len(state['evidence']),
    'retrieval': state['timing'],
    'generation': generation,
    'citation_check': validate_citations(state['answer'], state['evidence']),
}
out = ROOT / 'experiments/results/phase35'
out.mkdir(parents=True, exist_ok=True)
(out / 'ui_smoke.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf8')
print(json.dumps(result, ensure_ascii=False, indent=2))
