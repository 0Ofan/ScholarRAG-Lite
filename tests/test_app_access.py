import os
import unittest
from contextlib import nullcontext
from pathlib import Path
from unittest.mock import Mock, patch

from src.app_access import require_password


class Stopped(Exception):
    pass


def page(password=None):
    st = Mock()
    st.secrets = {} if password is None else {'APP_PASSWORD': password}
    st.session_state = {}
    st.form.return_value = nullcontext()
    st.stop.side_effect = Stopped
    return st


class AccessTests(unittest.TestCase):
    @patch.dict(os.environ, {'APP_PASSWORD': ''})
    def test_missing_password_denies_access(self):
        st = page()
        with self.assertRaises(Stopped):
            require_password(st)
        st.error.assert_called_once()
        st.text_input.assert_not_called()

    def submit(self, st, candidate):
        with self.assertRaises(Stopped):
            require_password(st)
        st.session_state['_access_password'] = candidate
        st.form_submit_button.call_args.kwargs['on_click']()

    def test_correct_password_allows_only_this_session_and_removes_input(self):
        st = page('示例-only-for-test')
        self.submit(st, '示例-only-for-test')
        self.assertNotIn('_access_password', st.session_state)
        self.assertNotIn('示例-only-for-test', st.session_state.values())
        st.stop.reset_mock()
        require_password(st)
        st.stop.assert_not_called()
        with self.assertRaises(Stopped):
            require_password(page('示例-only-for-test'))

    def test_wrong_password_does_not_grant_access(self):
        st = page('example-only-for-test')
        self.submit(st, 'wrong')
        self.assertNotIn('_access_granted', st.session_state)
        self.assertNotIn('_access_password', st.session_state)
        with self.assertRaises(Stopped):
            require_password(st)

    def test_rotation_invalidates_existing_session(self):
        st = page('old-example-only-for-test')
        self.submit(st, 'old-example-only-for-test')
        st.secrets['APP_PASSWORD'] = 'new-example-only-for-test'
        with self.assertRaises(Stopped):
            require_password(st)
        self.assertNotIn('_access_granted', st.session_state)

    def test_gate_precedes_download_and_page_import(self):
        code = (Path(__file__).resolve().parents[1] / 'community/app.py').read_text(encoding='utf-8')
        self.assertLess(code.index('require_password(st)'), code.index('prepare_runtime()'))
        self.assertLess(code.index('require_password(st)'), code.index('spec.loader.exec_module(web)'))


if __name__ == '__main__':
    unittest.main()
