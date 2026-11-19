"""Shared-password gate for the public Community Cloud entrypoint."""
import hashlib
import hmac
import os


def require_password(st):
    try:
        password = str(st.secrets.get('APP_PASSWORD', os.getenv('APP_PASSWORD', '')))
    except FileNotFoundError:
        password = os.getenv('APP_PASSWORD', '')
    if not password.strip():
        st.title('ScholarRAG Lite')
        st.error('管理员尚未配置访问密码。请在 Streamlit Secrets 中设置 APP_PASSWORD。')
        st.stop()

    fingerprint = hashlib.sha256(password.encode('utf-8')).hexdigest()
    proof = st.session_state.get('_access_granted', '')
    if isinstance(proof, str) and hmac.compare_digest(proof, fingerprint):
        return
    st.session_state.pop('_access_granted', None)

    def check_password():
        candidate = st.session_state.pop('_access_password', '')
        if hmac.compare_digest(str(candidate).encode('utf-8'), password.encode('utf-8')):
            st.session_state['_access_granted'] = fingerprint
            st.session_state.pop('_access_error', None)
        else:
            st.session_state.pop('_access_granted', None)
            st.session_state['_access_error'] = True

    st.title('ScholarRAG Lite')
    st.caption('请输入访问密码以使用研究助手。')
    with st.form('access_login'):
        st.text_input('访问密码', type='password', key='_access_password')
        st.form_submit_button('进入系统', on_click=check_password)
    if st.session_state.get('_access_error'):
        st.error('密码不正确，请重试。')
    st.stop()
