# pages/5_Old_Version.py
import streamlit as st

st.set_page_config(page_title="Legacy Version", layout="wide")


def old_version():
    st.title("🔄 Legacy Version")
    st.warning("This is the original combined interface. Use the new version for better security.")

    col1, col2 = st.columns([1, 4])
    with col1:
        if st.button("← Back to New Version"):
            st.switch_page("main.py")

    with col2:
        st.info("This shows your original app where students could see teacher options.")

    # Import and run your original main_old.py
    from main_old import run_app
    run_app()


if __name__ == "__main__":
    old_version()