import logging
import streamlit as st


def main() -> None:
    logging.info("Starting execution")
    st.switch_page("pages/dashboard_page.py")


if __name__ == "__main__":
    main()
