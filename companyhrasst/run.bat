@echo off
echo ==================================================
echo Launching Acme Corp HR Policy Assistant RAG App...
echo ==================================================

cd /d "%~dp0"

echo Starting Streamlit...
streamlit run app.py

pause
