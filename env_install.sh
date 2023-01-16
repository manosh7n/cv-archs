python3.9 -m venv cv_archs_env && \
source ./cv_archs_env/bin/activate && \
pip install -U pip setuptools wheel && \
pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cu116 -r ./cv-archs/requirements.txt && \
pip cache purge && \
python -m ipykernel install --name="cv_archs" --user