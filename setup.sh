# make sure uv is installed
uv -V

cd ./setup
uv sync

cd ../analysis
uv sync
uv pip install -e . --no-deps

cd ../experiment-ANNs
uv sync

cd ../experiment-Lab
uv sync
