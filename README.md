# ai-stock-price-forecasting

## How to run 

1. Make sure you have [uv](https://docs.astral.sh/uv/guides/install-python/) installed on your system

2. Clone the project 

```bash
git clone https://github.com/ogioldat/ai-stock-price-forecasting.git
```

3. Go to project root

```bash
cd ai-stock-price-forecasting.git
```

4. Run the API service with uv

```bash
uv run uvicorn api.src.main:app --host 0.0.0.0 --port 8000 --reload
```

You can also build a docker image using the following:


```bash 
docker image build -f api/Dockerfile . -t fastapi:1.0.0
```

5. To run tests do the following

```bash 
uv run pytest 
```


