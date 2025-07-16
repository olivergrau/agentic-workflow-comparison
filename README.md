# Agentic Workflow Comparison

This repository contains various sample agents demonstrating how to interact with the OpenAI API.

## Configuration

Create a `.env` file in the project root with at least the following variables:

```
OPENAI_API_KEY=your-key
# Optional alternative base URL
OPENAI_BASE_URL=https://openai.vocareum.com/v1
```

Use the helper functions from `config.py` to access these values:

```python
from config import load_openai_api_key, load_openai_base_url

api_key = load_openai_api_key()
base_url = load_openai_base_url()
```

All modules and tests in this repo import these functions instead of calling `load_dotenv()` directly.
