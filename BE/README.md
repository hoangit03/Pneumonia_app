## ⚙️ Backend Setup (BE)

```bash
cd BE
python -m venv .venv
```

# Activate the virtual environment:

# On Windows:

```bash
.venv\Scripts\activate
```

# On Unix/macOS:

```bash
source .venv/bin/activate
```

# Update pip and install library

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

# ▶️ Run Backend
```bash
uvicorn app.main:app --host 8000 --reload
```
