# Music Analyser

A Python project for music analysis.

## Development Setup

### Prerequisites
- Python 3.8 or higher
- Git

### Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd music-analyser
```

2. Create and activate a virtual environment:
```bash
# Windows PowerShell
python -m venv venv
.\venv\Scripts\Activate.ps1

# Windows Command Prompt
python -m venv venv
.\venv\Scripts\activate.bat

# Linux/macOS
python -m venv venv
source venv/bin/activate
```

3. Install dependencies:
```bash
pip install -e ".[dev]"
```

### Development Commands

- Run tests: `pytest`
- Format code: `black src tests`
- Lint code: `flake8 src tests`
- Type checking: `mypy src`
- Sort imports: `isort src tests`

### Project Structure

```
music-analyser/
├── src/                    # Source code
│   ├── __init__.py
│   └── main.py
├── tests/                  # Test files
│   ├── __init__.py
│   └── test_main.py
├── venv/                   # Virtual environment (not in git)
├── .vscode/                # VS Code/Cursor settings
├── .gitignore              # Git ignore rules
├── pyproject.toml          # Project configuration
├── requirements.txt        # Production dependencies
├── requirements-dev.txt    # Development dependencies
└── README.md              # This file
```

## License

MIT License - see LICENSE file for details.
