# LARMS (Large Language Models for Remedying Mental Status)

LARMS is an AI-powered mental health support chatbot that uses advanced language models and semantic similarity to provide empathetic and contextually relevant responses to users expressing mental health concerns.
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/release/python-380/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
## Overview

The system combines:
- Large Language Model (LLama 3.2) for generating responses
- Sentence transformers for semantic similarity matching
- A curated dataset of mental health conversations
- Streamlit for the user interface

## Features

- Real-time conversation with an AI mental health assistant
- Semantic similarity matching with previous conversation contexts
- Adjustable temperature settings for response generation
- Experiment mode for viewing system's decision-making process
- Conversation history tracking
- User-friendly web interface

## Prerequisites

- Python 3.8+
- Groq API key
- Required Python packages (see `requirements.txt`)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/narensen/LARMS-1.2.git
cd LARMS
```

2. Install required packages:
```bash
pip install -r requirements.txt
```

3. Create a `corpus` folder in the project directory:
```bash
mkdir corpus
```

4. **Important**: Before running the main application, you must first run the data merger script:
```bash
cd corpus
python merge_data.py
```
This will generate the necessary `merged_dataset.csv` file that LARMS requires to function.

## Configuration

1. Replace the Groq API key in `LARMS.py` with your own:
```python
groq_api_key = "your_api_key_here"
```

2. Ensure the `corpus` folder contains:
- `merge_data.py`
- Your source data files
- After running `merge_data.py`, it will contain:
  - `merged_dataset.csv`
  - `embeddings.pt` (generated on first run)

## Usage

1. Start the Streamlit application:
```bash
streamlit run LARMS.py
```

2. Access the web interface at `http://localhost:8501`

3. Use the sidebar to:
   - Adjust the temperature (0.0 - 1.0)
   - Toggle experiment mode
   - View current settings

4. Enter your message in the text area and receive AI-generated responses

## Features Explanation

### Temperature Setting
- Controls randomness in AI responses
- Lower values (0.0-0.4): More focused, consistent responses
- Higher values (0.5-1.0): More creative, varied responses

### Experiment Mode
When enabled, shows:
- Similar context from database
- Suggested response
- Similarity score
- Current temperature setting

### Conversation History
- Maintains a record of all interactions
- Accessible via expandable section in the interface

## Project Structure

```
LARMS/
├── LARMS.py              # Main application file
├── requirements.txt      # Python dependencies
├── README.md            # This file
└── corpus/
    ├── merge_data.py    # Data preparation script
    ├── merged_dataset.csv # Generated dataset
    └── embeddings.pt    # Generated embeddings
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.


## Acknowledgments

- This project uses the Groq API for language model inference
- Sentence transformers for semantic similarity
- Streamlit for the web interface
