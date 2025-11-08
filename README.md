# TIMSCDR Chatbot API

A FastAPI-based chatbot system for TIMSCDR (Thakur Institute of Management Studies, Career Development & Research) that uses FAISS vector indexing and Google's Gemini AI for intelligent question answering.

## Features

- **Vector Search**: Uses FAISS for efficient similarity search across multiple data sources
- **Multiple Data Sources**: Supports faculty details, CSV data, and custom meeting transcripts
- **AI-Powered Responses**: Integrates with Google Gemini 2.0 Flash for natural language responses
- **Performance Monitoring**: Built-in GPU/CPU monitoring and performance tracking
- **Flexible Data Ingestion**: Support for text chunks and CSV file ingestion

## Project Structure

```
.
├── app.py                          # Main FastAPI application
├── ingest_csv.py                   # CSV data ingestion for TIMSCDR data
├── ingest_faculty.py               # Faculty details ingestion
├── ingest_script.py                # General text ingestion script
├── requirements.txt                # Python dependencies
├── .env                            # Environment variables (API keys)
├── faculty_details.csv             # Faculty information
├── TIMSCDR.csv                     # TIMSCDR institutional data
├── timscdr_data.json              # Additional TIMSCDR data
└── faiss_indexes/                  # FAISS vector indexes and metadata
    ├── faculty_details.index
    ├── faculty_details_chunks.json
    ├── timscdr_csv.index
    ├── timscdr_csv_chunks.json
    ├── timscdr_full_profile.index
    └── timscdr_full_profile_chunks.json
```

## Prerequisites

- Python 3.8+
- CUDA-compatible GPU (optional, but recommended for better performance)
- Google Gemini API Key

## Installation

1. **Clone the repository**

   ```bash
   git clone https://github.com/maanis/RobotModel.git
   cd RobotModel
   ```

2. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables**
   Create a `.env` file with your Gemini API key:

   ```
   GEMINI_API_KEY=your_api_key_here
   ```

4. **Prepare data indexes**
   Run the ingestion scripts to create FAISS indexes:
   ```bash
   python ingest_csv.py
   python ingest_faculty.py
   python ingest_script.py
   ```

## Usage

### Starting the Server

```bash
python app.py
```

The server will start on `http://localhost:8000`

### API Endpoints

See [API_DOCUMENTATION.md](API_DOCUMENTATION.md) for detailed endpoint documentation.

## Key Components

### 1. Data Ingestion

- `ingest_csv.py`: Processes `TIMSCDR.csv` and creates vector embeddings
- `ingest_faculty.py`: Processes `faculty_details.csv`
- `ingest_script.py`: Processes `timscdr_data.json`

### 2. Vector Search

Uses FAISS (Facebook AI Similarity Search) with:

- Sentence transformers model: `all-MiniLM-L6-v2`
- IndexFlatL2 for similarity search
- Top-3 relevant chunks retrieval

### 3. Response Generation

- Retrieves relevant context from FAISS indexes
- Constructs prompts with context and user questions
- Uses Google Gemini 2.0 Flash API for natural language responses

### 4. Performance Monitoring

The `monitor_performance` decorator tracks:

- Execution time
- Memory usage (RAM)
- GPU memory allocation (if available)

## Configuration

### GPU Usage

The application is configured to use GPU if available:

- Set in `app.py` with `device = "cuda" if torch.cuda.is_available() else "cpu"`
- For CPU-only usage, models are initialized with `device="cpu"`

### Chunking Parameters

Text chunking can be customized in `chunk_text` function:

- `chunk_size`: Default 500 characters
- `overlap`: Default 50 characters

## Data Sources

The chatbot searches across multiple indexed sources:

1. **Faculty Details** (`faculty_details`)

   - Faculty names, designations, and details

2. **TIMSCDR CSV** (`timscdr_csv`)

   - Institutional data with categories, keywords, and responses

3. **TIMSCDR Full Profile** (`timscdr_full_profile`)
   - Comprehensive institutional information

## API Response Format

All successful responses follow this format:

```json
{
  "answer": "Response text from Gemini AI"
}
```

Error responses:

```json
{
  "detail": "Error message"
}
```

## Health Monitoring

Check system health and GPU status:

```bash
curl http://localhost:8000/health
```

## Development

### Adding New Data Sources

1. Create a new ingestion script following the pattern in `ingest_csv.py`
2. Generate embeddings using the sentence transformer model
3. Save the FAISS index and chunks to `faiss_indexes/`
4. The `get_answer` endpoint will automatically detect and use the new index

### Customizing the AI Prompt

Modify the prompt template in the `get_answer` function to adjust the chatbot's behavior and response style.

## Performance Tips

1. **GPU Acceleration**: Use CUDA-compatible GPU for faster embeddings
2. **Batch Processing**: Process multiple queries in batches for better throughput
3. **Index Optimization**: Use IVF indexes for larger datasets (>100k vectors)
4. **Memory Management**: Monitor memory usage with the `/health` endpoint

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**

   - Reduce batch size
   - Use CPU instead: set `device="cpu"` in model initialization

2. **Missing Indexes**

   - Run all ingestion scripts before starting the server
   - Check `faiss_indexes/` directory exists

3. **Gemini API Errors**
   - Verify API key in `.env` file
   - Check API quota and rate limits

## Mobile Application

This project includes a React Native mobile application for Android/iOS:

- Location: `chatbot-app/` directory
- Built with Expo
- Connects to the FastAPI backend
- See mobile app README for setup instructions

## License

MIT License

## Contributors

Developed by maanis

## Support

For issues and questions, please open an issue on GitHub at [https://github.com/maanis/RobotModel](https://github.com/maanis/RobotModel).
