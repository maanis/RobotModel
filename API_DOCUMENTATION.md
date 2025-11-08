# TIMSCDR Chatbot API Documentation

Complete API reference for the TIMSCDR Chatbot system.

## Base URL

```
http://localhost:8000
```

For production/remote access, use your deployed server URL or ngrok tunnel.

## Authentication

Currently, the API does not require authentication. The Google Gemini API key is configured server-side.

---

## Endpoints

### 1. Get Answer

Retrieve an AI-generated answer based on the provided question using context from FAISS indexes.

**Endpoint:** `POST /get-answer`

**Request Body:**

```json
{
  "question": "string"
}
```

**Parameters:**

| Field    | Type   | Required | Description                     |
| -------- | ------ | -------- | ------------------------------- |
| question | string | Yes      | The question to ask the chatbot |

**Example Request:**

```bash
curl -X POST "http://localhost:8000/get-answer" \
  -H "Content-Type: application/json" \
  -d '{
    "question": "How many faculty members are there?"
  }'
```

**Example Response:**

```json
{
  "answer": "Based on the faculty details, there are 15 faculty members at TIMSCDR across various departments including Management, IT, and Commerce."
}
```

**Response Codes:**

- `200 OK`: Successful response
- `404 Not Found`: No relevant data found for the question
- `500 Internal Server Error`: Server or API error

**Notes:**

- Searches across all available FAISS indexes automatically
- Retrieves top 3 relevant chunks per index
- Uses Google Gemini 2.0 Flash for response generation
- Handles greetings and unrelated questions appropriately

---

### 2. Ingest Text Data

Ingest and index custom text data for future queries.

**Endpoint:** `POST /ingest`

**Request Body:**

```json
{
  "id": "string",
  "text": "string"
}
```

**Parameters:**

| Field | Type   | Required | Description                            |
| ----- | ------ | -------- | -------------------------------------- |
| id    | string | Yes      | Unique identifier for the meeting/data |
| text  | string | Yes      | Text content to be indexed             |

**Example Request:**

```bash
curl -X POST "http://localhost:8000/ingest" \
  -H "Content-Type: application/json" \
  -d '{
    "id": "orientation_2024",
    "text": "The orientation program for new students will be held on January 15, 2024. All first-year students are required to attend."
  }'
```

**Example Response:**

```json
{
  "meeting_id": "orientation_2024",
  "num_vectors": 3
}
```

**Response Codes:**

- `200 OK`: Successfully ingested and indexed
- `500 Internal Server Error`: Processing error

**Processing Details:**

- Text is automatically chunked using the `chunk_text` function
- Default chunk size: 500 characters
- Default overlap: 50 characters
- Creates FAISS index at `faiss_indexes/{id}.index`
- Saves metadata at `faiss_indexes/{id}_chunks.json`

---

### 3. Health Check

Check the server status and resource utilization.

**Endpoint:** `GET /health`

**Parameters:** None

**Example Request:**

```bash
curl "http://localhost:8000/health"
```

**Example Response:**

```json
{
  "status": "healthy",
  "device": "cuda",
  "cuda_available": true,
  "gpu_memory_allocated": 245.5,
  "gpu_memory_reserved": 512.0
}
```

**Response Fields:**

| Field                | Type    | Description                   |
| -------------------- | ------- | ----------------------------- |
| status               | string  | Server health status          |
| device               | string  | Device being used (cuda/cpu)  |
| cuda_available       | boolean | Whether CUDA GPU is available |
| gpu_memory_allocated | number  | GPU memory allocated in MB    |
| gpu_memory_reserved  | number  | GPU memory reserved in MB     |

**Response Codes:**

- `200 OK`: Server is healthy

---

## Data Models

### Query

Used in `/get-answer` endpoint.

```python
class Query(BaseModel):
    question: str
```

### IngestRequest

Used in `/ingest` endpoint.

```python
class IngestRequest(BaseModel):
    id: str
    text: str
```

---

## Error Handling

### Error Response Format

```json
{
  "detail": "Error description"
}
```

### Common Errors

#### 404 Not Found

```json
{
  "detail": "No relevant data found"
}
```

**Cause:** No matching data found in FAISS indexes for the query

#### 500 Internal Server Error

```json
{
  "detail": "Gemini API error: 429"
}
```

**Cause:** Google Gemini API error (rate limit, invalid key, etc.)

```json
{
  "detail": "CUDA out of memory"
}
```

**Cause:** GPU memory exhausted

---

## FAISS Index Structure

### Index Files

Each data source creates two files in `faiss_indexes/`:

1. **`{meeting_id}.index`**: FAISS vector index (binary format)
2. **`{meeting_id}_chunks.json`**: Text chunks metadata (JSON array)

### Pre-built Indexes

| Index ID             | Source File         | Description         |
| -------------------- | ------------------- | ------------------- |
| faculty_details      | faculty_details.csv | Faculty information |
| timscdr_csv          | TIMSCDR.csv         | Institutional data  |
| timscdr_full_profile | timscdr_data.json   | Complete profile    |

---

## Usage Examples

### Python

```python
import requests

# Ask a question
response = requests.post(
    "http://localhost:8000/get-answer",
    json={"question": "What courses does TIMSCDR offer?"}
)
print(response.json()["answer"])

# Ingest new data
response = requests.post(
    "http://localhost:8000/ingest",
    json={
        "id": "new_announcement",
        "text": "Important announcement about campus activities..."
    }
)
print(f"Indexed {response.json()['num_vectors']} chunks")

# Check health
response = requests.get("http://localhost:8000/health")
print(response.json())
```

### JavaScript (fetch)

```javascript
// Ask a question
fetch("http://localhost:8000/get-answer", {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify({
    question: "Tell me about the faculty",
  }),
})
  .then((res) => res.json())
  .then((data) => console.log(data.answer));

// Ingest data
fetch("http://localhost:8000/ingest", {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify({
    id: "test_data",
    text: "Sample text for indexing",
  }),
})
  .then((res) => res.json())
  .then((data) => console.log(`Indexed ${data.num_vectors} chunks`));

// Check health
fetch("http://localhost:8000/health")
  .then((res) => res.json())
  .then((data) => console.log(data));
```

### React Native (Mobile App)

```javascript
// From chatbot-app/App.js
const sendMessage = async (inputText) => {
  try {
    const response = await fetch("http://10.175.49.43:8000/get-answer", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ question: inputText }),
    });
    const data = await response.json();
    return data.answer;
  } catch (error) {
    console.error("Error:", error);
    throw error;
  }
};
```

### cURL

```bash
# Ask a question
curl -X POST "http://localhost:8000/get-answer" \
  -H "Content-Type: application/json" \
  -d '{"question": "What is TIMSCDR?"}'

# Ingest data
curl -X POST "http://localhost:8000/ingest" \
  -H "Content-Type: application/json" \
  -d '{"id": "test_data", "text": "Sample text for indexing"}'

# Health check
curl "http://localhost:8000/health"
```

---

## Performance Considerations

### Response Time

- **Typical response**: 1-3 seconds
- **Factors affecting speed**:
  - Number of indexed documents
  - Query complexity
  - Gemini API response time
  - GPU availability

### Rate Limits

- **FAISS search**: No inherent limits (depends on hardware)
- **Gemini API**: Subject to Google's API quotas
- **Server capacity**: Depends on hardware resources

### Optimization Tips

1. **Use GPU** for faster embedding generation
2. Keep indexes under 100k vectors for optimal performance
3. Monitor memory with `/health` endpoint
4. Implement caching for frequently asked questions
5. Use connection pooling for database operations

---

## Chatbot Behavior

### Question Handling

The chatbot follows these rules (defined in the `get_answer` prompt):

1. **Greetings**: Responds warmly and introduces itself as TIMSCDR Bot
2. **TIMSCDR-related**: Answers based on indexed context
3. **Inference**: Can derive answers from context (counting, listing, summarizing)
4. **Unrelated questions**: Politely redirects to TIMSCDR topics
5. **No context**: Returns "I'm sorry, I can't answer that, ask me questions about TIMSCDR."

### Prompt Structure

The AI prompt includes:

- Behavior guidelines (be polite, conversational)
- Context from FAISS search
- User question
- Instructions for handling different types of queries
- Response format expectations

### Example Conversations

**Greeting:**

```
User: Hi
Bot: Hello! I'm the TIMSCDR Bot. I'm here to help you with any questions about Thakur Institute of Management Studies, Career Development & Research. How can I assist you today?
```

**TIMSCDR Query:**

```
User: How many faculty members are there?
Bot: Based on the faculty details, TIMSCDR has 15 faculty members across various departments including Management, IT, and Commerce.
```

**Unrelated Query:**

```
User: What's the weather today?
Bot: I'm sorry, I can't answer that, ask me questions about TIMSCDR.
```

---

## Integration Guide

### Mobile App Integration

The React Native app in `chatbot-app/` connects to this API:

1. Update the API URL in `App.js`:

   ```javascript
   const API_URL = "http://YOUR_IP:8000";
   ```

2. For Android emulator, use: `http://10.0.2.2:8000`
3. For physical devices, use your computer's IP: `http://192.168.x.x:8000`
4. For production, use ngrok or deploy to a server

### Web Integration

```html
<!DOCTYPE html>
<html>
  <head>
    <title>TIMSCDR Chatbot</title>
  </head>
  <body>
    <input type="text" id="question" placeholder="Ask a question..." />
    <button onclick="askQuestion()">Send</button>
    <div id="answer"></div>

    <script>
      async function askQuestion() {
        const question = document.getElementById("question").value;
        const response = await fetch("http://localhost:8000/get-answer", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ question }),
        });
        const data = await response.json();
        document.getElementById("answer").innerText = data.answer;
      }
    </script>
  </body>
</html>
```

---

## Troubleshooting

### No relevant data found (404)

**Solutions:**

1. Verify FAISS indexes exist in `faiss_indexes/` directory
2. Run ingestion scripts: `python ingest_csv.py`, `python ingest_faculty.py`
3. Check if the question is related to indexed data
4. Ensure index files are not corrupted

### Gemini API errors (500)

**Solutions:**

1. Verify API key is correct in `app.py` (line ~164)
2. Check Google Cloud API quota at [console.cloud.google.com](https://console.cloud.google.com)
3. Verify internet connectivity
4. Check if Gemini API is enabled in your Google Cloud project

### GPU memory errors

**Solutions:**

1. Switch to CPU: Set `device="cpu"` in model initialization
2. Reduce batch size in ingestion scripts
3. Clear GPU cache: `torch.cuda.empty_cache()`
4. Monitor GPU usage with `nvidia-smi` command

### Connection refused (from mobile app)

**Solutions:**

1. Ensure backend server is running: `python app.py`
2. Check firewall settings allow port 8000
3. Verify IP address is correct (use `ipconfig` on Windows or `ifconfig` on Linux/Mac)
4. For Android emulator, use `10.0.2.2` instead of `localhost`
5. Ensure both devices are on the same network

---

## Security Considerations

### API Key Protection

- Never commit API keys to version control
- Use environment variables or `.env` files
- Rotate API keys regularly

### Rate Limiting

Consider implementing rate limiting to prevent abuse:

```python
from slowapi import Limiter
from slowapi.util import get_remote_address

limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter

@app.post("/get-answer")
@limiter.limit("10/minute")
def get_answer(request: Request, query: Query):
    # ... existing code
```

### CORS Configuration

For web applications, configure CORS:

```python
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Update with specific origins in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

---

## Changelog

### Version 1.0 (Current)

- Initial release with FAISS vector search
- Google Gemini 2.0 Flash integration
- Multiple data source support
- Performance monitoring with GPU tracking
- Health check endpoint
- React Native mobile app support

---

## Future Enhancements

Planned features:

- User authentication and session management
- Conversation history tracking
- Multi-language support
- Voice input/output
- Document upload for dynamic ingestion
- Admin dashboard for managing indexes
- Analytics and usage statistics

---

## Support

For issues or questions:

- Check [README.md](README.md) for setup instructions
- Review error messages in server logs
- Monitor GPU/memory with `/health` endpoint
- Open an issue on GitHub: [https://github.com/maanis/RobotModel/issues](https://github.com/maanis/RobotModel/issues)

---

## API Testing Tools

### Postman Collection

Import this collection to test all endpoints:

```json
{
  "info": {
    "name": "TIMSCDR Chatbot API",
    "schema": "https://schema.getpostman.com/json/collection/v2.1.0/collection.json"
  },
  "item": [
    {
      "name": "Get Answer",
      "request": {
        "method": "POST",
        "header": [{ "key": "Content-Type", "value": "application/json" }],
        "body": {
          "mode": "raw",
          "raw": "{\"question\": \"What is TIMSCDR?\"}"
        },
        "url": { "raw": "http://localhost:8000/get-answer" }
      }
    },
    {
      "name": "Health Check",
      "request": {
        "method": "GET",
        "url": { "raw": "http://localhost:8000/health" }
      }
    }
  ]
}
```

### Interactive API Documentation

FastAPI provides automatic interactive documentation:

- **Swagger UI**: [http://localhost:8000/docs](http://localhost:8000/docs)
- **ReDoc**: [http://localhost:8000/redoc](http://localhost:8000/redoc)

---

**Last Updated:** November 8, 2025
