# Executive Agent Task Scheduler

A Modal-based task scheduling system that uses Claude 3.7 Sonnet to understand natural language requests and schedule tasks with automated notifications via Zapier.

## Features

- Natural language task scheduling
- SQLite-based task persistence
- WhatsApp notifications via Zapier integration
- RESTful API endpoint for task creation
- Powered by Claude 3.7 Sonnet for intelligent task parsing

## Prerequisites

- Python 3.12+
- Modal CLI
- Anthropic API key
- Zapier webhook URL (for notifications)

## Installation

1. Install the required dependencies:
```bash
pip install -r requirements.txt
```

2. Set up Modal secrets:
```bash
modal secret create executive-agent \
  --env ZAPIER_WEBHOOK_URL=<your-zapier-webhook> \
  --env ANTHROPIC_API_KEY=<your-anthropic-key>
```

3. Create a Modal volume:
```bash
modal volume create executiveagent
```

## Usage

1. Deploy the API:
```bash
modal deploy main.py
```

2. Send requests to the API endpoint:
```python
import requests

url = "https://your-modal-endpoint.modal.run"
data = {
    "query": "Schedule a meeting for tomorrow at 3pm",
    "metadata": {
        "email": "x@x.com"
        "phone_number":"666-6666-6666"
    }
}

response = requests.post(url, json=data)
print(response.json())
```

## API Reference

### POST /query

Schedule a new task using natural language. The endpoint processes natural language queries through Claude 3.7 Sonnet and forwards task details to Zapier for notifications.

**Request Body:**
```json
{
    "query": "string",
    "metadata": {
        // Any custom metadata fields
        "email": "user@example.com",
        "phone_number": "+1234567890",
        "custom_field": "value"
    }
}
```

**Metadata Flow:**
1. When you send a request, any metadata in the request is preserved throughout the processing pipeline
2. The query is processed by the ExecutiveAgent using Claude 3.7 Sonnet
3. Once processing is complete, both the original query and metadata are forwarded to Zapier with the following structure:
```json
{
    "query": "original query string",
    "response": "agent's response",
    "timestamp": "ISO formatted UTC timestamp",
    "metadata": {
        // All original metadata fields preserved
        "email": "user@example.com",
        "phone_number": "+1234567890",
        "custom_field": "value"
    }
}
```

This allows you to:
- Pass through user contact information
- Include custom identifiers or tags
- Add any additional context needed for notifications
- Configure different Zapier workflows based on metadata values

**Response:**
```json
{
    "response": {
        "message": "string",
        "message_type": "whatsapp"
    },
    "zapier_sent": true
}
```

**Example Usage:**
```python
import requests

url = "https://your-modal-endpoint.modal.run"
data = {
    "query": "Schedule a meeting with John tomorrow at 3pm",
    "metadata": {
        "email": "john@example.com",
        "phone_number": "+1234567890",
        "meeting_room": "Conference Room A",
        "priority": "high"
    }
}

response = requests.post(url, json=data)
print(response.json())
```

The metadata is flexible and can include any JSON-serializable data that your Zapier workflow needs to process the notification appropriately.

## Project Structure

- `main.py`: Modal API endpoint and Zapier integration
- `agent.py`: Core task scheduling logic and database operations
- `test.py`: Example usage and testing

## Database Schema

Tasks are stored in SQLite with the following schema:

```sql
CREATE TABLE tasks (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    description TEXT NOT NULL,
    scheduled_time TEXT NOT NULL,
    status TEXT NOT NULL,
    created_at TEXT NOT NULL,
    completed_at TEXT
)
```

## Technical Details

### Agent Architecture

The ExecutiveAgent is built with several key components that work together to provide task scheduling and persistence:

#### 1. Core Components
- **ExecutiveAgent**: Main orchestrator that integrates all tools and manages the Claude 3.7 Sonnet model
- **SQLite Database**: Persistent storage for tasks and other data
- **Tools System**: Collection of specialized tools for different operations

#### 2. Available Tools

##### Task Scheduler Tool
- Manages task creation and persistence
- Handles task statuses (PENDING, COMPLETED, CANCELLED)
- Stores tasks with:
  ```sql
  CREATE TABLE tasks (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      description TEXT NOT NULL,
      scheduled_time TEXT NOT NULL,
      status TEXT NOT NULL,
      created_at TEXT NOT NULL,
      completed_at TEXT
  )
  ```

##### SQL Engine Tool
- Provides direct SQL query capabilities
- Returns results as pandas DataFrames
- Enables complex queries across all tables
- Maintains current database state information

##### Table Creator Tool
- Dynamically creates new tables in the SQLite database
- Supports custom column definitions
- Allows bulk data insertion
- Example usage:
  ```python
  {
      "table_name": "employees",
      "columns": [
          ["id", "INTEGER"],
          ["name", "TEXT"],
          ["salary", "REAL"]
      ],
      "data": [
          {"id": 1, "name": "John", "salary": 50000}
      ]
  }
  ```

##### Display Tables Tool
- Lists all tables in the database
- Shows table schemas and row counts
- Provides sample data for each table

### Data Flow

1. **Query Processing**:
   ```
   User Query → Claude 3.7 Sonnet → Tool Selection → Action Execution
   ```

2. **Task Creation Flow**:
   ```
   Natural Language Query → Task Parsing → Task Validation → Database Storage
   ```

3. **Data Persistence**:
   - All data is stored in SQLite at `/alldata/tasks.db`
   - Database is maintained in a Modal volume for persistence
   - Tables are created automatically as needed

### Task Management

Tasks are managed through various states:

```python
class TaskStatusT(Enum):
    PENDING = "pending"
    COMPLETED = "completed"
    CANCELLED = "cancelled"
```

Each task includes:
```python
class TaskT(BaseModel):
    id: Optional[int]
    description: str
    scheduled_time: datetime
    status: TaskStatusT
    created_at: datetime
    completed_at: Optional[datetime]
```

### Message Types

The system supports different notification types:
```python
class MessageTypeT(str, Enum):
    WHATSAPP = "whatsapp"
```

### Extended Thinking

The agent uses Claude 3.7 Sonnet with extended thinking enabled:
- Token budget: 4000 tokens
- Allows for complex reasoning about task scheduling
- Maintains context across multiple operations

### Database Operations

The system supports:
- Task creation and scheduling
- Status updates
- Task queries and filtering
- Table creation and management
- Complex SQL queries via the SQL Engine tool

Example task operations:
```python
# Schedule a task
task = task_scheduler.forward({
    "description": "Team meeting",
    "scheduled_time": "2024-03-31T15:00:00"
})

# Get pending tasks
pending = task_scheduler.get_pending_tasks()

# Complete a task
completed = task_scheduler.complete_task(task_id=1)
```

This architecture ensures:
- Persistent storage of all data
- Flexible task management
- Extensible system for adding new features
- Robust error handling
- Clear separation of concerns between components

## Left to do

- Setup an automatic scheduled task checking job
  - Need to implement a periodic job that checks for pending tasks
  - Should run every minute to check for tasks that need to be executed
  - When a task's scheduled time is reached, trigger the appropriate notification
  - Consider using Modal's scheduled functions (@modal.schedule()) for this implementation