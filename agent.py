from datetime import datetime, UTC
import pandas as pd
from typing import List, Dict, Any, Union, Optional
from smolagents import Tool
from pydantic import BaseModel
import logging
import sqlite3
from smolagents import CodeAgent, LogLevel
from smolagents.models import LiteLLMModel
from smolagents import FinalAnswerTool
from typing import Type
from smolagents import LiteLLMModel
from typing import get_type_hints, get_origin, get_args
import inspect
from enum import Enum
from typing import Literal
from typing import cast, Generator
from smolagents import ActionStep, AgentType
import os
import json

def pydantic_to_schema(model_class: Type[BaseModel], description: str = None) -> Dict[str, Any]:
    """
    Convert a Pydantic model into a JSON schema format compatible with the Tool class.

    Args:
        model_class: A Pydantic model class (subclass of BaseModel)
        description: Optional description for the entire schema

    Returns:
        A dictionary representing the JSON schema
    """
    if not inspect.isclass(model_class) or not issubclass(model_class, BaseModel):
        raise TypeError("Input must be a Pydantic model class (subclass of BaseModel)")

    # Get model schema from Pydantic
    schema = model_class.model_json_schema()

    # Get field descriptions from docstrings if available
    field_descriptions = {}
    for field_name, field in model_class.model_fields.items():
        if field.description:
            field_descriptions[field_name] = field.description

    # Create the base schema
    result_schema = {
        "type": "object",
        "properties": {}
    }

    if description:
        result_schema["description"] = description
    elif "description" in schema:
        result_schema["description"] = schema["description"]
    else:
        result_schema["description"] = f"Schema for {model_class.__name__}"

    # Process each field
    properties = {}
    for field_name, field in model_class.model_fields.items():
        field_schema = process_field(field_name, field, model_class, field_descriptions)
        if field_schema:
            properties[field_name] = field_schema

    result_schema["properties"] = properties

    # Include required fields if any
    required_fields = [name for name, field in model_class.model_fields.items() if field.is_required()]
    if required_fields:
        result_schema["required"] = required_fields
    return result_schema


def process_field(field_name: str, field, model_class, field_descriptions: Dict[str, str]) -> Dict[str, Any]:
    """
    Process a single field and convert it to the appropriate schema format.

    Args:
        field_name: Name of the field
        field: Pydantic field object
        model_class: The parent model class
        field_descriptions: Dictionary of field descriptions

    Returns:
        A dictionary representing the field schema
    """
    # Get type hints to handle complex types
    type_hints = get_type_hints(model_class)
    field_type = type_hints.get(field_name)

    # Start with basic schema
    field_schema = {}

    # Add description if available
    if field_name in field_descriptions:
        field_schema["description"] = field_descriptions[field_name]
    elif field.description:
        field_schema["description"] = field.description
    else:
        field_schema["description"] = f"The {field_name} field"

    # Handle different types
    origin = get_origin(field_type)
    args = get_args(field_type)

    # Handle Union types (Optional is Union[T, None])
    if origin is Union:
        non_none_types = [arg for arg in args if arg is not type(None)]
        if len(non_none_types) == 1:
            # This is an Optional[T] field
            field_schema = process_type(non_none_types[0], field_schema)
            field_schema["nullable"] = True
        else:
            # Complex Union type - use anyOf
            field_schema["anyOf"] = [process_type(arg, {})["type"] for arg in non_none_types]

    # Handle List types
    elif origin is list:
        field_schema["type"] = "array"
        if args:
            item_type = args[0]
            if inspect.isclass(item_type) and issubclass(item_type, BaseModel):
                field_schema["items"] = pydantic_to_schema(item_type)
            else:
                field_schema["items"] = process_type(item_type, {})

    # Handle Dict types
    elif origin is dict:
        field_schema["type"] = "object"
        if len(args) >= 2:
            # For Dict[str, Something], we can provide additionalProperties
            value_type = args[1]
            if inspect.isclass(value_type) and issubclass(value_type, BaseModel):
                field_schema["additionalProperties"] = pydantic_to_schema(value_type)
            else:
                field_schema["additionalProperties"] = process_type(value_type, {})

    # Handle nested models
    elif inspect.isclass(field_type) and issubclass(field_type, BaseModel):
        nested_schema = pydantic_to_schema(field_type)
        field_schema.update(nested_schema)

    # Handle basic types
    else:
        field_schema = process_type(field_type, field_schema)

    return field_schema


def process_type(python_type, schema: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert Python type to JSON schema type.

    Args:
        python_type: The Python type
        schema: Existing schema to update

    Returns:
        Updated schema dictionary
    """
    type_mapping = {
        str: "string",
        int: "integer",
        float: "number",
        bool: "boolean",
        list: "array",
        dict: "object",
        None: "null",
    }

    # Handle Literal types
    origin = get_origin(python_type)
    if origin is Literal:
        args = get_args(python_type)
        schema["enum"] = list(args)
        # Determine the type from the first value
        if args:
            first_arg_type = type(args[0])
            if first_arg_type in type_mapping:
                schema["type"] = type_mapping[first_arg_type]
        return schema

    # Map the type if it's in our mapping
    if python_type in type_mapping:
        schema["type"] = type_mapping[python_type]
    else:
        # Default to "string" for unknown types
        schema["type"] = "string"

    return schema


class CustomFinalAnswerTool(FinalAnswerTool):
    def __init__(self, model : Type[BaseModel], description:str = "A user object", *args, **kwargs):
        self.inputs : Dict[str, Any] = {"answer":pydantic_to_schema(model, description)}
        self.model_pydantic : Type[BaseModel] = model
        super().__init__(*args, **kwargs)
    def forward(self, answer : dict) -> dict:
        data = self.model_pydantic.model_validate(answer)
        return data.model_dump()


logger = logging.getLogger(__name__)

TableNameT = str
ColumnNameT = str


class ForeignKeyReferenceT(BaseModel):
    table_name: TableNameT
    column_name: ColumnNameT


class ColumnMetadataT(BaseModel):
    type: Optional[str] = None
    description: Optional[str] = None
    foreign_key_references: Optional[ForeignKeyReferenceT] = None


class TableMetadataT(BaseModel):
    description: str


SchemaDictT = dict[TableNameT, dict[ColumnNameT, ColumnMetadataT]]
TableDictT = dict[TableNameT, TableMetadataT]


class SchemaT(BaseModel):
    schema_dict: SchemaDictT = {}
    table_description_dict: TableDictT = {}


class TableCreator(Tool):
    name = "table_creator"
    inputs = {
        "table_name": {
            "type": "string",
            "description": "Name of the table to create",
        },
        "columns": {
            "type": "array",
            "description": "List of column definitions in format [name, type]",
            "items": {
                "type": "array",
                "items": {"type": "string"},
                "minItems": 2,
                "maxItems": 2,
            },
        },
        "data": {
            "type": "array",
            "description": "List of rows to insert into the table",
            "items": {"type": "object"},
        },
    }
    output_type = "string"
    
    description = """Create a new table in the SQLite database with specified columns and data.

    This tool allows you to:
    1. Create a new table with custom column definitions
    2. Insert initial data rows into the newly created table
    3. Handle proper SQLite data types and constraints
    
    Args:
        table_name: Name of the table to create
        columns: List of [column_name, column_type] pairs defining the table schema
        data: List of dictionaries containing the initial data rows to insert
        
    Returns:
        A success message with details about the created table
        
    Example:
        table_name: "employees"
        columns: [["id", "INTEGER"], ["name", "TEXT"], ["salary", "REAL"]]
        data: [{"id": 1, "name": "John", "salary": 50000}]
    """
    
    def connect(self):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn
    def __init__(self, db_path: str = "tasks.db"):
        """Initialize TableCreator with SQLite connection."""
        self.db_path = db_path
        super().__init__()

    def _print_tables(self):
        """Print information about all tables in the database."""
        # Get list of all tables
        conn = self.connect()
        cursor = conn.execute("""
            SELECT name FROM sqlite_master 
            WHERE type='table' 
            ORDER BY name;
        """)
        tables = cursor.fetchall()
        
        if not tables:
            print("\nNo tables found in database.")
            return
            
        print("\nCurrent tables in database:")
        for table in tables:
            table_name = table[0]
            # Get row count
            count_cursor = conn.execute(f"SELECT COUNT(*) FROM {table_name}")
            row_count = count_cursor.fetchone()[0]
            
            # Get schema
            schema_cursor = conn.execute(f"PRAGMA table_info({table_name})")
            columns = schema_cursor.fetchall()
            
            print(f"\nTable: {table_name} ({row_count} rows)")
            print("Columns:")
            for col in columns:
                print(f"  - {col[1]}: {col[2]}")
            
            # Print first row as sample
            sample_cursor = conn.execute(f"SELECT * FROM {table_name} LIMIT 1")
            sample = sample_cursor.fetchone()
            if sample:
                print("Sample row:")
                print(f"  {dict(sample)}")
        conn.close()

    def forward(self, table_name: str, columns: List[List[str]], data: List[Dict[str, Any]]) -> str:
        """Create a new table and insert data."""
        conn = self.connect()
        try:
            # Create table
            column_defs = [f"{col[0]} {col[1]}" for col in columns]
            create_query = f"CREATE TABLE IF NOT EXISTS {table_name} ({', '.join(column_defs)})"
            conn.execute(create_query)

            # Insert data
            if data:
                placeholders = ','.join(['?' for _ in columns])
                insert_query = f"INSERT INTO {table_name} VALUES ({placeholders})"
                values = [[row.get(col[0]) for col in columns] for row in data]
                conn.executemany(insert_query, values)
                conn.commit()

            # Print current state of all tables
            self._print_tables()
            conn.close()
            return f"Successfully created table '{table_name}' with {len(data)} rows"
        except Exception as e:
            conn.close()
            raise Exception(f"Error creating table: {str(e)}")


INITIAL = f"""Allows you to perform SQL queries on the following tables.
Feel free to join tables together in your queries.
Beware that this tool's output is a pandas Dataframe of the execution output. (Note: Current time is {datetime.now(UTC)} (UTC))

Please try to join tables.
"""
class SQLAgent(Tool):
    name = "sql_engine"
    inputs = {
        "query": {
            "type": "string",
            "description": "The query to perform. This should be correct SQL.",
        }
    }
    output_type = "object"
    
    description = INITIAL
    def connect(self):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn
    def _fetch_tables_descriptions(self) -> str:
        """Get information about all tables in the database."""
        # Get list of all tables
        conn = self.connect()
        cursor = conn.execute("""
            SELECT name FROM sqlite_master 
            WHERE type='table' 
            ORDER BY name;
        """)
        tables = cursor.fetchall()
        
        if not tables:
            return "No tables found in database."
            
        description = "Current tables in database:"
        for table in tables:
            table_name = table[0]
            # Get row count
            count_cursor = conn.execute(f"SELECT COUNT(*) FROM {table_name}")
            row_count = count_cursor.fetchone()[0]
            
            # Get schema
            schema_cursor = conn.execute(f"PRAGMA table_info({table_name})")
            columns = schema_cursor.fetchall()
            
            description += f"\n\nTable: {table_name} ({row_count} rows)"
            description += "\nColumns:"
            for col in columns:
                description += f"\n  - {col[1]}: {col[2]}"
            
            # Get first row as sample
            sample_cursor = conn.execute(f"SELECT * FROM {table_name} LIMIT 1")
            sample = sample_cursor.fetchone()
            if sample:
                description += "\nSample row:"
                description += f"\n  {dict(sample)}"
        print(description)
        conn.close()
        return description
    def __init__(
        self,
        db_path: str = ":memory:",
    ):
        """
        Initialize SQLAgent with SQLite backend.

        Args:
            data: Either a List of JSON objects/dictionaries that will be loaded into a single table
                 or a Dictionary where keys are table names and values are lists of JSON objects
            column_metadata: Schema metadata for tables and columns
            db_path: Path to SQLite database file (defaults to in-memory)
        """
        print("Initializing SQLAgent with column metadata and data")
        self.db_path = db_path
        self.table_names = []
        self.description = INITIAL + self._fetch_tables_descriptions()
        super().__init__() 

    

    def forward(self, query: str) -> pd.DataFrame:
        """Execute SQL query and return results as DataFrame."""
        pd.set_option("display.max_columns", 30)
        conn = self.connect()
        self.description = INITIAL + self._fetch_tables_descriptions()
        try:
            result = pd.read_sql_query(query, conn)
            result_log = conn.execute(query).fetchall()
            if not result_log:
                logger.info("Query executed successfully. No results to display.")
            output = ""
            for row in result_log:
                output += "\n" + str(row)
            logger.info(output)
            return result
        except Exception as e:
            raise Exception(f"Error executing query: {str(e)}")



class DisplayTables(Tool):
    name = "display_tables"
    inputs = {}
    output_type = "string"
    description = "Display the current tables in the database."
    def connect(self):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn
    def _fetch_tables_descriptions(self) -> str:
        """Get information about all tables in the database."""
        # Get list of all tables
        conn = self.connect()
        cursor = conn.execute("""
            SELECT name FROM sqlite_master 
            WHERE type='table' 
            ORDER BY name;
        """)
        tables = cursor.fetchall()
        
        if not tables:
            return "No tables found in database."
            
        description = "Current tables in database:"
        for table in tables:
            table_name = table[0]
            # Get row count
            count_cursor = conn.execute(f"SELECT COUNT(*) FROM {table_name}")
            row_count = count_cursor.fetchone()[0]
            
            # Get schema
            schema_cursor = conn.execute(f"PRAGMA table_info({table_name})")
            columns = schema_cursor.fetchall()
            
            description += f"\n\nTable: {table_name} ({row_count} rows)"
            description += "\nColumns:"
            for col in columns:
                description += f"\n  - {col[1]}: {col[2]}"
            
            # Get first row as sample
            sample_cursor = conn.execute(f"SELECT * FROM {table_name} LIMIT 1")
            sample = sample_cursor.fetchone()
            if sample:
                description += "\nSample row:"
                description += f"\n  {dict(sample)}"
        print(description)
        conn.close()
        return description
    
    def __init__(
        self,
        db_path: str = ":memory:",
    ):
        """
        Initialize SQLAgent with SQLite backend.

        Args:
            data: Either a List of JSON objects/dictionaries that will be loaded into a single table
                 or a Dictionary where keys are table names and values are lists of JSON objects
            column_metadata: Schema metadata for tables and columns
            db_path: Path to SQLite database file (defaults to in-memory)
        """
        print("Initializing DisplayTables")
        self.db_path = db_path
        super().__init__() 
        
    
    def forward(self) -> str:
        return self._fetch_tables_descriptions()

class MessageTypeT(str, Enum):
    WHATSAPP = "whatsapp"

class MessageT(BaseModel):
    message: str
    message_type: MessageTypeT
    

class TaskStatusT(Enum):
    PENDING = "pending"
    COMPLETED = "completed"
    CANCELLED = "cancelled"

class TaskT(BaseModel):
    id: Optional[int] = None
    description: str
    scheduled_time: datetime
    status: TaskStatusT = TaskStatusT.PENDING
    created_at: datetime = datetime.now(UTC)
    completed_at: Optional[datetime] = None

class TaskScheduler(Tool):
    name = "task_scheduler"
    inputs = {
        "task": {
            "type": "object",
            "description": "Task details including description and scheduled time",
            "properties": {
                "description": {
                    "type": "string",
                    "description": "Description of the task"
                },
                "scheduled_time": {
                    "type": "string",
                    "description": "ISO format datetime string when the task should be executed"
                }
            },
            "required": ["description", "scheduled_time"]
        }
    }
    
    
    output_type = "object"

    description = """Schedule tasks for future execution.
    
    This tool allows you to:
    1. Create new tasks with descriptions and scheduled times
    2. Store tasks in a SQLite database for persistence
    3. Track task status (pending, completed, cancelled)
    
    Args:
        task: Dictionary containing:
            - description: Text description of what the task involves
            - scheduled_time: ISO format datetime string for when task should execute
            
    Returns:
        Dictionary containing the created task details including:
        - id: Unique task identifier
        - description: Task description
        - scheduled_time: When task is scheduled for
        - status: Current task status
        - created_at: When task was created
        - completed_at: When task was completed (if applicable)
        
    Example:
        task = {
            "description": "Send reminder of meeting with John Doe",
            "scheduled_time": "2024-03-31T09:00:00"
        }
    """ + datetime.now(UTC).isoformat()
    def connect(self):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def __init__(self, db_path: str = "tasks.db"):
        """Initialize TaskScheduler with SQLite database."""
        self.db_path = db_path
        self._init_db()
        super().__init__()

    def _init_db(self):
        """Initialize the database schema."""
        conn = self.connect()
        
        conn.execute("""
            CREATE TABLE IF NOT EXISTS tasks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                description TEXT NOT NULL,
                scheduled_time TEXT NOT NULL,
                status TEXT NOT NULL,
                created_at TEXT NOT NULL,
                completed_at TEXT
            )
        """)
        conn.commit()
        conn.close()
    def forward(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Schedule a new task."""
        try:
            conn = self.connect()
            # Parse the scheduled time
            scheduled_time = datetime.fromisoformat(task["scheduled_time"])
            
            # Insert the task
            cursor = conn.execute("""
                INSERT INTO tasks (description, scheduled_time, status, created_at)
                VALUES (?, ?, ?, ?)
            """, (
                task["description"],
                scheduled_time.isoformat(),
                TaskStatusT.PENDING.value,
                datetime.now(UTC).isoformat()
            ))
            
            conn.commit()
            
            # Get the created task
            task_id = cursor.lastrowid
            created_task = dict(conn.execute(
                "SELECT * FROM tasks WHERE id = ?", 
                (task_id,)
            ).fetchone())
            conn.close()
            return created_task
        except Exception as e:
            raise Exception(f"Error scheduling task: {str(e)}")

    def get_pending_tasks(self) -> List[Dict[str, Any]]:
        """Get all pending tasks."""
        conn = self.connect()
        cursor = conn.execute(
            "SELECT * FROM tasks WHERE status = ? ORDER BY scheduled_time",
            (TaskStatusT.PENDING.value,)
        )
        results = [dict(row) for row in cursor.fetchall()]
        conn.close()
        return results

    def complete_task(self, task_id: int) -> Dict[str, Any]:
        """Mark a task as completed."""
        conn = self.connect()
        conn.execute("""
            UPDATE tasks 
            SET status = ?, completed_at = ?
            WHERE id = ?
        """, (
            TaskStatusT.COMPLETED.value,
            datetime.now(UTC).isoformat(),
            task_id
        ))
        conn.commit()
        updated_task = dict(conn.execute(
            "SELECT * FROM tasks WHERE id = ?", 
            (task_id,)
        ).fetchone())
        conn.close()
        return updated_task

    def cancel_task(self, task_id: int) -> Dict[str, Any]:
        """Cancel a task."""
        conn = self.connect()
        conn.execute("""
            UPDATE tasks 
            SET status = ?
            WHERE id = ?
        """, (TaskStatusT.CANCELLED.value, task_id))
        conn.commit()
        
        updated_task = dict(conn.execute(
            "SELECT * FROM tasks WHERE id = ?", 
            (task_id,)
        ).fetchone())
        conn.close()
        return updated_task


class ExecutiveAgent:
    """Main Executive Agent implementation integrating all tools."""

    def __init__(self, data={}, enable_extended_thinking: bool = True, db_path="/alldata/tasks.db"):
        # Initialize tools
        self.table_creator = TableCreator(db_path=db_path)
        self.sql_engine = SQLAgent(db_path=db_path)
        self.task_scheduler = TaskScheduler(db_path=db_path)
        self.display_tables = DisplayTables(db_path=db_path)
        # Collect all tools
        tools : list[Tool] = [
            self.table_creator,
            self.sql_engine,
            self.task_scheduler,
            self.display_tables,
            CustomFinalAnswerTool(MessageT),
        ]

        # Initialize model - Using Claude 3.7 Sonnet with extended thinking
        model_params = {
        }
        if enable_extended_thinking:
            model_params["thinking"] = {"type": "enabled", "budget_tokens": 4000}
        model = LiteLLMModel(
                "anthropic/claude-3-7-sonnet-20250219",
                **model_params
            )
        
        self.agent = CodeAgent(
            tools=tools,
            model=model,
            additional_authorized_imports=[
                "datetime",
                "json",
                "numpy",
                "pandas",
                "math",
                "statistics"
            ],
            verbosity_level=LogLevel.INFO,
            step_callbacks=[]
        )
    def run(self, query: str):
        """
        Run the executive agent with the provided query.

        Args:
            query: The user's question or request

        Returns:
            Generator yielding steps and visualizations for the UI
        """
        for step in cast(
            Generator[ActionStep | AgentType, None, None],
            self.agent.run(query, stream=True)
        ):
            # For UI purposes, collect maps and charts after each step
            # maps = self.display_map.get_last_maps()
            # charts = self.display_chart.get_last_charts()
            # self.display_map.clear_last_maps()
            # self.display_chart.clear_last_charts()
            # Yield both the step and any visualizations for the UI
            yield step
        print(self.task_scheduler.get_pending_tasks())
            
if __name__ == "__main__":
    agent = ExecutiveAgent(db_path="tasks.db")
    # Create the query with current datetime
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    SYSTEM_PROMPT = """
    You are a helpful assistant that can schedule tasks for the user.
    Always check to see if the task is already scheduled.
    If it is, return the task details.
    If it is not, schedule the task and return the task details.
    """
    query = SYSTEM_PROMPT +"\n\n" + f"Message: \n hi, I'm meeting someone tomorrow at 3pm, could you please remind me? Today is {current_time}"
    data = {
        "query": query,
        "phone_number": "+12035083967"
    }
    for step in agent.run(query):
        print(step)
    print(agent.task_scheduler.get_pending_tasks())