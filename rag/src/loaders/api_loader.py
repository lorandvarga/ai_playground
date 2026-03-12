import json
from typing import List, Dict, Any, Optional
from datetime import datetime
import requests
from langchain_core.documents import Document


def _flatten_json(data: Any, parent_key: str = "", sep: str = ".") -> Dict[str, Any]:
    """
    Flatten nested JSON structure into a single-level dictionary.

    Args:
        data: JSON data to flatten
        parent_key: Key prefix for nested items
        sep: Separator between nested keys

    Returns:
        Flattened dictionary
    """
    items = []

    if isinstance(data, dict):
        for k, v in data.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, (dict, list)):
                items.extend(_flatten_json(v, new_key, sep=sep).items())
            else:
                items.append((new_key, v))
    elif isinstance(data, list):
        for i, v in enumerate(data):
            new_key = f"{parent_key}[{i}]"
            if isinstance(v, (dict, list)):
                items.extend(_flatten_json(v, new_key, sep=sep).items())
            else:
                items.append((new_key, v))
    else:
        items.append((parent_key, data))

    return dict(items)


def _json_to_text(data: Any) -> str:
    """
    Convert JSON data to formatted text for document processing.

    Args:
        data: JSON data to convert

    Returns:
        Formatted text representation
    """
    if isinstance(data, dict):
        lines = []
        for key, value in data.items():
            if isinstance(value, (dict, list)):
                lines.append(f"{key}:")
                lines.append(_json_to_text(value))
            else:
                lines.append(f"{key}: {value}")
        return "\n".join(lines)
    elif isinstance(data, list):
        lines = []
        for i, item in enumerate(data):
            if isinstance(item, (dict, list)):
                lines.append(f"Item {i + 1}:")
                lines.append(_json_to_text(item))
            else:
                lines.append(f"- {item}")
        return "\n".join(lines)
    else:
        return str(data)


def load_from_api(
    url: str,
    headers: Optional[Dict[str, str]] = None,
    method: str = "GET",
    params: Optional[Dict[str, Any]] = None,
    json_data: Optional[Dict[str, Any]] = None
) -> List[Document]:
    """
    Fetch data from a REST API and convert to Document objects.

    Args:
        url: API endpoint URL
        headers: Optional HTTP headers
        method: HTTP method (GET, POST, etc.)
        params: Optional query parameters
        json_data: Optional JSON body for POST requests

    Returns:
        List of Document objects with API data and metadata

    Raises:
        requests.RequestException: If there's a network error
        ValueError: If the response is not valid JSON
    """
    try:
        response = requests.request(
            method=method.upper(),
            url=url,
            headers=headers,
            params=params,
            json=json_data,
            timeout=30
        )
        response.raise_for_status()

        try:
            data = response.json()
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON response from API: {str(e)}")

        # Convert JSON to text format
        text_content = _json_to_text(data)

        # Create metadata
        metadata = {
            "source": url,
            "type": "api",
            "endpoint": url,
            "timestamp": datetime.now().isoformat(),
            "status_code": response.status_code
        }

        # Create Document
        document = Document(
            page_content=text_content,
            metadata=metadata
        )

        return [document]

    except requests.RequestException as e:
        raise requests.RequestException(f"Error fetching data from API {url}: {str(e)}")
    except Exception as e:
        raise Exception(f"Unexpected error loading API data from {url}: {str(e)}")
