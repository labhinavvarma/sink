import json
import logging
import shlex
from pydantic import BaseModel, Field
from typing import Any, Dict, List, Optional
import httpx
import snowflake.connector

HOST = "carelon-eda-nonprod.privatelink.snowflakecomputing.com"
url = f"https://{HOST}/api/v2/cortex/inference:complete"

DEFAULT_MAX_TOKENS = 16000
DEFAULT_MODEL = "claude-4-sonnet"
#DEFAULT_MODEL = "snowflake-llama-3.3-70b"

# Snowflake connection defaults
DEFAULT_ACCOUNT = "st69414.us-east-2.privatelink"
DEFAULT_PORT = 443
DEFAULT_WAREHOUSE = "D01_EDA_GENAI_USER_WH_S"
DEFAULT_ROLE = "D01_RSTRCTD_EDA_GENAI_SUPPORTCBT_CORTEX_DVLPR"
DEFAULT_AUTHENTICATOR = "https://portalsso.elevancehealth.com/snowflake/okta"

logger = logging.getLogger(__name__)


def resolve_refs(schema: dict) -> dict:
    """
    Recursively resolve $ref in a Pydantic JSON schema using $defs.
    Returns a new dict with all references replaced by their definitions.
    """
    defs = schema.get("$defs", {})
    def _resolve(obj):
        if isinstance(obj, dict):
            if "$ref" in obj:
                ref_path = obj["$ref"]
                # Only support local refs like "#/$defs/SomeType"
                if ref_path.startswith("#/$defs/"):
                    def_key = ref_path.split("/")[-1]
                    return _resolve(defs[def_key])
                else:
                    raise ValueError(f"Unsupported $ref path: {ref_path}")
            else:
                return {k: _resolve(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [_resolve(item) for item in obj]
        else:
            return obj
    # Remove $defs from the top-level after resolving
    resolved = _resolve(schema)
    if "$defs" in resolved:
        del resolved["$defs"]
    return resolved


class LLMClient:
    def __init__(
        self, 
        config: dict | None = None, 
        session: Any = None,
        # Snowflake connection parameters
        user: Optional[str] = None,
        password: Optional[str] = None,
        account: str = DEFAULT_ACCOUNT,
        host: str = HOST,
        port: int = DEFAULT_PORT,
        warehouse: str = DEFAULT_WAREHOUSE,
        role: str = DEFAULT_ROLE,
        authenticator: str = DEFAULT_AUTHENTICATOR
    ):
        if config is None:
            config = {
                "temperature": 0,
                "max_tokens": DEFAULT_MAX_TOKENS,
                "model": DEFAULT_MODEL,
                "base_url": url
            }

        self.config = config
        if "max_tokens" not in config:
            self.config["max_tokens"] = DEFAULT_MAX_TOKENS
        if "model" not in config:
            self.config["model"] = DEFAULT_MODEL
        if "base_url" not in config:
            self.config["base_url"] = url
       
        self.session = session
        
        # Store Snowflake config for creating connection if needed
        self.snowflake_config = {
            "user": user,
            "password": password,
            "account": account,
            "host": host,
            "port": port,
            "warehouse": warehouse,
            "role": role,
            "authenticator": authenticator
        }
        self._connection_created = False

    def _get_session(self):
        """Get or create Snowflake session."""
        if self.session is None:
            # Create connection if user and password are provided
            if self.snowflake_config["user"] and self.snowflake_config["password"]:
                logger.info("Creating Snowflake connection...")
                self.session = snowflake.connector.connect(**self.snowflake_config)
                self._connection_created = True
                logger.info("Snowflake connection established")
            else:
                raise ValueError(
                    "Either 'session' parameter or 'user' and 'password' parameters must be provided"
                )
        return self.session

    def close(self):
        """Close Snowflake connection if it was created by this client."""
        if self._connection_created and self.session:
            self.session.close()
            self.session = None
            self._connection_created = False
            logger.info("Snowflake connection closed")

       
    async def _generate_response(
        self,
        messages: list[dict[str,Any]],
        response_model: type[BaseModel] | None = None,
    ) -> dict[str, Any]:

        response_format = {}
        if response_model:
            resolved_schema = resolve_refs(response_model.model_json_schema())
            # Recursively clean properties: remove 'title', replace 'anyOf' with 'type', remove 'default' if present
            def clean_property(obj):
                if isinstance(obj, dict):
                    obj.pop("title", None)
                    if "anyOf" in obj:
                        # Find the first dict in anyOf with a 'type' key
                        for entry in obj["anyOf"]:
                            if isinstance(entry, dict) and "type" in entry:
                                obj["type"] = entry["type"]
                                break
                        obj.pop("anyOf", None)
                        obj.pop("default", None)
                    for v in obj.values():
                        clean_property(v)
                elif isinstance(obj, list):
                    for item in obj:
                        clean_property(item)
            properties = resolved_schema.get("properties", {})
            clean_property(properties)
            response_format = {
                "response_format": {
                    "type": "json",
                    "schema": {
                        "properties": properties,
                        "required": resolved_schema.get("required", []),
                        "type": resolved_schema.get("type", "object")
                    }
                }
            }
       
        options = {
            "temperature": self.config["temperature"],
            "max_tokens": self.config["max_tokens"] if self.config["max_tokens"] is not None else 16000,
            **response_format
        }

        request_body = {
            "model": self.config["model"],
            "messages": messages,
            **options,
        }
       
        # Get session and use its token
        session = self._get_session()
        headers = {
            "Authorization": f'Snowflake Token="{session.rest.token}"',
            "Content-Type": "application/json",
            "Accept": "application/json"
        }
        
        try:
            async with httpx.AsyncClient(timeout=httpx.Timeout(300.0)) as client:
                response = await client.post(
                    url=self.config["base_url"],
                    headers=headers,
                    json=request_body
                )
                response.raise_for_status()
                response_chunks = []
                if response.status_code == 200:
                    async for line in response.aiter_lines():
                        if line:
                            line_content = line.replace('data: ', '')
                            try:
                                json_line = json.loads(line_content)
                                response_chunks.append(json_line)
                            except json.JSONDecodeError:
                                logger.warning(f"Error decoding JSON line: {line_content[:100]}")
                                continue
                    
                    # Combine chunks - same logic as your working code
                    if response_chunks:
                        result = {
                            "id": response_chunks[0].get("id"),
                            "created": response_chunks[0].get("created"),
                            "model": response_chunks[0].get("model"),
                            "choices": [],
                            "usage": response_chunks[-1].get("usage", {})
                        }
                        
                        # Use "text" field like your working code
                        combined_content = ''.join(
                            chunk["choices"][0]["delta"]["text"] 
                            for chunk in response_chunks 
                            if "choices" in chunk and chunk["choices"][0].get("delta")
                        )
                        
                        result["choices"].append({"content": combined_content})
                        
                        # If response_model specified, parse as JSON
                        if response_model:
                            try:
                                return json.loads(combined_content)
                            except json.JSONDecodeError:
                                cleaned = combined_content.strip("```").strip("json").strip()
                                return json.loads(cleaned)
                        else:
                            return result
                    else:
                        raise ValueError("No response chunks received")
                        
        except Exception as e:
            logger.error(f'Error in generating LLM response: {e}')
            print("-----<curl command>--------")
            session = self._get_session()
            curl_cmd = [
                "curl",
                "-X", "POST",
                shlex.quote(self.config["base_url"]),
                "-H", shlex.quote(f'Authorization: Snowflake Token="{session.rest.token}"'),
                "-H", "'Content-Type: application/json'",
                "-H", "'Accept: application/json'",
                "-d", shlex.quote(json.dumps(request_body))
            ]
            print(" ".join(curl_cmd))
            print("-----<curl command>--------")
            raise


# Example usage
if __name__ == "__main__":
    import snowflake.connector
    
    # Option 1: Create connection externally (recommended for production)
    conn = snowflake.connector.connect(
        user="your_username",
        password="your_password",
        account="st69414.us-east-2.privatelink",
        host="carelon-eda-nonprod.privatelink.snowflakecomputing.com",
        port=443,
        warehouse="D01_EDA_GENAI_USER_WH_S",
        role="D01_RSTRCTD_EDA_GENAI_SUPPORTCBT_CORTEX_DVLPR",
        authenticator="https://portalsso.elevancehealth.com/snowflake/okta"
    )
    
    client = LLMClient(
        session=conn,
        config={
            "model": "snowflake-llama-3.3-70b",
            "temperature": 0,
            "max_tokens": 4000
        }
    )
    
    # Option 2: Pass credentials directly (client creates connection)
    # client = LLMClient(
    #     user="your_username",
    #     password="your_password",
    #     config={
    #         "model": "snowflake-llama-3.3-70b",
    #         "temperature": 0,
    #         "max_tokens": 4000
    #     }
    # )
    
    # Use the client
    import asyncio
    
    async def example():
        prompt = "States of America"
        context = "You are powerful AI assistant in providing accurate answers. Be Concise in providing answers based on context."
        
        messages = [
            {
                "role": "user",
                "content": context + prompt
            }
        ]
        
        result = await client._generate_response(messages)
        print("Response:", result["choices"][0]["content"])
        print("\nFull response structure:", json.dumps(result, indent=2))
        
        # Close connection if created by client
        client.close()
    
    # Uncomment to run
    # asyncio.run(example())
