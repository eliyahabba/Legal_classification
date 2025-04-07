import json
import time
from typing import Dict, Any, List, Optional

import anthropic
import openai

from src.utils.constants import LLMModels, LLMTypes


class LLM:
    """
    Interface for OpenAI and Claude models
    Supports both regular and batch processing for Claude
    """

    def __init__(self, model_type: str, api_key: str):
        self.model_type = model_type
        if model_type == LLMTypes.OPENAI:
            self.openai_client = openai.OpenAI(api_key=api_key)
        else:  # claude
            self.claude = anthropic.Anthropic(api_key=api_key)

    def generate(self, messages: List[Dict[str, Any]], system=None) -> str:
        """
        Generate completion using either OpenAI or Claude
        """
        try:
            if self.model_type == LLMTypes.OPENAI:
                if system:
                    messages = [{"role": "system", "content": system}] + messages
                response = self.openai_client.chat.completions.create(
                    model=LLMModels.OPENAI,
                    messages=messages,
                    temperature=0.2
                )
                text_response = response.choices[0].message.content

            else:  # claude
                # Convert messages to Claude format
                claude_messages = self._convert_to_claude_messages(messages)
                claude_messages = claude_messages + [{"role": "assistant", "content": "{"}]
                response = self.claude.messages.create(
                    model=LLMModels.CLAUDE,
                    max_tokens=4096,
                    temperature=0.2,
                    messages=claude_messages,
                    system=system,  # System prompt as a separate parameter
                )
                text_response = response.content[0].text

            return text_response
        except Exception as e:
            print(f"Error generating completion: {e}")
            return ""

    def generate_batch(self, batch_requests: List[Dict[str, Any]], system=None,
                       poll_interval: int = 30, timeout: int = 3600) -> Dict[str, Any]:
        """
        Generate completions using Claude's Message Batches API

        Args:
            batch_requests: List of dictionaries, each containing:
                - custom_id: A unique identifier for the request
                - messages: List of message dictionaries
            system: Optional system prompt to apply to all requests
            poll_interval: Seconds to wait between polling for batch completion
            timeout: Maximum seconds to wait for batch completion

        Returns:
            Dictionary mapping custom_ids to their respective responses
        """
        if self.model_type != LLMTypes.CLAUDE:
            raise ValueError("Batch processing is only supported for Claude models")

        try:
            # Prepare batch requests
            requests = []
            for req in batch_requests:
                custom_id = req.get("custom_id")
                req_messages = req.get("messages", [])

                # Convert messages to Claude format
                claude_messages = self._convert_to_claude_messages(req_messages)
                claude_messages = claude_messages + [{"role": "assistant", "content": "{"}]

                from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
                from anthropic.types.messages.batch_create_params import Request

                requests.append(
                    Request(
                        custom_id=custom_id,
                        params=MessageCreateParamsNonStreaming(
                            model=LLMModels.CLAUDE,
                            max_tokens=4096,
                            temperature=0.2,
                            messages=claude_messages,
                            system=system
                        )
                    )
                )

            # Create batch
            message_batch = self.claude.messages.batches.create(requests=requests)
            batch_id = message_batch.id

            # Poll for completion
            start_time = time.time()
            while time.time() - start_time < timeout:
                batch_status = self.claude.messages.batches.retrieve(batch_id)

                if batch_status.processing_status == "ended":
                    break

                time.sleep(poll_interval)

            if batch_status.processing_status != "ended":
                raise TimeoutError(f"Batch processing timed out after {timeout} seconds")

            # Retrieve and process results
            results = {}
            for result in self.claude.messages.batches.results(batch_id):
                custom_id = result.custom_id
                if result.result.type == "succeeded":
                    # Extract the text response
                    message = result.result.message
                    text_response = message.content[0].text
                    try:
                        extraction = json.loads("{" + text_response)
                        results[custom_id] = extraction
                    except json.JSONDecodeError:
                        results[custom_id] = {"error": "Failed to parse JSON response"}
                else:
                    # Handle errors
                    error_type = result.result.type
                    error_message = getattr(result.result, "error", {"message": "Unknown error"})
                    results[custom_id] = {"error": f"{error_type}: {error_message}"}

            return results

        except Exception as e:
            print(f"Error generating batch completion: {e}")
            return {}

    def _convert_to_claude_messages(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Convert generic messages format to Claude-specific format
        """
        claude_messages = []

        for msg in messages:
            content = msg['content']

            # Handle file attachments
            if 'file' in msg:
                content += f"\n\nFile content ({msg['file']['name']}):\n{msg['file']['content']}"

            claude_messages.append({
                "role": msg['role'],
                "content": content
            })

        return claude_messages