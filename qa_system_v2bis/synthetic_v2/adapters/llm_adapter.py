from typing import Dict, Any


class RotatingClientAdapter:
    """
    Adapter around an existing rotating LLM client.
    Expected output: raw text string from the LLM.

    Usage:
        from qa_system_v2bis.llm.rotating_clients import RotatingClient
        client = RotatingClient(...)
        adapter = RotatingClientAdapter(client)
        # Then pass adapter as llm_client to run_synthetic_v2
    """

    def __init__(self, client):
        self.client = client

    def generate(self, prompt: str, seed_row: Dict[str, Any]) -> str:
        """
        Override this method body with the actual call pattern
        used by your rotating client.
        """
        # Example pseudo-call:
        # resp = self.client.complete(
        #     prompt=prompt,
        #     temperature=0.4,
        #     max_tokens=1200,
        #     metadata={
        #         "seed_id": seed_row.get("seed_id"),
        #         "model_name": seed_row.get("model_name"),
        #     }
        # )
        # return resp.text
        raise NotImplementedError(
            "RotatingClientAdapter.generate() must be implemented "
            "with your actual LLM client call pattern."
        )
