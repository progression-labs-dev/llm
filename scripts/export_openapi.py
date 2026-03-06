"""Export the OpenAPI spec from the FastAPI app to openapi.json."""

import json
from pathlib import Path

from progression_labs.llm.server.app import app


def main() -> None:
    spec = app.openapi()
    output = Path(__file__).resolve().parent.parent / "openapi.json"
    output.write_text(json.dumps(spec, indent=2) + "\n")
    print(f"OpenAPI spec written to {output}")


if __name__ == "__main__":
    main()
