import os
import json
import sys
from unittest.mock import MagicMock

# 1. Fake the environment variables required by start.py's global scope
os.environ["SQLITE_DB_PATH"] = ":memory:"
os.environ["HERMES_API_KEY"] = "dummy_key"

# 2. Mock out the monitors so they don't try to connect to Asterisk/Triton during import
sys.modules["monitors"] = MagicMock()
sys.modules["monitors.clusters_monitor"] = MagicMock()
sys.modules["monitors.asterisk_monitor_dependent"] = MagicMock()

# 3. Import your app (FastAPI will build the routes but won't start the server or lifespan)
from start import app

# 4. Define the Scalar HTML template with a secure placeholder
SCALAR_HTML_TEMPLATE = """<!DOCTYPE html>
<html>
  <head>
    <title>Hermes API Documentation</title>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <style>
      /* Optional: Dark mode by default for a sleeker look */
      body { background-color: #111; margin: 0; }
    </style>
  </head>
  <body>
    <script id="api-reference" type="application/json">
__OPENAPI_JSON_PLACEHOLDER__
    </script>
    
    <script src="https://cdn.jsdelivr.net/npm/@scalar/api-reference"></script>
  </body>
</html>
"""

# 5. Extract the OpenAPI schema as a formatted JSON string
schema_json_str = json.dumps(app.openapi(), indent=2)

# 6. Inject the JSON string into the HTML template safely
final_html = SCALAR_HTML_TEMPLATE.replace(
    "__OPENAPI_JSON_PLACEHOLDER__", schema_json_str
)

# 7. Write the complete HTML to a file with UTF-8 encoding
with open("../../docs/swagger.html", "w", encoding="utf-8") as f:
    f.write(final_html)

json.dump(app.openapi(), open("../../docs/openapi.json", "w"), indent=2)

print("Documentation generated successfully as swagger.html!")
