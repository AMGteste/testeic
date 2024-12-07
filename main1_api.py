import os
import uvicorn
from api import app  # Importa o app do FastAPI

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
