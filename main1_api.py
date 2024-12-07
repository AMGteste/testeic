# -*- coding: utf-8 -*-
"""
Created on Thu Dec  5 19:59:37 2024

@author: guerr
"""

import uvicorn
from api import app  # Importa o app do FastAPI

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
