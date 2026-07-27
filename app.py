from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn
import os

from config import API_HOST, API_PORT

path_prefix = os.environ["DOMINO_RUN_HOST_PATH"].rstrip("/")

print("DOMINO_RUN_HOST_PATH =", path_prefix)

app = FastAPI(root_path=path_prefix)

print("FastAPI root_path =", app.root_path)
print("OpenAPI URL =", app.openapi_url)


class Item(BaseModel):
    name: str
    description: str
    price: float


@app.get("/")
async def root():
    return {"message": "Hello World"}


@app.get("/items/{item_id}")
async def read_item(item_id: int):
    return {"item_id": item_id}


@app.post("/items/")
async def create_item(item: Item):
    return item


@app.get("/env")
def env():
    return {
        k: v
        for k, v in os.environ.items()
        if "DOMINO" in k or "APP" in k
    }

@app.get("/debug")
def debug():
    return {
        "root_path": app.root_path,
        "openapi_url": app.openapi_url,
        "docs_url": app.docs_url,
        "run_host_path": os.environ.get("DOMINO_RUN_HOST_PATH")
    }

if __name__ == "__main__":
    uvicorn.run(app, host=API_HOST, port=API_PORT)