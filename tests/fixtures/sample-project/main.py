from fastapi import FastAPI

app = FastAPI()


@app.get("/health")
def health():
    return {"status": "ok"}


# TODO: implement authentication

def authenticate():
    raise NotImplementedError
