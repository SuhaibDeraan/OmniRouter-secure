from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from serverRouter.core.providers import initialize_providers
from serverRouter.routes import (
    model_routes,
    completion_routes,
    smart_routes,
    reasoning_routes,
    health_routes,
)


app = FastAPI(title="OmniLLM", description="One Key, One API, Hundreds of Models")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize providers during startup. Each provider is constructed
# independently -- a missing credential for one does not abort the others.
initialize_providers()

# Include routers from separate files
app.include_router(model_routes.router)
app.include_router(completion_routes.router)
app.include_router(smart_routes.router)
app.include_router(reasoning_routes.router)
app.include_router(health_routes.router)


@app.get("/")
async def root():
    return {"message": "Welcome to OmniLLM!"}


# uvicorn serverRouter.router:app --reload
