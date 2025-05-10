from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from .core.config import settings
from .api.v1.endpoints import video

app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    openapi_url=settings.OPENAPI_URL,
    docs_url=settings.DOCS_URL,
    redoc_url=settings.REDOC_URL,
    description="""
    # RTOG API Documentation

    ## REST Endpoints
    * `POST /api/v1/video/upload` - Upload a video file
    * `GET /api/v1/video/process/{file_path}` - Process a video file

    ## WebSocket Endpoints
    * `WS /api/v1/video/ws/{client_id}` - Real-time video processing updates
        * Connect with a unique client_id
        * Receive real-time updates about video processing
        * Send messages to control the processing

    ## WebSocket Message Format
    ```json
    {
        "type": "control",
        "action": "start|stop|pause",
        "data": {}
    }
    ```

    ## WebSocket Response Format
    ```json
    {
        "type": "update",
        "status": "processing|completed|error",
        "data": {
            "progress": 0.75,
            "message": "Processing frame 150 of 200"
        }
    }
    ```
    """
)

# Set up CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.BACKEND_CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(video.router, prefix=settings.API_V1_STR + "/video", tags=["video"])

@app.get("/")
async def root():
    return {"message": "Welcome to RTOG API"} 