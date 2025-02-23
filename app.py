from fastapi import FastAPI, File, UploadFile, HTTPException, Depends, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordBearer
from ibm_watsonx_ai.foundation_models import Model
from ibm_watsonx_ai.metanames import GenTextParamsMetaNames
from ibm_watsonx_ai.rag import RetrievalAugmentedGenerator
from ibm_watsonx_ai.tooling import ToolChain
from jose import JWTError, jwt
from pydantic import BaseModel
from typing import List, Optional
import pandas as pd
import numpy as np
import json
import os
from dotenv import load_dotenv
from datetime import datetime, timedelta
from prometheus_client import Counter, Histogram
import asyncio
from typing import Dict, List, Optional, Union
from ml_models import PredictiveAnalytics, MarketIntelligence
from visualizations import SupplyChainVisualizer
from alerts import AlertSystem

# Add metrics
api_requests = Counter('api_requests_total', 'Total API requests', ['endpoint'])
model_latency = Histogram('model_inference_seconds', 'Model inference duration')
# Add new Pydantic models
class SupplyChainMetrics(BaseModel):
    inventory_turnover: float
    order_fulfillment_rate: float
    supplier_performance: Dict[str, float]
    risk_score: float

class InventoryOptimizationRequest(BaseModel):
    current_stock: Dict[str, int]
    historical_demand: List[Dict[str, float]]
    lead_times: Dict[str, int]
    holding_cost: float
    stockout_cost: float
# Add new endpoint for comprehensive supply chain analysis
@app.post("/analyze-supply-chain", response_model=SupplyChainMetrics)
@limiter.limit("5/minute")
@cache(expire=600)
async def analyze_supply_chain(request: Request, user=Depends(get_current_user)):
    api_requests.labels(endpoint='/analyze-supply-chain').inc()
    
    with model_latency.time():
        prompt = """Analyze the entire supply chain and provide:
        1. Inventory turnover rate
        2. Order fulfillment rate
        3. Supplier performance scores
        4. Overall risk assessment
        
        Format: JSON with numerical metrics"""
        
        response = await asyncio.gather(
            granite_model.generate_text(prompt),
            simulate_risk(RiskSimulationRequest(disruption=5.0, demand_spike=10.0))
        )
        
        analysis = json.loads(response[0].generated_text)
        return SupplyChainMetrics(**analysis)
# Add inventory optimization endpoint
@app.post("/optimize-inventory")
@limiter.limit("10/minute")
async def optimize_inventory(
    request: Request,
    data: InventoryOptimizationRequest,
    user=Depends(get_current_user)
):
    api_requests.labels(endpoint='/optimize-inventory').inc()
    
    prompt = f"""Optimize inventory levels considering:
    - Current stock: {data.current_stock}
    - Historical demand: {data.historical_demand}
    - Lead times: {data.lead_times}
    - Holding cost: {data.holding_cost}
    - Stockout cost: {data.stockout_cost}
    
    Provide optimal order quantities and reorder points."""
    
    response = granite_model.generate_text(prompt)
    return json.loads(response.generated_text)
# Add health check endpoint
@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "model_status": "available" if granite_model else "unavailable",
        "cache_status": "connected" if FastAPICache.get_cache() else "disconnected"
    }
# Load environment variables
load_dotenv()
# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler('app.log'), logging.StreamHandler()]
)
logger = logging.getLogger(__name__)
# Initialize rate limiter
limiter = Limiter(key_func=get_remote_address)
app = FastAPI(title="SupplyGenius Pro API")
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
# Security configurations
SECRET_KEY = os.getenv("SECRET_KEY", "supersecret")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")
# CORS Configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# Initialize IBM Granite model
try:
    granite_model = Model(
        model_id="ibm/granite-13b-instruct-v1",
        credentials={
            "apikey": os.getenv("IBM_CLOUD_API_KEY"),
            "url": os.getenv("IBM_CLOUD_URL")
        },
        project_id=os.getenv("IBM_PROJECT_ID")
    )
except Exception as e:
    print(f"Error initializing IBM Granite model: {e}")
    granite_model = None
# Knowledge Base Implementation
class SupplyChainKB:
    def __init__(self):
        self.knowledge_base = {
            "suppliers": {
                "ACME Corp": {"lead_time": 14, "rating": 4.8, "pricing": "competitive"},
                "GlobalParts": {"lead_time": 7, "rating": 4.5, "pricing": "premium"}
            },
            "compliance_standards": ["ISO-9001", "REACH", "RoHS"],
            "risk_mitigation": {
                "supply_disruption": ["Diversify suppliers", "Safety stock"],
                "demand_spike": ["Flexible contracts", "Ramp up production"]
            }
        }
    def retrieve(self, query: str) -> str:
        return json.dumps(self.knowledge_base.get(query, {}))
# Initialize RAG
rag = RetrievalAugmentedGenerator(
    model=granite_model,
    retriever=SupplyChainKB().retrieve
)
# Tool Chain Initialization
tool_chain = ToolChain(granite_model)
# Pydantic Models
class User(BaseModel):
    username: str
    password: str
    
    class Config:
        min_anystr_length = 3
        max_anystr_length = 50
class ROIRequest(BaseModel):
    current_cost: float
    optimized_cost: float
    
    @validator('current_cost', 'optimized_cost')
    def validate_costs(cls, v):
        if v < 0:
            raise ValueError('Cost cannot be negative')
        return v
class DocumentAnalysisResponse(BaseModel):
    document_type: str
    extracted_data: dict
    confidence_score: float
class PredictionRequest(BaseModel):
    historical_data: List[float]
    forecast_period: int
class PredictionResponse(BaseModel):
    forecast: List[float]
    confidence_intervals: dict
class RiskSimulationRequest(BaseModel):
    disruption: float
    demand_spike: float
# Authentication Functions
def create_access_token(data: dict):
    expires = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    data.update({"exp": expires})
    return jwt.encode(data, SECRET_KEY, algorithm=ALGORITHM)
async def get_current_user(token: str = Depends(oauth2_scheme)):
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        return payload
    except JWTError:
        raise HTTPException(status_code=401, detail="Invalid credentials")
# Enhanced API endpoints with rate limiting and caching
@app.post("/token")
@limiter.limit("5/minute")
async def login(request: Request, user: User):
    if user.username != "admin" or user.password != "supplygenius":
        raise HTTPException(status_code=400, detail="Invalid credentials")
    return {"access_token": create_access_token({"sub": user.username})}
@app.post("/process-document", response_model=DocumentAnalysisResponse)
@limiter.limit("10/minute")
async def process_document(request: Request, file: UploadFile = File(...), user=Depends(get_current_user)):
    if not file.content_type.startswith('text/'):
        raise HTTPException(status_code=400, detail="Only text files are supported")
    
    if not granite_model:
        raise HTTPException(status_code=503, detail="AI model not available")
    
    try:
        content = await file.read()
        content_text = content.decode("utf-8")
        
        prompt = f"""Analyze this supply chain document and extract structured data:
        {content_text}
        
        Return JSON with:
        - document_type
        - parties_involved
        - key_dates
        - financial_terms
        - quantities
        - compliance_status
        """
        
        response = granite_model.generate_text(
            prompt=prompt,
            params={GenTextParamsMetaNames.MAX_NEW_TOKENS: 500}
        )
        
        analysis = json.loads(response.generated_text)
        return DocumentAnalysisResponse(
            document_type=analysis.get("document_type", "unknown"),
            extracted_data=analysis,
            confidence_score=0.97
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
@app.post("/predict-demand", response_model=PredictionResponse)
@limiter.limit("20/minute")
@cache(expire=300)
async def predict_demand(request: Request, pred_request: PredictionRequest, user=Depends(get_current_user)):
    if not granite_model:
        raise HTTPException(status_code=503, detail="AI model not available")
    
    try:
        prompt = f"""Analyze historical demand data and predict next {request.forecast_period} periods:
        Data: {request.historical_data}
        Include confidence intervals."""
        
        response = granite_model.generate_text(prompt)
        forecast = json.loads(response.generated_text)
        return PredictionResponse(**forecast)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
@app.post("/simulate-risk")
async def simulate_risk(request: RiskSimulationRequest, user=Depends(get_current_user)):
    prompt = f"""Simulate supply chain risk with:
    {request.disruption}% disruption and {request.demand_spike}% demand spike.
    Suggest mitigation strategies."""
    
    response = rag.generate(prompt)
    return {"strategies": response.split("\n")}
@tool_chain.tool
def generate_purchase_order(supplier: str, items: dict):
    return f"PO-{datetime.now().timestamp()}\nSupplier: {supplier}\nItems: {json.dumps(items)}"
@app.post("/generate-po")
async def create_po(supplier: str = Form(...), items: str = Form(...), user=Depends(get_current_user)):
    return tool_chain.run(f"Generate PO for {supplier} with items: {items}")
class ROICalculator:
    @staticmethod
    def calculate(data: dict):
        current = data.get('current_cost', 0)
        optimized = data.get('optimized_cost', 0)
        return {
            "savings": current - optimized,
            "roi": f"{((current - optimized)/current)*100:.1f}%"
        }
# Enhanced ROI calculator endpoint
@app.post("/calculate-roi")
@limiter.limit("30/minute")
async def calculate_roi(request: Request, data: ROIRequest, user=Depends(get_current_user)):
    try:
        result = ROICalculator.calculate(data.dict())
        logger.info(f"ROI calculation completed for user {user['sub']}")
        return result
    except Exception as e:
        logger.error(f"ROI calculation failed: {str(e)}")
        raise HTTPException(status_code=500, detail="ROI calculation failed")
# Startup event to initialize cache
@app.on_event("startup")
async def startup():
    try:
        redis = aioredis.from_url("redis://localhost", encoding="utf8", decode_responses=True)
        FastAPICache.init(RedisBackend(redis), prefix="supplygenius-cache")
        logger.info("Cache initialized successfully")
    except Exception as e:
        logger.error(f"Cache initialization failed: {str(e)}")
# Shutdown event for cleanup
@app.on_event("shutdown")
async def shutdown():
    logger.info("Shutting down application")
# Add imports at the top
from ml_models import PredictiveAnalytics, MarketIntelligence
from visualizations import SupplyChainVisualizer
from alerts import AlertSystem
# Initialize components after existing initializations
predictive_analytics = PredictiveAnalytics()
visualizer = SupplyChainVisualizer()
alert_system = AlertSystem()
# Add new endpoint for advanced analytics
@app.post("/advanced-analytics")
@limiter.limit("10/minute")
async def get_advanced_analytics(
    request: Request,
    historical_data: List[float],
    forecast_horizon: int = 12,
    user=Depends(get_current_user)
):
    forecasts = predictive_analytics.predict_demand(
        np.array(historical_data),
        forecast_horizon
    )
    
    visualization = visualizer.create_demand_forecast_plot(
        historical_data,
        forecasts["ensemble_forecast"]
    )
    
    return {
        "forecasts": forecasts,
        "visualization": visualization
    }
# Add WebSocket endpoint for real-time alerts
@app.websocket("/ws/alerts")
async def alert_websocket(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            inventory_alerts = await alert_system.check_inventory_alerts(
                await get_current_inventory_levels()
            )
            if inventory_alerts:
                await websocket.send_json({"alerts": inventory_alerts})
            await asyncio.sleep(10)
    except WebSocketDisconnect:
        logger.info("Alert WebSocket disconnected")
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)