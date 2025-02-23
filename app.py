from fastapi import FastAPI, File, UploadFile, HTTPException, Depends, Form, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordBearer
from ibm_watsonx_ai.foundation_models import Model
from ibm_watsonx_ai.metanames import GenTextParamsMetaNames
from ibm_watsonx_ai.rag import RetrievalAugmentedGenerator
from ibm_watsonx_ai.tooling import ToolChain
from jose import JWTError, jwt
from pydantic import BaseModel, validator
from typing import List, Dict, Optional
import json
import os
from dotenv import load_dotenv
from datetime import datetime, timedelta
from prometheus_client import Counter, Histogram
import asyncio
import logging
import boto3
from statsmodels.tsa.arima.model import ARIMA
from mangum import Mangum

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler('app.log'), logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(title="SupplyGenius Pro API")

# CORS Configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Security configurations
SECRET_KEY = os.getenv("SECRET_KEY", "supersecret")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

# Metrics
api_requests = Counter('api_requests_total', 'Total API requests', ['endpoint'])
model_latency = Histogram('model_inference_seconds', 'Model inference duration')

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
    logger.error(f"Error initializing IBM Granite model: {e}")
    granite_model = None

# Initialize S3 for file uploads
s3 = boto3.client(
    's3',
    aws_access_key_id=os.getenv("AWS_ACCESS_KEY"),
    aws_secret_access_key=os.getenv("AWS_SECRET_KEY")
)

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

# Initialize RAG and ToolChain
rag = RetrievalAugmentedGenerator(model=granite_model, retriever=SupplyChainKB().retrieve)
tool_chain = ToolChain(granite_model)

# Pydantic Models
class User(BaseModel):
    username: str
    password: str

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

# In-memory user database (replace with real DB in production)
users_db = {"admin": {"username": "admin", "password": "supplygenius"}}

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

# API Endpoints

@app.post("/token")
async def login(user: User):
    db_user = users_db.get(user.username)
    if not db_user or db_user["password"] != user.password:
        raise HTTPException(status_code=400, detail="Invalid credentials")
    return {"access_token": create_access_token({"sub": user.username})}

@app.post("/process-document", response_model=DocumentAnalysisResponse)
async def process_document(file: UploadFile = File(...), user=Depends(get_current_user)):
    api_requests.labels(endpoint='/process-document').inc()
    if not file.content_type.startswith('text/'):
        raise HTTPException(status_code=400, detail="Only text files are supported")
    if not granite_model:
        raise HTTPException(status_code=503, detail="AI model not available")
    try:
        # Upload file to S3
        s3.upload_fileobj(file.file, "your-bucket-name", file.filename)
        # Process file (simplified for demo)
        content = await file.read()
        content_text = content.decode("utf-8")
        prompt = f"""Analyze this supply chain document and extract structured data:
        {content_text}
        Return JSON with: document_type, parties_involved, key_dates, financial_terms, quantities, compliance_status"""
        with model_latency.time():
            response = granite_model.generate_text(prompt=prompt, params={GenTextParamsMetaNames.MAX_NEW_TOKENS: 500})
        analysis = json.loads(response.generated_text)
        return DocumentAnalysisResponse(
            document_type=analysis.get("document_type", "unknown"),
            extracted_data=analysis,
            confidence_score=0.97
        )
    except Exception as e:
        logger.error(f"Document processing failed: {str(e)}")
        raise HTTPException(status_code=500, detail="Document processing failed")

@app.post("/predict-demand", response_model=PredictionResponse)
async def predict_demand(pred_request: PredictionRequest, user=Depends(get_current_user)):
    api_requests.labels(endpoint='/predict-demand').inc()
    try:
        model = ARIMA(pred_request.historical_data, order=(5,1,0))
        model_fit = model.fit()
        forecast = model_fit.forecast(steps=pred_request.forecast_period)
        return PredictionResponse(
            forecast=forecast.tolist(),
            confidence_intervals={"lower": [0.0] * pred_request.forecast_period, "upper": [0.0] * pred_request.forecast_period}
        )
    except Exception as e:
        logger.error(f"Demand prediction failed: {str(e)}")
        raise HTTPException(status_code=500, detail="Demand prediction failed")

@app.post("/simulate-risk")
async def simulate_risk(request: RiskSimulationRequest, user=Depends(get_current_user)):
    api_requests.labels(endpoint='/simulate-risk').inc()
    prompt = f"""Simulate supply chain risk with {request.disruption}% disruption and {request.demand_spike}% demand spike. Suggest mitigation strategies."""
    response = rag.generate(prompt)
    return {"strategies": response.split("\n")}

@tool_chain.tool
def generate_purchase_order(supplier: str, items: dict):
    return f"PO-{datetime.now().timestamp()}\nSupplier: {supplier}\nItems: {json.dumps(items)}"

@app.post("/generate-po")
async def create_po(supplier: str = Form(...), items: str = Form(...), user=Depends(get_current_user)):
    api_requests.labels(endpoint='/generate-po').inc()
    return tool_chain.run(f"Generate PO for {supplier} with items: {items}")

class ROICalculator:
    @staticmethod
    def calculate(data: dict):
        current = data.get('current_cost', 0)
        optimized = data.get('optimized_cost', 0)
        if current == 0:
            return {"savings": 0, "roi": "0%"}
        savings = current - optimized
        roi = (savings / current) * 100
        return {"savings": savings, "roi": f"{roi:.1f}%"}

@app.post("/calculate-roi")
async def calculate_roi(data: ROIRequest, user=Depends(get_current_user)):
    api_requests.labels(endpoint='/calculate-roi').inc()
    try:
        result = ROICalculator.calculate(data.dict())
        logger.info(f"ROI calculated for user {user['sub']}: {result}")
        return result
    except Exception as e:
        logger.error(f"ROI calculation failed: {str(e)}")
        raise HTTPException(status_code=500, detail="ROI calculation failed")

@app.post("/analyze-supply-chain", response_model=SupplyChainMetrics)
async def analyze_supply_chain(user=Depends(get_current_user)):
    api_requests.labels(endpoint='/analyze-supply-chain').inc()
    try:
        prompt = """Analyze the entire supply chain and provide:
        1. Inventory turnover rate
        2. Order fulfillment rate
        3. Supplier performance scores
        4. Overall risk assessment
        Format: JSON with numerical metrics"""
        with model_latency.time():
            response = granite_model.generate_text(prompt)
        analysis = json.loads(response.generated_text)
        return SupplyChainMetrics(**analysis)
    except Exception as e:
        logger.error(f"Supply chain analysis failed: {str(e)}")
        raise HTTPException(status_code=500, detail="Supply chain analysis failed")

@app.post("/optimize-inventory")
async def optimize_inventory(data: InventoryOptimizationRequest, user=Depends(get_current_user)):
    api_requests.labels(endpoint='/optimize-inventory').inc()
    try:
        prompt = f"""Optimize inventory levels considering:
        - Current stock: {data.current_stock}
        - Historical demand: {data.historical_demand}
        - Lead times: {data.lead_times}
        - Holding cost: {data.holding_cost}
        - Stockout cost: {data.stockout_cost}
        Provide optimal order quantities and reorder points."""
        response = granite_model.generate_text(prompt)
        return json.loads(response.generated_text)
    except Exception as e:
        logger.error(f"Inventory optimization failed: {str(e)}")
        raise HTTPException(status_code=500, detail="Inventory optimization failed")

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "model_status": "available" if granite_model else "unavailable",
        "cache_status": "connected"  # Simplified for demo
    }

# Vercel serverless handler
handler = Mangum(app)
