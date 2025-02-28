from fastapi import FastAPI, File, UploadFile, HTTPException, Depends, Form, Request, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from ibm_watsonx_ai.foundation_models import Model
from ibm_watsonx_ai.metanames import GenTextParamsMetaNames
from ibm_watsonx_ai.rag import RetrievalAugmentedGenerator
from ibm_watsonx_ai.tooling import ToolChain
from jose import JWTError, jwt
from passlib.context import CryptContext
from pydantic import BaseModel, validator, Field
from typing import List, Dict, Optional, Any, Union
import json
import os
import uuid
from dotenv import load_dotenv
from datetime import datetime, timedelta
from prometheus_client import Counter, Histogram, generate_latest
from starlette.responses import Response
import asyncio
import logging
import boto3
from statsmodels.tsa.arima.model import ARIMA
from mangum import Mangum
from functools import lru_cache
import numpy as np

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
app = FastAPI(
    title="SupplyGenius Pro API",
    description="Advanced Supply Chain Management API with AI capabilities",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS Configuration with more restrictive origins
ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "http://localhost:3000").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["Authorization", "Content-Type"],
)

# Security configurations with improved secret handling
SECRET_KEY = os.getenv("SECRET_KEY")
if not SECRET_KEY:
    logger.warning("No SECRET_KEY set in environment variables! Using a random key.")
    SECRET_KEY = str(uuid.uuid4())

ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "30"))
S3_BUCKET_NAME = os.getenv("S3_BUCKET_NAME", "supply-genius-docs")

# Password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

# Metrics
api_requests = Counter('api_requests_total', 'Total API requests', ['endpoint', 'status'])
model_latency = Histogram('model_inference_seconds', 'Model inference duration')
request_duration = Histogram('request_duration_seconds', 'Request duration in seconds', ['endpoint'])

# Model initialization with caching
@lru_cache()
def get_granite_model():
    try:
        return Model(
            model_id="ibm/granite-13b-instruct-v1",
            credentials={
                "apikey": os.getenv("IBM_CLOUD_API_KEY"),
                "url": os.getenv("IBM_CLOUD_URL")
            },
            project_id=os.getenv("IBM_PROJECT_ID")
        )
    except Exception as e:
        logger.error(f"Error initializing IBM Granite model: {e}")
        return None

# Initialize S3 for file uploads with session
def get_s3_client():
    session = boto3.Session(
        aws_access_key_id=os.getenv("AWS_ACCESS_KEY"),
        aws_secret_access_key=os.getenv("AWS_SECRET_KEY"),
        region_name=os.getenv("AWS_REGION", "us-east-1")
    )
    return session.client('s3')

# Knowledge Base Implementation with improved data structure
class SupplyChainKB:
    def __init__(self):
        # Load data from JSON file or database in production
        self.knowledge_base = {
            "suppliers": {
                "ACME Corp": {"lead_time": 14, "rating": 4.8, "pricing": "competitive", "reliability": 0.92},
                "GlobalParts": {"lead_time": 7, "rating": 4.5, "pricing": "premium", "reliability": 0.89},
                "EcoSupply": {"lead_time": 21, "rating": 4.7, "pricing": "economy", "reliability": 0.85},
                "FastTrack": {"lead_time": 5, "rating": 4.2, "pricing": "premium", "reliability": 0.88}
            },
            "compliance_standards": {
                "ISO-9001": {"description": "Quality Management", "required": True},
                "REACH": {"description": "Chemical Registration", "required": True},
                "RoHS": {"description": "Hazardous Substances", "required": True},
                "ISO-14001": {"description": "Environmental Management", "required": False}
            },
            "risk_mitigation": {
                "supply_disruption": ["Diversify suppliers", "Safety stock", "Alternative sourcing"],
                "demand_spike": ["Flexible contracts", "Ramp up production", "Priority allocation"],
                "logistics_failure": ["Backup carriers", "Alternative routes", "Buffer shipping time"],
                "quality_issues": ["Inspection protocols", "Supplier certification", "Testing regimen"]
            }
        }

    def retrieve(self, query: str) -> str:
        if query in self.knowledge_base:
            return json.dumps(self.knowledge_base[query])
        
        # Perform fuzzy matching for more robust retrieval
        for key in self.knowledge_base:
            if query.lower() in key.lower():
                return json.dumps(self.knowledge_base[key])
        
        return json.dumps({})

# Initialize RAG and ToolChain
kb = SupplyChainKB()

# Initialize RAG with improved error handling
def get_rag():
    model = get_granite_model()
    if model:
        return RetrievalAugmentedGenerator(model=model, retriever=kb.retrieve)
    return None

# Initialize ToolChain with improved error handling
def get_tool_chain():
    model = get_granite_model()
    if model:
        return ToolChain(model)
    return None

# Pydantic Models with enhanced validation
class User(BaseModel):
    username: str
    password: str

class UserInDB(User):
    hashed_password: str
    is_active: bool = True
    role: str = "user"

class Token(BaseModel):
    access_token: str
    token_type: str

class TokenData(BaseModel):
    sub: Optional[str] = None
    role: Optional[str] = None
    exp: Optional[datetime] = None

class ROIRequest(BaseModel):
    current_cost: float = Field(..., gt=0, description="Current cost (must be greater than 0)")
    optimized_cost: float = Field(..., ge=0, description="Optimized cost (must be non-negative)")

    @validator('optimized_cost')
    def validate_optimized_cost(cls, v, values):
        if 'current_cost' in values and v > values['current_cost']:
            raise ValueError('Optimized cost should not exceed current cost')
        return v

class DocumentAnalysisResponse(BaseModel):
    document_type: str
    extracted_data: Dict[str, Any]
    confidence_score: float = Field(..., ge=0, le=1)
    processing_time: float

class PredictionRequest(BaseModel):
    historical_data: List[float] = Field(..., min_items=10, description="At least 10 historical data points required")
    forecast_period: int = Field(..., gt=0, le=52, description="Forecast period (1-52 weeks)")

class PredictionResponse(BaseModel):
    forecast: List[float]
    confidence_intervals: Dict[str, List[float]]
    accuracy_metrics: Dict[str, float]

class RiskSimulationRequest(BaseModel):
    disruption: float = Field(..., ge=0, le=100, description="Disruption percentage (0-100)")
    demand_spike: float = Field(..., ge=0, le=100, description="Demand spike percentage (0-100)")
    scenario: Optional[str] = Field(None, description="Optional scenario name")

class SupplyChainMetrics(BaseModel):
    inventory_turnover: float
    order_fulfillment_rate: float
    supplier_performance: Dict[str, float]
    risk_score: float
    timestamp: datetime = Field(default_factory=datetime.now)

class InventoryOptimizationRequest(BaseModel):
    current_stock: Dict[str, int]
    historical_demand: List[Dict[str, float]]
    lead_times: Dict[str, int]
    holding_cost: float = Field(..., gt=0)
    stockout_cost: float = Field(..., gt=0)

class PORequest(BaseModel):
    supplier: str
    items: Dict[str, Union[int, float]]
    delivery_date: Optional[datetime] = None

# A more secure in-memory user database (replace with real DB in production)
users_db = {
    "admin": {
        "username": "admin",
        "hashed_password": pwd_context.hash("supplygenius"),
        "is_active": True,
        "role": "admin"
    }
}

# Authentication Functions
def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)

def get_user(username: str):
    if username in users_db:
        user_dict = users_db[username]
        return UserInDB(**user_dict)
    return None

def authenticate_user(username: str, password: str):
    user = get_user(username)
    if not user or not verify_password(password, user.hashed_password):
        return False
    return user

def create_access_token(data: dict, expires_delta: timedelta = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

async def get_current_user(token: str = Depends(oauth2_scheme)):
    credentials_exception = HTTPException(
        status_code=401,
        detail="Invalid authentication credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username = payload.get("sub")
        if username is None:
            raise credentials_exception
        token_data = TokenData(sub=username, role=payload.get("role"), exp=payload.get("exp"))
    except JWTError:
        raise credentials_exception
    
    # Check token expiration
    if token_data.exp and datetime.fromtimestamp(token_data.exp) < datetime.utcnow():
        raise credentials_exception
        
    user = get_user(username=token_data.sub)
    if user is None:
        raise credentials_exception
    if not user.is_active:
        raise HTTPException(status_code=400, detail="Inactive user")
    return {"sub": user.username, "role": user.role}

# Middleware for request timing and logging
@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    start_time = datetime.now()
    response = await call_next(request)
    process_time = (datetime.now() - start_time).total_seconds()
    
    # Record request duration
    try:
        endpoint = request.url.path
        request_duration.labels(endpoint=endpoint).observe(process_time)
        response.headers["X-Process-Time"] = str(process_time)
    except:
        pass
    
    return response

# API Endpoints
@app.post("/token", response_model=Token)
async def login_for_access_token(form_data: OAuth2PasswordRequestForm = Depends()):
    user = authenticate_user(form_data.username, form_data.password)
    if not user:
        api_requests.labels(endpoint='/token', status='failure').inc()
        raise HTTPException(
            status_code=401,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    # Get user role for token
    user_data = users_db.get(form_data.username, {})
    role = user_data.get("role", "user")
    
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user.username, "role": role}, 
        expires_delta=access_token_expires
    )
    
    api_requests.labels(endpoint='/token', status='success').inc()
    return {"access_token": access_token, "token_type": "bearer"}

@app.post("/process-document", response_model=DocumentAnalysisResponse)
async def process_document(
    file: UploadFile = File(...), 
    background_tasks: BackgroundTasks = None,
    user=Depends(get_current_user)
):
    api_requests.labels(endpoint='/process-document', status='processing').inc()
    start_time = datetime.now()
    
    # Validate file type
    valid_content_types = ["text/plain", "application/pdf", "application/json", "application/vnd.openxmlformats-officedocument.wordprocessingml.document"]
    if file.content_type not in valid_content_types:
        api_requests.labels(endpoint='/process-document', status='failure').inc()
        raise HTTPException(status_code=400, detail=f"Unsupported file type. Supported types: {', '.join(valid_content_types)}")
    
    # Check model availability
    model = get_granite_model()
    if not model:
        api_requests.labels(endpoint='/process-document', status='failure').inc()
        raise HTTPException(status_code=503, detail="AI model not available")
    
    try:
        # Generate a unique file name to prevent overwriting
        file_ext = os.path.splitext(file.filename)[1]
        unique_filename = f"{str(uuid.uuid4())}{file_ext}"
        
        # Upload file to S3
        s3 = get_s3_client()
        file_content = await file.read()
        s3.put_object(
            Bucket=S3_BUCKET_NAME,
            Key=unique_filename,
            Body=file_content,
            ContentType=file.content_type,
            Metadata={
                "username": user["sub"],
                "upload_time": datetime.now().isoformat()
            }
        )
        
        # Process file content
        content_text = ""
        if file.content_type == "text/plain":
            content_text = file_content.decode("utf-8")
        else:
            # For production, use appropriate parsers for other document types
            content_text = f"[Content extracted from {file.filename} - {file.content_type}]"
            
        # Create a more detailed prompt
        prompt = f"""
        Analyze this supply chain document and extract structured information:
        
        {content_text[:5000]}  # Limit text to avoid token limits
        
        Return a valid JSON object with the following fields:
        - document_type (string): Type of document (PO, invoice, shipping manifest, etc.)
        - parties_involved (object): All organizations mentioned with their roles
        - key_dates (object): All relevant dates mentioned (order date, delivery date, etc.)
        - financial_terms (object): Payment terms, prices, totals, currencies
        - quantities (object): Item quantities, units, and SKUs
        - compliance_status (object): Any regulatory or compliance information
        - risks (array): Potential risks identified in the document
        """
        
        # Log processing start
        logger.info(f"Processing document {unique_filename} for user {user['sub']}")
        
        # Generate analysis with timing
        with model_latency.time():
            response = model.generate_text(
                prompt=prompt, 
                params={
                    GenTextParamsMetaNames.MAX_NEW_TOKENS: 1000,
                    GenTextParamsMetaNames.TEMPERATURE: 0.1,
                    GenTextParamsMetaNames.STOP_SEQUENCES: ["}"]
                }
            )
        
        # Parse the generated text as JSON, handling potential JSON errors
        try:
            # Make sure the JSON is properly closed
            raw_text = response.generated_text
            if not raw_text.strip().endswith("}"):
                raw_text += "}"
            analysis = json.loads(raw_text)
        except json.JSONDecodeError:
            # Fallback for malformed JSON
            logger.warning(f"Malformed JSON from model: {response.generated_text}")
            analysis = {
                "document_type": "unknown",
                "error": "Could not parse document structure",
                "raw_text": response.generated_text[:100] + "..."
            }
        
        # Calculate processing time
        processing_time = (datetime.now() - start_time).total_seconds()
        
        # Update file metadata with results if needed
        if background_tasks:
            background_tasks.add_task(
                update_document_metadata,
                unique_filename, 
                {"analysis_complete": True, "document_type": analysis.get("document_type", "unknown")}
            )
        
        api_requests.labels(endpoint='/process-document', status='success').inc()
        return DocumentAnalysisResponse(
            document_type=analysis.get("document_type", "unknown"),
            extracted_data=analysis,
            confidence_score=0.92,  # In production, derive this from model
            processing_time=processing_time
        )
    except Exception as e:
        logger.error(f"Document processing failed: {str(e)}", exc_info=True)
        api_requests.labels(endpoint='/process-document', status='failure').inc()
        raise HTTPException(status_code=500, detail=f"Document processing failed: {str(e)}")

# Function to update document metadata in background
async def update_document_metadata(filename, metadata_dict):
    try:
        s3 = get_s3_client()
        # Get existing metadata
        response = s3.head_object(Bucket=S3_BUCKET_NAME, Key=filename)
        existing_metadata = response.get('Metadata', {})
        
        # Update metadata
        updated_metadata = {**existing_metadata, **metadata_dict}
        
        # Copy object to itself with updated metadata
        s3.copy_object(
            Bucket=S3_BUCKET_NAME,
            CopySource={'Bucket': S3_BUCKET_NAME, 'Key': filename},
            Key=filename,
            Metadata=updated_metadata,
            MetadataDirective='REPLACE'
        )
    except Exception as e:
        logger.error(f"Failed to update document metadata: {str(e)}")

@app.post("/predict-demand", response_model=PredictionResponse)
async def predict_demand(pred_request: PredictionRequest, user=Depends(get_current_user)):
    api_requests.labels(endpoint='/predict-demand', status='processing').inc()
    
    try:
        # Ensure we have enough data points
        if len(pred_request.historical_data) < 10:
            raise HTTPException(status_code=400, detail="At least 10 historical data points are required")
        
        # Perform ARIMA forecasting
        # In production, use a more sophisticated model selection approach
        best_aic = float('inf')
        best_model = None
        best_order = None
        
        # Try a few different models and select the best one
        for p in range(0, 3):
            for d in range(0, 2):
                for q in range(0, 3):
                    try:
                        model = ARIMA(pred_request.historical_data, order=(p, d, q))
                        model_fit = model.fit()
                        if model_fit.aic < best_aic:
                            best_aic = model_fit.aic
                            best_model = model_fit
                            best_order = (p, d, q)
                    except:
                        continue
        
        if best_model is None:
            # Fallback to a simple model if automated selection fails
            model = ARIMA(pred_request.historical_data, order=(1, 1, 0))
            best_model = model.fit()
            best_order = (1, 1, 0)
        
        # Generate forecast
        forecast_result = best_model.get_forecast(steps=pred_request.forecast_period)
        forecast_mean = forecast_result.predicted_mean
        forecast_conf = forecast_result.conf_int(alpha=0.05)  # 95% confidence interval
        
        # Calculate accuracy metrics using last 30% of historical data
        split_point = int(len(pred_request.historical_data) * 0.7)
        train_data = pred_request.historical_data[:split_point]
        test_data = pred_request.historical_data[split_point:]
        
        validation_model = ARIMA(train_data, order=best_order)
        validation_fit = validation_model.fit()
        validation_forecast = validation_fit.forecast(steps=len(test_data))
        
        # Calculate error metrics
        mae = np.mean(np.abs(validation_forecast - test_data))
        mape = np.mean(np.abs((validation_forecast - test_data) / test_data)) * 100 if any(x != 0 for x in test_data) else 0
        rmse = np.sqrt(np.mean((validation_forecast - test_data) ** 2))
        
        api_requests.labels(endpoint='/predict-demand', status='success').inc()
        return PredictionResponse(
            forecast=forecast_mean.tolist(),
            confidence_intervals={
                "lower": forecast_conf[:, 0].tolist(),
                "upper": forecast_conf[:, 1].tolist()
            },
            accuracy_metrics={
                "mae": float(mae),
                "mape": float(mape),
                "rmse": float(rmse),
                "aic": float(best_model.aic),
                "bic": float(best_model.bic),
                "model_order": best_order
            }
        )
    except Exception as e:
        logger.error(f"Demand prediction failed: {str(e)}", exc_info=True)
        api_requests.labels(endpoint='/predict-demand', status='failure').inc()
        raise HTTPException(status_code=500, detail=f"Demand prediction failed: {str(e)}")

@app.post("/simulate-risk")
async def simulate_risk(request: RiskSimulationRequest, user=Depends(get_current_user)):
    api_requests.labels(endpoint='/simulate-risk', status='processing').inc()
    
    try:
        # Get RAG with error handling
        rag_instance = get_rag()
        if not rag_instance:
            raise HTTPException(status_code=503, detail="RAG service not available")
        
        # Create a more detailed prompt
        prompt = f"""
        Simulate a supply chain risk scenario with:
        - {request.disruption}% supplier disruption probability
        - {request.demand_spike}% unexpected demand spike
        {f"- Scenario: {request.scenario}" if request.scenario else ""}
        
        Provide a structured risk assessment with the following sections:
        1. Risk impact (high/medium/low for different aspects)
        2. Probability of disruption
        3. Expected financial impact
        4. Specific mitigation strategies
        5. Recommended immediate actions
        
        Format your response as a structured list of recommendations.
        """
        
        # Generate response with RAG
        with model_latency.time():
            response = rag_instance.generate(prompt)
        
        # Process the response into a structured format
        strategies = [line.strip() for line in response.split("\n") if line.strip()]
        
        # Group strategies by category
        result = {
            "risk_assessment": {
                "disruption_probability": request.disruption / 100,
                "demand_spike_impact": request.demand_spike / 100,
                "overall_risk_level": calculate_risk_level(request.disruption, request.demand_spike)
            },
            "mitigation_strategies": strategies
        }
        
        api_requests.labels(endpoint='/simulate-risk', status='success').inc()
        return result
    except Exception as e:
        logger.error(f"Risk simulation failed: {str(e)}", exc_info=True)
        api_requests.labels(endpoint='/simulate-risk', status='failure').inc()
        raise HTTPException(status_code=500, detail=f"Risk simulation failed: {str(e)}")

def calculate_risk_level(disruption, demand_spike):
    """Calculate overall risk level based on disruption and demand spike"""
    combined_risk = (disruption + demand_spike) / 2
    if combined_risk > 75:
        return "HIGH"
    elif combined_risk > 40:
        return "MEDIUM"
    else:
        return "LOW"

# Tool definition improved with structured output
def generate_purchase_order(supplier: str, items: Dict[str, Any], delivery_date=None):
    """Generate a structured purchase order with validation"""
    # Verify supplier exists in knowledge base
    kb = SupplyChainKB()
    suppliers = json.loads(kb.retrieve("suppliers"))
    
    supplier_info = suppliers.get(supplier, {})
    po_id = f"PO-{uuid.uuid4().hex[:8].upper()}"
    
    if delivery_date is None:
        # Calculate default delivery date based on supplier lead time
        lead_time = supplier_info.get("lead_time", 14)
        delivery_date = (datetime.now() + timedelta(days=lead_time)).strftime("%Y-%m-%d")
    elif isinstance(delivery_date, datetime):
        delivery_date = delivery_date.strftime("%Y-%m-%d")
    
    total_amount = sum(qty * price for item, (qty, price) in items.items()) if isinstance(next(iter(items.values())), tuple) else sum(items.values())
    
    return {
        "po_id": po_id,
        "supplier": supplier,
        "supplier_rating": supplier_info.get("rating", "N/A"),
        "issue_date": datetime.now().strftime("%Y-%m-%d"),
        "delivery_date": delivery_date,
        "items": items,
        "total_amount": total_amount,
        "terms": "Net 30",
        "status": "Draft"
    }

@app.post("/generate-po")
async def create_po(po_request: PORequest, user=Depends(get_current_user)):
    api_requests.labels(endpoint='/generate-po', status='processing').inc()
    
    try:
        # Get toolchain
        tool_chain = get_tool_chain()
        if not tool_chain:
            # Fallback implementation if toolchain not available
            result = generate_purchase_order(
                po_request.supplier, 
                po_request.items,
                po_request.delivery_date
            )
        else:
            # Register our tool
            @tool_chain.tool
            def generate_po(supplier: str, items: dict, delivery_date=None):
                return generate_purchase_order(supplier, items, delivery_date)
            
            # Use toolchain to execute
            items_json = json.dumps(po_request.items)
            delivery_date_str = po_request.delivery_date.isoformat() if po_request.delivery_date else None
            command = f"generate_po('{po_request.supplier}', {items_json}, '{delivery_date_str}')" if delivery_date_str else f"generate_po('{po_request.supplier}', {items_json})"
            result = tool_chain.run(command)
        
        # Log PO creation
        logger.info(f"PO generated for supplier {po_request.supplier} by user {user['sub']}")
        
        api_requests.labels(endpoint='/generate-po', status='success').inc()
        return result
    except Exception as e:
        logger.error(f"PO generation failed: {str(e)}", exc_info=True)
        api_requests.labels(endpoint='/generate-po', status='failure').inc()
        raise HTTPException(status_code=500, detail=f"PO generation failed: {str(e)}")

class ROICalculator:
    @staticmethod
    def calculate(data: dict):
        """Calculate ROI with improved validation and error handling"""
        try:
            current = float(data.get('current_cost', 0))
            optimized = float(data.get('optimized_cost', 0))
            
            if current <= 0:
                return {"error": "Current cost must be greater than zero", "savings": 0, "roi": "0%"}
            
            if optimized < 0:
                return {"error": "Optimized cost cannot be negative", "savings": 0, "roi": "0%"}
                
            if optimized > current:
                return {"error": "Optimized cost should not exceed current cost", "savings": 0, "roi": "0%"}
                
            savings = current - optimized
            roi = (savings / current) * 100
            payback_period = None
            
            if 'implementation_cost' in data and data['implementation_cost'] > 0:
                impl_cost = float(data['implementation_cost'])
                annual_savings = savings * 12  # Assuming monthly costs
                payback_period = impl_cost / annual_savings
                
            result = {
                "savings": round(savings, 2),
                "roi": f"{roi:.1f}%",
                "roi_numeric": round(roi, 2)
            }
            
            if payback_period is not None:
                result["payback_period_years"] = round(payback_period, 2)
                
            return result
        except (ValueError, TypeError, ZeroDivisionError) as e:
            return {"error": f"Calculation error: {str(e)}", "savings": 0, "roi": "0%"}

@app.post("/calculate-roi")
async def calculate_roi(data: ROIRequest, user=Depends(get_current_user)):
    api_requests.labels(endpoint='/calculate-roi', status='processing').inc()
    
    try:
        result = ROICalculator.calculate(data.dict())
        
        # Log successful calculation
        logger.info(f"ROI calculated for user {user['sub']}: {result}")
        
        api_requests.labels(endpoint='/calculate-roi', status='success').inc()
        return result
    except Exception as e:
        logger.error(f"ROI calculation failed: {str(e)}", exc_info=True)
        api_requests.labels(endpoint='/calculate-roi', status='failure').inc()
        raise HTTPException(status_code=500, detail=f"ROI calculation failed: {str(e)}")

@app.post("/analyze-supply-chain", response_model=SupplyChainMetrics)
async def analyze_supply_chain(user=Depends(get_current_user)):
    api_requests.labels(endpoint='/analyze-supply-chain', status='processing').inc()

@app.post("/analyze-supply-chain", response_model=SupplyChainMetrics)
async def analyze_supply_chain(user=Depends(get_current_user)):
    api_requests.labels(endpoint='/analyze-supply-chain', status='processing').inc()
    
    # Check if user has admin role
    if user.get("role") != "admin":
        api_requests.labels(endpoint='/analyze-supply-chain', status='failure').inc()
        raise HTTPException(status_code=403, detail="Admin access required for this endpoint")
    
    try:
        # In production, this would connect to actual supply chain data sources
        # Here we're using sample data for demonstration
        metrics = SupplyChainMetrics(
            inventory_turnover=5.8,
            order_fulfillment_rate=0.93,
            supplier_performance={
                "ACME Corp": 0.92,
                "GlobalParts": 0.89,
                "EcoSupply": 0.85,
                "FastTrack": 0.88
            },
            risk_score=0.28
        )
        
        api_requests.labels(endpoint='/analyze-supply-chain', status='success').inc()
        return metrics
    except Exception as e:
        logger.error(f"Supply chain analysis failed: {str(e)}", exc_info=True)
        api_requests.labels(endpoint='/analyze-supply-chain', status='failure').inc()
        raise HTTPException(status_code=500, detail=f"Supply chain analysis failed: {str(e)}")

@app.post("/optimize-inventory")
async def optimize_inventory(request: InventoryOptimizationRequest, user=Depends(get_current_user)):
    api_requests.labels(endpoint='/optimize-inventory', status='processing').inc()
    
    try:
        # Get the RAG instance
        rag_instance = get_rag()
        if not rag_instance:
            raise HTTPException(status_code=503, detail="RAG service not available")
        
        # Create a detailed context for inventory optimization
        prompt = f"""
        Optimize inventory levels for the following items based on:
        
        Current stock levels: {json.dumps(request.current_stock)}
        Historical demand data: {json.dumps(request.historical_demand)}
        Supplier lead times: {json.dumps(request.lead_times)}
        Holding cost per unit: {request.holding_cost}
        Stockout cost per unit: {request.stockout_cost}
        
        For each item, calculate and provide:
        1. Optimal reorder point
        2. Economic order quantity (EOQ)
        3. Safety stock level
        4. Recommended inventory level
        5. Expected service level
        
        Format your response as a JSON object where each key is an item name and the value is another 
        object containing the optimization parameters.
        """
        
        # Generate optimization with RAG
        with model_latency.time():
            response = rag_instance.generate(prompt)
        
        # Parse the response as JSON
        try:
            # Ensure the response is valid JSON
            if not response.strip().startswith("{"):
                response = "{" + response.split("{", 1)[1]
            if not response.strip().endswith("}"):
                response = response.rsplit("}", 1)[0] + "}"
                
            result = json.loads(response)
        except json.JSONDecodeError:
            # Fallback for malformed JSON
            logger.warning(f"Malformed JSON from model: {response}")
            
            # Create a simple fallback response
            result = {}
            for item in request.current_stock:
                # Simple EOQ calculation as fallback
                avg_demand = sum(period.get(item, 0) for period in request.historical_demand) / len(request.historical_demand)
                lead_time = request.lead_times.get(item, 14)
                
                result[item] = {
                    "reorder_point": int(avg_demand * lead_time * 1.5),
                    "economic_order_quantity": int(((2 * avg_demand * 365) * request.stockout_cost / request.holding_cost) ** 0.5),
                    "safety_stock": int(avg_demand * lead_time * 0.5),
                    "recommended_level": int(avg_demand * lead_time * 2),
                    "service_level": 0.95,
                    "note": "Calculated using fallback method due to processing error"
                }
        
        api_requests.labels(endpoint='/optimize-inventory', status='success').inc()
        return {
            "optimized_inventory": result,
            "total_holding_cost": sum(level["recommended_level"] * request.holding_cost for item, level in result.items()),
            "expected_service_level": sum(level["service_level"] for item, level in result.items()) / len(result) if result else 0
        }
    except Exception as e:
        logger.error(f"Inventory optimization failed: {str(e)}", exc_info=True)
        api_requests.labels(endpoint='/optimize-inventory', status='failure').inc()
        raise HTTPException(status_code=500, detail=f"Inventory optimization failed: {str(e)}")

@app.get("/metrics")
async def metrics(user=Depends(get_current_user)):
    # Only admin users can access metrics
    if user.get("role") != "admin":
        raise HTTPException(status_code=403, detail="Admin access required for metrics")
    
    return Response(generate_latest(), media_type="text/plain")

@app.get("/health")
async def health_check():
    # Basic health check endpoint
    services_status = {
        "api": "healthy",
        "model_service": "healthy" if get_granite_model() else "degraded",
        "database": "healthy"  # In production, check actual DB connection
    }
    
    overall_status = "healthy" if all(status == "healthy" for status in services_status.values()) else "degraded"
    
    return {
        "status": overall_status,
        "version": "1.0.0",
        "services": services_status,
        "timestamp": datetime.now().isoformat()
    }

# For deployment to AWS Lambda with API Gateway
handler = Mangum(app)

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run(app, host="0.0.0.0", port=port)
