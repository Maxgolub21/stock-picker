import streamlit as st
import os
import json
import hashlib
from typing import List, Optional
from datetime import datetime, timedelta
from pathlib import Path
from google.adk.models.lite_llm import LiteLlm
from google.adk.agents import Agent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from dotenv import load_dotenv
from pydantic import BaseModel, Field
import asyncio      
import uuid
import logging
import requests
from functools import wraps
import time
from collections import deque

load_dotenv(override=True)

import pandas as pd

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="Stock Analysis Agent",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .stock-card {
        padding: 1.5rem;
        border-radius: 10px;
        border: 2px solid #e0e0e0;
        margin-bottom: 1rem;
        background-color: #f8f9fa;
        font-size: 1rem;
        color: #31333f;
    }
    .buy-stock {
        background-color: #d4edda;
        border-color: #31333f;
    }
    .pass-stock {
        background-color: #f8d7da;
        border-color: #dc3545;
    }
    .stats-box {
        padding: 1rem;
        border-radius: 5px;
        background-color: #e7f3ff;
        border-left: 4px solid #1f77b4;
    }
</style>
""", unsafe_allow_html=True)

# Pydantic Models
class StockRecommendation(BaseModel):
    stock_ticker: str
    stock_price: float
    reasoning: str
    buy_stock: bool

class StockList(BaseModel):
    recommendations: List[StockRecommendation]

# =============================================================================
# DIRECT API TOOL IMPLEMENTATIONS (No MCP Required)
# =============================================================================

class RateLimiter:
    """Simple rate limiter for API calls"""
    
    def __init__(self, calls_per_minute: int = 5):
        self.calls_per_minute = calls_per_minute
        self.calls = deque()
        self.cache = {}  # Simple cache for repeated calls
        self.cache_duration = 300  # 5 minutes
    
    def wait_if_needed(self):
        """Wait if we've exceeded rate limit"""
        now = time.time()
        
        # Remove calls older than 1 minute
        while self.calls and self.calls[0] < now - 60:
            self.calls.popleft()
        
        # If we've hit the limit, wait
        if len(self.calls) >= self.calls_per_minute:
            sleep_time = 60 - (now - self.calls[0]) + 0.5  # Add 0.5s buffer
            if sleep_time > 0:
                logger.info(f"Rate limit reached. Waiting {sleep_time:.1f}s...")
                time.sleep(sleep_time)
                # Clean old calls after waiting
                now = time.time()
                while self.calls and self.calls[0] < now - 60:
                    self.calls.popleft()
        
        # Record this call
        self.calls.append(time.time())
    
    def get_cached(self, key: str):
        """Get cached result if available and fresh"""
        if key in self.cache:
            result, timestamp = self.cache[key]
            if time.time() - timestamp < self.cache_duration:
                logger.info(f"Using cached result for: {key}")
                return result
            else:
                del self.cache[key]
        return None
    
    def set_cached(self, key: str, result):
        """Cache a result"""
        self.cache[key] = (result, time.time())


class PolygonTool:
    """Direct Polygon.io API integration with rate limiting"""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://api.polygon.io"
        self.rate_limiter = RateLimiter(calls_per_minute=5)  # Free tier limit
    
    def search_tickers(self, search_term: str, limit: int = 10) -> dict:
        """Search for ticker symbols by company name or ticker"""
        try:
            search_term = search_term.strip()
            cache_key = f"search:{search_term}:{limit}"
            
            # Check cache first
            cached = self.rate_limiter.get_cached(cache_key)
            if cached:
                return cached
            
            # Wait if needed for rate limit
            self.rate_limiter.wait_if_needed()
            
            url = f"{self.base_url}/v3/reference/tickers"
            params = {
                "apiKey": self.api_key,
                "search": search_term,
                "active": "true",
                "limit": limit,
                "market": "stocks"
            }
            
            response = requests.get(url, params=params, timeout=10)
            
            if response.status_code == 404:
                result = {
                    "search_term": search_term,
                    "error": "No tickers found",
                    "success": False,
                    "results": []
                }
                self.rate_limiter.set_cached(cache_key, result)
                return result
            
            if response.status_code == 429:
                logger.warning(f"Rate limit hit for search: {search_term}")
                return {
                    "search_term": search_term,
                    "error": "Rate limit exceeded. Please wait a moment and try again.",
                    "success": False,
                    "results": []
                }
            
            response.raise_for_status()
            data = response.json()
            
            results = []
            for item in data.get("results", []):
                results.append({
                    "ticker": item.get("ticker"),
                    "name": item.get("name"),
                    "market": item.get("market"),
                    "type": item.get("type"),
                    "locale": item.get("locale")
                })
            
            result = {
                "search_term": search_term,
                "results": results,
                "count": len(results),
                "success": True
            }
            
            # Cache the result
            self.rate_limiter.set_cached(cache_key, result)
            return result
            
        except Exception as e:
            logger.error(f"Polygon ticker search error for '{search_term}': {e}")
            return {
                "search_term": search_term,
                "error": str(e),
                "success": False,
                "results": []
            }
    
    def get_stock_price(self, ticker: str) -> dict:
        """Get current stock price for a ticker"""
        try:
            # Clean and uppercase ticker
            ticker = ticker.strip().upper()
            cache_key = f"price:{ticker}"
            
            # Check cache first
            cached = self.rate_limiter.get_cached(cache_key)
            if cached:
                return cached
            
            # Wait if needed for rate limit
            self.rate_limiter.wait_if_needed()
            
            # Get previous close (most reliable for current price)
            url = f"{self.base_url}/v2/aggs/ticker/{ticker}/prev"
            params = {"apiKey": self.api_key}
            
            response = requests.get(url, params=params, timeout=10)
            
            # Handle 404 specifically
            if response.status_code == 404:
                result = {
                    "ticker": ticker,
                    "error": f"Ticker '{ticker}' not found. Please verify the ticker symbol is valid and traded on US exchanges.",
                    "success": False
                }
                self.rate_limiter.set_cached(cache_key, result)
                return result
            
            # Handle rate limit
            if response.status_code == 429:
                logger.warning(f"Rate limit hit for ticker: {ticker}")
                return {
                    "ticker": ticker,
                    "error": "Rate limit exceeded. Please wait and try again.",
                    "success": False,
                    "rate_limited": True
                }
            
            response.raise_for_status()
            data = response.json()
            
            if data.get("results"):
                result_data = data["results"][0]
                result = {
                    "ticker": ticker,
                    "price": result_data.get("c"),  # Close price
                    "volume": result_data.get("v"),
                    "high": result_data.get("h"),
                    "low": result_data.get("l"),
                    "timestamp": result_data.get("t"),
                    "success": True
                }
                # Cache successful results
                self.rate_limiter.set_cached(cache_key, result)
                return result
            else:
                result = {
                    "ticker": ticker,
                    "error": f"No price data available for {ticker}",
                    "success": False
                }
                self.rate_limiter.set_cached(cache_key, result)
                return result
                
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 404:
                result = {
                    "ticker": ticker,
                    "error": f"Ticker '{ticker}' not found",
                    "success": False
                }
                self.rate_limiter.set_cached(cache_key, result)
                return result
            elif e.response.status_code == 429:
                return {
                    "ticker": ticker,
                    "error": "Rate limit exceeded",
                    "success": False,
                    "rate_limited": True
                }
            logger.error(f"Polygon API HTTP error for {ticker}: {e}")
            return {"ticker": ticker, "error": f"API error: {str(e)}", "success": False}
        except Exception as e:
            logger.error(f"Polygon API error for {ticker}: {e}")
            return {"ticker": ticker, "error": str(e), "success": False}
    
    def get_company_info(self, ticker: str) -> dict:
        """Get company details"""
        try:
            # Clean and uppercase ticker
            ticker = ticker.strip().upper()
            cache_key = f"info:{ticker}"
            
            # Check cache first
            cached = self.rate_limiter.get_cached(cache_key)
            if cached:
                return cached
            
            # Wait if needed for rate limit
            self.rate_limiter.wait_if_needed()
            
            url = f"{self.base_url}/v3/reference/tickers/{ticker}"
            params = {"apiKey": self.api_key}
            
            response = requests.get(url, params=params, timeout=10)
            
            # Handle 404 specifically
            if response.status_code == 404:
                result = {
                    "ticker": ticker,
                    "error": f"Ticker '{ticker}' not found. Please verify the ticker symbol.",
                    "success": False
                }
                self.rate_limiter.set_cached(cache_key, result)
                return result
            
            # Handle rate limit
            if response.status_code == 429:
                logger.warning(f"Rate limit hit for company info: {ticker}")
                return {
                    "ticker": ticker,
                    "error": "Rate limit exceeded. Please wait and try again.",
                    "success": False,
                    "rate_limited": True
                }
            
            response.raise_for_status()
            data = response.json()
            
            if data.get("results"):
                result_data = data["results"]
                result = {
                    "ticker": ticker,
                    "name": result_data.get("name"),
                    "market": result_data.get("market"),
                    "locale": result_data.get("locale"),
                    "type": result_data.get("type"),
                    "currency": result_data.get("currency_name"),
                    "description": result_data.get("description"),
                    "success": True
                }
                # Cache successful results
                self.rate_limiter.set_cached(cache_key, result)
                return result
            else:
                result = {
                    "ticker": ticker,
                    "error": f"No company info available for {ticker}",
                    "success": False
                }
                self.rate_limiter.set_cached(cache_key, result)
                return result
                
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 404:
                result = {
                    "ticker": ticker,
                    "error": f"Ticker '{ticker}' not found",
                    "success": False
                }
                self.rate_limiter.set_cached(cache_key, result)
                return result
            elif e.response.status_code == 429:
                return {
                    "ticker": ticker,
                    "error": "Rate limit exceeded",
                    "success": False,
                    "rate_limited": True
                }
            logger.error(f"Polygon company info error for {ticker}: {e}")
            return {"ticker": ticker, "error": f"API error: {str(e)}", "success": False}
        except Exception as e:
            logger.error(f"Polygon company info error for {ticker}: {e}")
            return {"ticker": ticker, "error": str(e), "success": False}


class BraveSearchTool:
    """Direct Brave Search API integration"""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://api.search.brave.com/res/v1/web/search"
    
    def search(self, query: str, count: int = 5) -> dict:
        """Search the web using Brave Search"""
        try:
            headers = {
                "Accept": "application/json",
                "X-Subscription-Token": self.api_key
            }
            
            params = {
                "q": query,
                "count": count
            }
            
            response = requests.get(
                self.base_url,
                headers=headers,
                params=params,
                timeout=10
            )
            response.raise_for_status()
            
            data = response.json()
            
            # Extract relevant results
            results = []
            for item in data.get("web", {}).get("results", []):
                results.append({
                    "title": item.get("title"),
                    "url": item.get("url"),
                    "description": item.get("description"),
                })
            
            return {
                "query": query,
                "results": results,
                "total": len(results)
            }
            
        except Exception as e:
            logger.error(f"Brave Search error for '{query}': {e}")
            return {"error": str(e), "results": []}


class WebFetchTool:
    """Simple web page fetcher"""
    
    def fetch(self, url: str) -> dict:
        """Fetch content from a URL"""
        try:
            headers = {
                "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"
            }
            
            response = requests.get(url, headers=headers, timeout=15)
            response.raise_for_status()
            
            return {
                "url": url,
                "status": response.status_code,
                "content": response.text[:5000],  # First 5000 chars
                "content_type": response.headers.get("content-type", "")
            }
            
        except Exception as e:
            logger.error(f"Web fetch error for {url}: {e}")
            return {"error": str(e)}


class FileSystemTool:
    """Simple filesystem operations"""
    
    def __init__(self, base_path: str = "./analysis_outputs"):
        self.base_path = Path(base_path).resolve()
        self.base_path.mkdir(parents=True, exist_ok=True)
    
    def write_file(self, filename: str, content: str) -> dict:
        """Write content to a file"""
        try:
            filepath = self.base_path / filename
            filepath.write_text(content)
            return {
                "success": True,
                "path": str(filepath),
                "size": len(content)
            }
        except Exception as e:
            logger.error(f"File write error: {e}")
            return {"error": str(e)}
    
    def read_file(self, filename: str) -> dict:
        """Read content from a file"""
        try:
            filepath = self.base_path / filename
            content = filepath.read_text()
            return {
                "success": True,
                "path": str(filepath),
                "content": content
            }
        except Exception as e:
            logger.error(f"File read error: {e}")
            return {"error": str(e)}
    
    def list_files(self) -> dict:
        """List all files in the output directory"""
        try:
            files = [f.name for f in self.base_path.iterdir() if f.is_file()]
            return {
                "success": True,
                "files": files,
                "count": len(files)
            }
        except Exception as e:
            logger.error(f"File list error: {e}")
            return {"error": str(e)}


# =============================================================================
# TOOL WRAPPER FOR ADK INTEGRATION
# =============================================================================

def create_direct_tools():
    """Create tool instances that can be used by the agent"""
    tools_dict = {}
    
    # Polygon tool
    polygon_key = os.getenv("POLYGON_API_KEY")
    if polygon_key:
        tools_dict['polygon'] = PolygonTool(polygon_key)
        st.sidebar.success("✅ Polygon API initialized")
    else:
        st.sidebar.warning("⚠️ POLYGON_API_KEY not found")
    
    # Brave search tool
    brave_key = os.getenv("BRAVE_API_KEY")
    if brave_key:
        tools_dict['brave'] = BraveSearchTool(brave_key)
        st.sidebar.success("✅ Brave Search initialized")
    else:
        st.sidebar.warning("⚠️ BRAVE_API_KEY not found")
    
    # Web fetch tool (no API key needed)
    tools_dict['fetch'] = WebFetchTool()
    st.sidebar.success("✅ Web Fetch initialized")
    
    # Filesystem tool (no API key needed)
    tools_dict['filesystem'] = FileSystemTool()
    st.sidebar.success("✅ Filesystem initialized")
    
    return tools_dict


# =============================================================================
# CACHE MANAGER
# =============================================================================

class CacheManager:
    """Manage caching of analysis results"""
    
    def __init__(self, cache_dir: str = "./cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.cache_duration = timedelta(hours=6)
    
    def _generate_cache_key(self, strategy: str) -> str:
        return hashlib.md5(strategy.encode()).hexdigest()
    
    def _get_cache_path(self, cache_key: str) -> Path:
        return self.cache_dir / f"{cache_key}.json"
    
    def get(self, strategy: str) -> Optional[dict]:
        cache_key = self._generate_cache_key(strategy)
        cache_path = self._get_cache_path(cache_key)
        
        if not cache_path.exists():
            return None
        
        try:
            with open(cache_path, 'r') as f:
                cached_data = json.load(f)
            
            cached_time = datetime.fromisoformat(cached_data['timestamp'])
            if datetime.now() - cached_time > self.cache_duration:
                return None
            
            return cached_data
        except Exception:
            return None
    
    def set(self, strategy: str, results: dict, metadata: dict = None):
        cache_key = self._generate_cache_key(strategy)
        cache_path = self._get_cache_path(cache_key)
        
        cache_data = {
            'timestamp': datetime.now().isoformat(),
            'strategy': strategy,
            'results': results,
            'metadata': metadata or {}
        }
        
        try:
            with open(cache_path, 'w') as f:
                json.dump(cache_data, f, indent=2)
        except Exception as e:
            st.error(f"Cache write error: {e}")
    
    def clear(self):
        for cache_file in self.cache_dir.glob("*.json"):
            cache_file.unlink()
    
    def list_cached(self) -> List[dict]:
        cached_items = []
        for cache_file in self.cache_dir.glob("*.json"):
            try:
                with open(cache_file, 'r') as f:
                    data = json.load(f)
                    cached_items.append({
                        'strategy': data['strategy'],
                        'timestamp': data['timestamp'],
                        'file': cache_file.name,
                        'metadata': data.get('metadata', {})
                    })
            except Exception:
                continue
        return cached_items


# =============================================================================
# TOOL CALL LIMITER
# =============================================================================

class ToolCallLimiter:
    def __init__(self, max_calls: int = 20):
        self.max_calls = max_calls
        self.call_count = 0
        self.call_history = []
    
    def before_tool_call(self, tool_name: str):
        if self.call_count >= self.max_calls:
            raise Exception(f"Tool call limit reached ({self.max_calls} calls)")
        
        self.call_count += 1
        self.call_history.append({
            'tool': tool_name,
            'timestamp': datetime.now().isoformat(),
            'call_number': self.call_count
        })
    
    def get_stats(self) -> dict:
        tool_counts = {}
        for call in self.call_history:
            tool = call['tool']
            tool_counts[tool] = tool_counts.get(tool, 0) + 1
        
        return {
            'total_calls': self.call_count,
            'max_calls': self.max_calls,
            'remaining_calls': self.max_calls - self.call_count,
            'tool_breakdown': tool_counts
        }
    
    def reset(self):
        self.call_count = 0
        self.call_history = []


# =============================================================================
# SESSION STATE INITIALIZATION
# =============================================================================

if 'cache_manager' not in st.session_state:
    st.session_state.cache_manager = CacheManager()

if 'analysis_results' not in st.session_state:
    st.session_state.analysis_results = None

if 'current_strategy' not in st.session_state:
    st.session_state.current_strategy = None

if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

if 'analysis_runner' not in st.session_state:
    st.session_state.analysis_runner = None

if 'qa_runner' not in st.session_state:
    st.session_state.qa_runner = None

if 'user_id' not in st.session_state:
    st.session_state.user_id = "user_1"

if 'session_id' not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

if 'created_sessions' not in st.session_state:
    st.session_state.created_sessions = set()

if 'tools' not in st.session_state:
    st.session_state.tools = None


# =============================================================================
# AGENT CREATION (Using Direct API Calls)
# =============================================================================

def create_analysis_agent_with_tools(strategy: str, tools_dict: dict, tool_limiter: ToolCallLimiter):
    """Create analysis agent that uses direct API tools instead of MCP"""
    
    # Create tool call functions
    tool_functions = []
    
    # Polygon stock price tool
    if 'polygon' in tools_dict:
        def search_tickers(search_term: str, limit: int = 10) -> str:
            """Search for valid ticker symbols by company name or partial ticker. Use this first if unsure about ticker symbols."""
            tool_limiter.before_tool_call("polygon_search")
            result = tools_dict['polygon'].search_tickers(search_term, limit)
            return json.dumps(result)
        tool_functions.append(search_tickers)
        
        def get_stock_price(ticker: str) -> str:
            """Get the current stock price for a ticker symbol"""
            tool_limiter.before_tool_call("polygon_price")
            result = tools_dict['polygon'].get_stock_price(ticker)
            return json.dumps(result)
        tool_functions.append(get_stock_price)
        
        def get_company_info(ticker: str) -> str:
            """Get company information for a ticker symbol"""
            tool_limiter.before_tool_call("polygon_info")
            result = tools_dict['polygon'].get_company_info(ticker)
            return json.dumps(result)
        tool_functions.append(get_company_info)
    
    # Brave search tool
    if 'brave' in tools_dict:
        def search_web(query: str, count: int = 5) -> str:
            """Search the web for information"""
            tool_limiter.before_tool_call("brave_search")
            result = tools_dict['brave'].search(query, count)
            return json.dumps(result)
        tool_functions.append(search_web)
    
    # Web fetch tool
    if 'fetch' in tools_dict:
        def fetch_url(url: str) -> str:
            """Fetch content from a URL"""
            tool_limiter.before_tool_call("web_fetch")
            result = tools_dict['fetch'].fetch(url)
            return json.dumps(result)
        tool_functions.append(fetch_url)
    
    # # Filesystem tool
    # if 'filesystem' in tools_dict:
    #     def write_file(filename: str, content: str) -> str:
    #         """Write content to a file in the analysis outputs directory"""
    #         tool_limiter.before_tool_call("file_write")
    #         result = tools_dict['filesystem'].write_file(filename, content)
    #         return json.dumps(result)
    #     tool_functions.append(write_file)
    
    instructions = f"""
You are a trader that manages a portfolio of shares.
You have access to the following tools:
- search_tickers(search_term, limit): Search for valid ticker symbols by company name (USE THIS FIRST!)
- get_stock_price(ticker): Get current stock price (returns success: true/false and error if invalid)
- get_company_info(ticker): Get company details (optional - use sparingly due to rate limits)
- search_web(query, count): Search the web for information
- fetch_url(url): Fetch content from a specific URL

⚠️ CRITICAL RATE LIMIT CONSTRAINTS:
The Polygon API has a strict limit of 5 calls per minute on the free tier. Results are cached for 5 minutes.

REQUIRED WORKFLOW TO AVOID RATE LIMITS:
1. Use search_tickers() ONCE to find 5 valid tickers matching your strategy
2. From those results, select the 5 most promising based on company name/type
3. Call get_stock_price() ONLY on those 5 tickers (that's 5 API calls - at the limit!)
4. Build recommendations from those 5 stocks
5. DO NOT call get_company_info() unless absolutely necessary (adds extra API calls)
6. Use search_web() for news relating to the industry that the company is in. Ex, Apple would be in the technology industry.
7. Use search_web() for additional research if needed (different API, no limit impact)

ERROR HANDLING:
- If any tool returns "rate_limited": true, STOP making API calls immediately
- Work with whatever data you've already collected
- If you get rate limited early, make recommendations based on partial data
- Always check "success" field before using data
- Skip stocks that return errors

IMPORTANT LIMITS:
- You have {tool_limiter.max_calls} total tool calls across ALL tools
- Polygon API: Maximum 5 calls per minute
- Strategy: 1 search_tickers + 5 get_stock_price = perfect fit!

Your investment strategy for your portfolio is:
{strategy}

Please create a list of 5 stock recommendations with:
- stock_ticker: The ticker symbol (must have successful price lookup)
- stock_price: Current stock price (from get_stock_price)
- reasoning: Your detailed reasoning
- industy_news: Any recent news on the industry inputed in the strategy. "None" if no industry is inputed.
- buy_stock: True or False based on your decision

WORKFLOW EXAMPLE:
1. search_tickers("technology companies", 10) → Get 10 tech tickers
2. Select best 5 from results (e.g., MSFT, AAPL, GOOGL, NVDA, AMD)
3. get_stock_price("MSFT") → $420.50
4. get_stock_price("AAPL") → $185.25
5. get_stock_price("GOOGL") → $142.30
6. get_stock_price("NVDA") → $875.60
7. get_stock_price("AMD") → $165.80
8. Build 5 recommendations from these stocks
9. write_file("analysis_report.md", content)

DO NOT exceed 5 Polygon API calls or you will hit rate limits!
"""
    
    root_agent = Agent(
        model=LiteLlm(model='openai/gpt-4o-mini'),
        name='stock_analysis_agent',
        description='Stock analysis agent that researches and recommends stocks',
        instruction=instructions,
        tools=tool_functions,
        output_schema=StockList,
    )
    
    return root_agent


def create_qa_agent(strategy: str, results: dict):
    """Create Q&A agent"""


    formatted_results = format_results_for_context(results)
    
    qa_instructions = f"""
You are a financial analyst assistant helping to explain and discuss stock recommendations.

The investment strategy used was:
{strategy}

The analysis results are:
{formatted_results}

Answer questions about these recommendations, the strategy, and provide additional insights.
You can reference specific stocks, explain the reasoning, or discuss market conditions.
Be helpful and informative, but remind users that this is not financial advice.
"""
    
    qa_agent = Agent(
        model=LiteLlm(model='openai/gpt-4o-mini'),
        name='qa_agent',
        description='Q&A agent for discussing stock analysis',
        instruction=qa_instructions
    )
    
    return qa_agent


def format_results_for_context(results: dict) -> str:
    """Format analysis results for context"""
    if not results:
        return "No analysis results available yet."
    
    formatted = "Stock Recommendations:\n\n"
    for i, rec in enumerate(results.get('recommendations', []), 1):
        formatted += f"{i}. {rec['stock_ticker']} - ${rec['stock_price']}\n"
        formatted += f"   Decision: {'BUY' if rec['buy_stock'] else 'PASS'}\n"
        formatted += f"   Reasoning: {rec['reasoning']}\n\n"
    
    return formatted


def display_stock_card(rec: dict, index: int):
    """Display a stock recommendation card"""
    is_buy = rec['buy_stock']
    card_class = "buy-stock" if is_buy else "pass-stock"
    
    st.markdown(f"""
    <div class="stock-card {card_class}">
        <h3>{index}. {rec['stock_ticker']} - ${rec['stock_price']:.2f}</h3>
        <p><strong>{'✅ BUY' if is_buy else '❌ PASS'}</strong></p>
        <p><strong>Reasoning:</strong> {rec['reasoning']}</p>
    </div>
    """, unsafe_allow_html=True)


@st.cache_resource
def get_session_service():
    """Initialize session service once and cache it"""
    return InMemorySessionService()


async def call_agent_async(query: str, runner: Runner, user_id: str, session_id: str, session_service: InMemorySessionService, app_name: str):
    """Sends a query to the agent and returns the final response."""
    from google.genai import types
    
    # Ensure session exists before calling runner
    session_key = f"{user_id}:{session_id}"
    if session_key not in st.session_state.created_sessions:
        try:
            await session_service.create_session(app_name=app_name, user_id=user_id, session_id=session_id)
            st.session_state.created_sessions.add(session_key)
            logger.info(f"Created session: {session_key}")
        except Exception as e:
            try:
                await session_service.get_session(app_name=app_name, user_id=user_id, session_id=session_id)
                st.session_state.created_sessions.add(session_key)
                logger.info(f"Retrieved existing session: {session_key}")
            except Exception as e2:
                raise Exception(f"Failed to create or get session: {e}, {e2}")
    
    # Prepare the user's message
    content = types.Content(role='user', parts=[types.Part(text=query)])
    
    final_response_text = "Agent did not produce a final response."
    
    try:
        # Iterate through events
        async for event in runner.run_async(user_id=user_id, session_id=session_id, new_message=content):
            if event.is_final_response():
                if event.content and event.content.parts:
                    final_response_text = event.content.parts[0].text
                elif event.actions and event.actions.escalate:
                    final_response_text = f"Agent escalated: {event.error_message or 'No specific message.'}"
                break
    except Exception as e:
        logger.error(f"Error during agent execution: {e}")
        raise
    
    return final_response_text


# =============================================================================
# MAIN APP
# =============================================================================

def main():
    APP_NAME = "stock_analysis"
    
    session_service = get_session_service()
    
    st.markdown('<p class="main-header">📈 AI Stock Analysis Agent (Direct API)</p>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Settings")
        
        # Initialize tools
        if st.session_state.tools is None:
            with st.spinner("Initializing API tools..."):
                st.session_state.tools = create_direct_tools()
        
        # Session info
        st.subheader("🔐 Session Info")
        st.text(f"User: {st.session_state.user_id}")
        st.text(f"Session: {st.session_state.session_id[:8]}...")
        
        # Cache management
        st.subheader("Cache Management")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🗑️ Clear Cache", use_container_width=True):
                st.session_state.cache_manager.clear()
                st.success("Cache cleared!")
                st.rerun()
        
        with col2:
            if st.button("📚 View Cache", use_container_width=True):
                cached = st.session_state.cache_manager.list_cached()
                if cached:
                    st.write(f"**{len(cached)} cached analyses**")
                    for item in cached:
                        ts = datetime.fromisoformat(item['timestamp'])
                        st.text(f"• {ts.strftime('%m/%d %H:%M')}")
                else:
                    st.info("No cached analyses")
        
        # Tool settings
        st.subheader("Tool Settings")
        max_tool_calls = st.slider("Max Tool Calls", 20, 50, 20)
        use_cache = st.checkbox("Use Cache", value=True)
        force_refresh = st.checkbox("Force Refresh", value=False)
        
        # API Status
        st.subheader("🔑 API Status")
        polygon_key = os.getenv("POLYGON_API_KEY")
        brave_key = os.getenv("BRAVE_API_KEY")
        openai_key = os.getenv("OPENAI_API_KEY")
        
        st.write(f"Polygon: {'✅' if polygon_key else '❌'}")
        st.write(f"Brave: {'✅' if brave_key else '❌'}")
        st.write(f"OpenAI: {'✅' if openai_key else '❌'}")
        
        # Rate limit info
        if polygon_key:
            st.info("⚠️ Polygon Free Tier: 5 API calls/minute")
            if st.session_state.tools and 'polygon' in st.session_state.tools:
                polygon_tool = st.session_state.tools['polygon']
                if hasattr(polygon_tool, 'rate_limiter'):
                    recent_calls = len(polygon_tool.rate_limiter.calls)
                    cache_size = len(polygon_tool.rate_limiter.cache)
                    st.caption(f"📊 Recent calls: {recent_calls}/5")
                    st.caption(f"💾 Cached results: {cache_size}")
        
        # Reset button
        if st.button("🔄 Reset All", use_container_width=True):
            st.session_state.analysis_runner = None
            st.session_state.qa_runner = None
            st.session_state.session_id = str(uuid.uuid4())
            st.session_state.created_sessions = set()
            st.session_state.tools = None
            st.success("Reset complete!")
            st.rerun()
    
    # Main tabs
    tab1, tab2 = st.tabs(["🔍 Analysis", "💬 Q&A"])
    
    # Tab 1: Analysis
    with tab1:
        st.header("Stock Analysis")
        
        # Rate limit warning
        polygon_key = os.getenv("POLYGON_API_KEY")
        if polygon_key:
            st.info("ℹ️ **Polygon Free Tier:** Limited to 5 API calls/minute. The app will automatically wait if needed and cache results for 5 minutes.")
        
        if "strategy_input_widget" not in st.session_state:
            st.session_state.strategy_input_widget = ""
        
        strategy = st.text_area(
            "Enter your investment strategy:",
            placeholder="Focus on tech stocks with P/E ratios under 15 and strong revenue growth...",
            height=100,
            key="strategy_input_widget"
        )
        
        def add_text_to_area(new_text):
            st.session_state.strategy_input_widget += f"{new_text}\n"
        
        with st.expander("📝 Example Strategies"):
            st.button("Undervalued Tech", on_click=add_text_to_area,
                     args=["Focus on undervalued tech stocks with strong cash flow and market cap under $10B"])
            st.button("Dividend Growth", on_click=add_text_to_area,
                     args=["Find stocks with consistent dividend growth over 5 years and yield above 3%"])
            st.button("Small Cap Growth", on_click=add_text_to_area,
                     args=["Identify small-cap growth stocks in emerging sectors with revenue growth over 30%"])
        
        if st.button("🚀 Run Analysis", type="primary", use_container_width=True):
            if not strategy:
                st.error("Please enter an investment strategy!")
            else:
                # Check cache
                if use_cache and not force_refresh:
                    cached = st.session_state.cache_manager.get(strategy)
                    if cached:
                        st.success("✨ Using cached results!")
                        st.session_state.analysis_results = cached['results']
                        st.session_state.current_strategy = strategy
                        
                        qa_agent = create_qa_agent(strategy, cached['results'])
                        st.session_state.qa_runner = Runner(
                            agent=qa_agent,
                            app_name=APP_NAME,
                            session_service=session_service
                        )
                        
                        if 'metadata' in cached:
                            st.info(f"📊 Tool calls used: {cached['metadata'].get('tool_calls', 'unknown')}")
                        st.rerun()
                
                # Run fresh analysis
                with st.spinner("🔍 Analyzing stocks... (estimated time: 1-3 minutes)"):
                    try:
                        progress_bar = st.progress(0)
                        status_text = st.empty()
                        
                        status_text.text("Setting up tools...")
                        progress_bar.progress(20)
                        
                        tool_limiter = ToolCallLimiter(max_tool_calls)
                        
                        status_text.text("Creating agent...")
                        progress_bar.progress(40)
                        
                        root_agent = create_analysis_agent_with_tools(
                            strategy,
                            st.session_state.tools,
                            tool_limiter
                        )
                        
                        if st.session_state.analysis_runner is None:
                            st.session_state.analysis_runner = Runner(
                                agent=root_agent,
                                app_name=APP_NAME,
                                session_service=session_service
                            )
                        else:
                            st.session_state.analysis_runner._agent = root_agent
                        
                        runner = st.session_state.analysis_runner
                        
                        status_text.text("Running analysis...")
                        progress_bar.progress(60)
                        
                        result = asyncio.run(call_agent_async(
                            "Please analyze stocks based on the strategy and create recommendations.",
                            runner=runner,
                            user_id=st.session_state.user_id,
                            session_id=st.session_state.session_id,
                            session_service=session_service,
                            app_name=APP_NAME
                        ))
                        
                        progress_bar.progress(80)
                        status_text.text("Processing results...")
                        
                        if result and result != "Agent did not produce a final response.":
                            try:
                                output = json.loads(result)
                            except json.JSONDecodeError:
                                st.error(f"Invalid response format: {result}")
                                return
                            

                            qa_agent = create_qa_agent(strategy, output)
                            st.session_state.qa_runner = Runner(
                                agent=qa_agent,
                                app_name=APP_NAME,
                                session_service=session_service
                            )
                            
                            progress_bar.progress(100)
                            status_text.text("✅ Analysis complete!")
                            st.session_state.analysis_results = output
                            st.session_state.current_strategy = strategy
                            
                            
                            #RESULTS
                            st.header("Analysis Results")
        
                            if st.session_state.analysis_results is None:
                                st.info("👆 Please run an analysis first")
                            else:
                                st.subheader("📋 Strategy Used")
                                st.info(st.session_state.current_strategy)
                                
                                st.subheader("🎯 Stock Recommendations")
                                
                                recommendations = st.session_state.analysis_results.get('recommendations', [])
                                
                                if recommendations:
                                    buy_count = sum(1 for rec in recommendations if rec.get('buy_stock', False))
                                    pass_count = len(recommendations) - buy_count
                                    
                                    col1, col2, col3 = st.columns(3)
                                    with col1:
                                        st.metric("Total Analyzed", len(recommendations))
                                    with col2:
                                        st.metric("Buy Recommendations", buy_count)
                                    with col3:
                                        st.metric("Pass Recommendations", pass_count)
                                    
                                    st.divider()
                                    
                                    for i, rec in enumerate(recommendations, 1):
                                        display_stock_card(rec, i)
                                    
                                    st.subheader("📥 Export Results")
                                    col1, col2 = st.columns(2)
                                    
                                    with col1:
                                        json_data = json.dumps(st.session_state.analysis_results, indent=2)
                                        st.download_button(
                                            label="Download JSON",
                                            data=json_data,
                                            file_name=f"stock_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                                            mime="application/json"
                                        )
                                    
                                    with col2:
                                        df = pd.DataFrame(recommendations)
                                        csv = df.to_csv(index=False)
                                        st.download_button(
                                            label="Download CSV",
                                            data=csv,
                                            file_name=f"stock_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                                            mime="text/csv"
                                        )

                            tool_stats = tool_limiter.get_stats()
                            
                            if use_cache:
                                st.session_state.cache_manager.set(
                                    strategy,
                                    output,
                                    metadata={
                                        'tool_calls': tool_stats['total_calls'],
                                        'tool_breakdown': tool_stats['tool_breakdown']
                                    }
                                )
                            
                            st.success(f"Analysis complete! Used {tool_stats['total_calls']}/{tool_stats['max_calls']} tool calls")
                            
                            with st.expander("📊 Tool Usage Breakdown"):
                                for tool, count in tool_stats['tool_breakdown'].items():
                                    st.write(f"**{tool}**: {count} calls")
                                else:
                                    st.warning("No recommendations found.")
                            # st.rerun()
                        else:
                            st.warning("No response from agent. Please try again.")
                        
                    except Exception as e:
                        st.error(f"❌ Error: {str(e)}")
                        import traceback
                        with st.expander("Show traceback"):
                            st.code(traceback.format_exc())
    
    # Tab 2: Q&A
    with tab2:
        st.header("Ask Questions About the Analysis")
        
        if st.session_state.analysis_results is None:
            st.info("👆 Please run an analysis first")
        else:
            for message in st.session_state.chat_history:
                with st.chat_message(message["role"]):
                    st.write(message["content"])
            
            if prompt := st.chat_input("Ask a question..."):
                st.session_state.chat_history.append({"role": "user", "content": prompt})
                
                with st.chat_message("user"):
                    st.write(prompt)
                
                with st.chat_message("assistant"):
                    with st.spinner("Thinking..."):
                        try:
                            if st.session_state.qa_runner is None:
                                qa_agent = create_qa_agent(
                                    st.session_state.current_strategy,
                                    st.session_state.analysis_results,
                                    st.session_state.tools,
                                    tool_limiter
                                )
                                st.session_state.qa_runner = Runner(
                                    agent=qa_agent,
                                    app_name=APP_NAME,
                                    session_service=session_service
                                )
                            
                            answer = asyncio.run(call_agent_async(
                                prompt,
                                runner=st.session_state.qa_runner,
                                user_id=st.session_state.user_id,
                                session_id=st.session_state.session_id,
                                session_service=session_service,
                                app_name=APP_NAME
                            ))
                            
                            st.write(answer)
                            st.session_state.chat_history.append({"role": "assistant", "content": answer})
                            
                        except Exception as e:
                            st.error(f"Error: {str(e)}")
            
            if st.button("🗑️ Clear Chat"):
                st.session_state.chat_history = []
                st.session_state.session_id = str(uuid.uuid4())
                st.session_state.created_sessions = set()
                st.rerun()
    


if __name__ == "__main__":
    main()