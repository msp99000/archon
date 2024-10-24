from fastapi import FastAPI, BackgroundTasks, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager
import asyncio
from typing import Optional
import os
import re
import time
import logging
from langchain_community.document_loaders import TextLoader
from langchain.chains import load_summarize_chain
from langchain_openai import AzureChatOpenAI
from langchain.prompts import ChatPromptTemplate
import math
from dotenv import load_dotenv

load_dotenv()  

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
MAX_TOKENS_PER_SUMMARY = 3000 

app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Dictionary to store job statuses
job_statuses = {}

class JobStatus:
    def __init__(self):
        self.status = "processing"
        self.blog_post = None
        self.error = None
        self.progress = "Initializing"

class SearchRequest(BaseModel):
    keyword: str

class SearchResponse(BaseModel):
    job_id: str
    status: str
    message: str

class JobStatusResponse(BaseModel):
    status: str
    progress: str
    blog_post: Optional[str] = None
    error: Optional[str] = None

# Helper Functions
def setup_driver():
    chrome_options = Options()
    chrome_options.add_argument("--headless")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    service = Service(ChromeDriverManager().install())
    return webdriver.Chrome(service=service, options=chrome_options)

def sanitize_filename(filename):
    return re.sub(r'[\\/*?:"<>|]', "", filename)

def save_to_file(filename, content):
    try:
        if not os.path.exists('files'):
            os.makedirs('files')
        filepath = os.path.join('files', sanitize_filename(filename))
        logger.info(f"Saving content to {filepath}")
        with open(filepath, 'w', encoding='utf-8') as file:
            file.write(content)
        
        # Verify if file was saved correctly
        if os.path.exists(filepath) and os.path.getsize(filepath) > 0:
            logger.info(f"Content saved successfully to {filepath} (size: {os.path.getsize(filepath)} bytes)")
        else:
            logger.error(f"File saved but is empty or inaccessible: {filepath}")
        return True
    except Exception as e:
        logger.error(f"Error saving file {filename}: {e}")
        return False

# Scrape Functionality
async def scrape_page_async(driver, result_element, index, retries=3):
    try:
        await asyncio.sleep(2)  # Simulating asynchronous scraping process
        return scrape_result(driver, result_element, index, retries)
    except Exception as e:
        logger.error(f"Error in async scrape for result {index}: {e}")
        return False, str(e)

def scrape_result(driver, result_element, index, retries=3):
    try:
        result_elements = driver.find_elements(By.CSS_SELECTOR, "div.g")
        title_element = result_elements[index].find_element(By.CSS_SELECTOR, "h3")
        url = result_elements[index].find_element(By.CSS_SELECTOR, "a").get_attribute('href')

        current_url_before_click = driver.current_url
        
        driver.execute_script("arguments[0].scrollIntoView(true);", title_element)
        time.sleep(1)
        driver.execute_script("arguments[0].click();", title_element)

        WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.TAG_NAME, "body"))
        )
        time.sleep(3)

        current_url_after_click = driver.current_url

        if current_url_after_click == current_url_before_click:
            return False, f"Skipped result {index + 1}: Duplicate URL."

        page_title = driver.title
        paragraphs = driver.find_elements(By.TAG_NAME, "p")
        content = '\n\n'.join([p.text for p in paragraphs if p.text.strip()])

        filename = f"page_{index+1}_content.txt"
        if save_to_file(filename, f"Title: {page_title}\n\nURL: {current_url_after_click}\n\nContent:\n{content}"):
            return True, f"Page {index + 1} scraped successfully: {page_title}"
        else:
            return False, f"Failed to save content for page {index + 1}"

    except Exception as e:
        if retries > 0:
            logger.warning(f"Retrying scrape for result {index + 1}. Error: {e}")
            return scrape_result(driver, result_element, index, retries - 1)
        else:
            logger.error(f"Error scraping result {index + 1}: {e}")
            return False, f"Error scraping result {index + 1}"
    finally:
        try:
            driver.back()
            WebDriverWait(driver, 10).until(
                EC.presence_of_element_located((By.ID, "search"))
            )
            time.sleep(2)
        except Exception as e:
            logger.error(f"Error navigating back: {e}")

def read_files_in_directory(directory_path):
    """Read all files in a directory and return their paths."""
    file_paths = []
    try:
        for root, dirs, files in os.walk(directory_path):
            for file_name in files:
                file_path = os.path.join(root, file_name)
                file_paths.append(file_path)
    except (FileNotFoundError, PermissionError) as e:
        print(f"Error accessing '{directory_path}': {e}")
    return file_paths

def load_text_documents(file_paths):
    """Load text documents from file paths with detailed logging."""
    all_docs = []
    for pathfile in file_paths:
        try:
            # Check if the file exists and is readable
            if not os.path.exists(pathfile):
                logger.error(f"File does not exist: {pathfile}")
                continue
            if os.path.getsize(pathfile) == 0:
                logger.error(f"File is empty: {pathfile}")
                continue
            
            # Attempt to load the file with TextLoader
            logger.info(f"Loading file: {pathfile} (size: {os.path.getsize(pathfile)} bytes)")
            loader = TextLoader(file_path=pathfile, encoding='utf-8')
            docs = loader.load()
            all_docs.extend(docs)
        except (ValueError, Exception) as e:
            logger.error(f"Error processing file {pathfile}: {e}")
    
    logger.info(f"Loaded {len(all_docs)} documents in total.")
    return all_docs

def summarize_documents(documents):
    """Summarize the loaded documents using Azure OpenAI while minimizing token usage."""
    if not documents:
        print("No documents to summarize.")
        return None
    
    try:
        llm = AzureChatOpenAI(
            azure_deployment="gpt-4o",
            model="gpt-4o",
            api_version="2023-03-15-preview",
        )
        
        # Estimate tokens per document and calculate batch size
        avg_tokens_per_doc = 300
        batch_size = math.floor(MAX_TOKENS_PER_SUMMARY / avg_tokens_per_doc)
        
        all_summaries = []
        
        # Process documents in batches
        for i in range(0, len(documents), batch_size):
            document_batch = documents[i:i + batch_size]
            chain = load_summarize_chain(llm, chain_type="map_reduce")
            batch_summary = chain.run(document_batch)
            all_summaries.append(batch_summary)
        
        # Combine all summaries
        final_summary = "\n\n".join(all_summaries)
        return final_summary
    except Exception as e:
        print(f"Error during summarization: {e}")
        return None
    
def generate_blog_post(summary):
    """Generate a blog post from the summary using Azure OpenAI while minimizing token usage."""
    try:
        llm = AzureChatOpenAI(
            azure_deployment="gpt-4o",
            model="gpt-4o",
            api_version="2023-03-15-preview",
        )
        
        blog_prompt = ChatPromptTemplate.from_messages([
            ("system", """You're a blog writer for Upcore Technologies. Create an engaging post based on the given summary:

            1. Write a catchy, SEO-friendly title
            2. Include an introduction, 3-4 main points with subheadings, and a conclusion
            3. Naturally mention Upcore Technologies where relevant
            4. Use a professional yet conversational tone
            5. Aim for 800-1200 words
            6. Use markdown for formatting
            7. Fact-check and avoid mentioning other companies

            Deliver a well-structured, informative post that aligns with Upcore Technologies' brand."""),
            ("user", "{summary}")
        ])

        blog_chain = blog_prompt | llm
        
        # Generate the blog post with a token limit
        blog_post = blog_chain.invoke({"summary": summary})
        print(blog_post.content,"Blog COntent ")
        return blog_post.content
    except Exception as e:
        print(f"Error during blog generation: {e}")
        return None
    
def save_blog_post(filename, content):
    try:
        # Create the directory if it doesn't exist
        if not os.path.exists('blogs'):
            os.makedirs('blogs')
        
        filepath = os.path.join('blogs', sanitize_filename(filename))
        
        # Overwrite existing files (mode 'w')
        with open(filepath, 'w', encoding='utf-8') as file:
            file.write(content)
        
        # Verify that the file was saved correctly
        if os.path.exists(filepath) and os.path.getsize(filepath) > 0:
            logger.info(f"Blog post saved successfully to {filepath} (size: {os.path.getsize(filepath)} bytes)")
        else:
            logger.error(f"File saved but is empty or inaccessible: {filepath}")
        return True
    except Exception as e:
        logger.error(f"Error saving blog post {filename}: {e}")
        return False


async def process_keyword_async(keyword: str, job_id: str):
    job_status = job_statuses[job_id]
    driver = None
    try:
        # Scraping phase
        job_status.progress = "Setting up web scraper"
        driver = setup_driver()
        
        job_status.progress = "Performing Google search"
        driver.get("https://www.google.com")
        search_box = WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.NAME, "q"))
        )
        search_box.send_keys(keyword)
        search_box.send_keys(Keys.RETURN)

        job_status.progress = "Scraping search results"
        WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.ID, "search"))
        )

        result_elements = driver.find_elements(By.CSS_SELECTOR, "div.g")
        if len(result_elements) == 0:
            raise Exception("No search results found")

        # Concurrently scrape pages asynchronously
        tasks = [scrape_page_async(driver, result_element, index) for index, result_element in enumerate(result_elements[:3])]
        results = await asyncio.gather(*tasks)

        successful_scrapes = sum([1 for success, message in results if success])
        if successful_scrapes == 0:
            raise Exception("Failed to scrape any search results")

        # Summarization phase
        job_status.progress = "Loading scraped content"
        file_paths = read_files_in_directory("files")
        documents = load_text_documents(file_paths)
        
        job_status.progress = "Summarizing content"
        summary = summarize_documents(documents)
        if not summary:
            raise Exception("Failed to generate summary")
            # Blog generation phase
        job_status.progress = "Generating blog post"
        blog_post = generate_blog_post(summary)
        if not blog_post:
            raise Exception("Failed to generate blog post")

        # Save the blog post to a file
        blog_filename = f"{sanitize_filename(keyword)}.txt"
        if save_blog_post(blog_filename, blog_post):
            logger.info(f"Blog post saved to {blog_filename}")
        else:
            logger.error(f"Failed to save blog post to {blog_filename}")
            raise Exception("Failed to save blog post to file")
        
        job_status.status = "completed"
        job_status.blog_post = blog_post
        job_status.progress = "Blog post generated and saved successfully"

    except Exception as e:
        job_status.status = "failed"
        job_status.error = str(e)
        logger.error(f"Error processing keyword '{keyword}': {e}")
    finally:
        if driver:
            driver.quit()

@app.post("/generate-blog", response_model=SearchResponse)
async def generate_blog(request: SearchRequest, background_tasks: BackgroundTasks):
    job_id = f"{request.keyword}-{int(time.time())}"
    job_statuses[job_id] = JobStatus()
    
    background_tasks.add_task(process_keyword_async, request.keyword, job_id)
    
    return SearchResponse(
        job_id=job_id,
        status="processing",
        message=f"Blog generation started for keyword: {request.keyword}"
    )

@app.get("/blog-status/{job_id}", response_model=JobStatusResponse)
async def check_blog_status(job_id: str):
    if job_id not in job_statuses:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job_status = job_statuses[job_id]
    return JobStatusResponse(
        status=job_status.status,
        progress=job_status.progress,
        blog_post=job_status.blog_post,
        error=job_status.error
    )

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
