from fastapi import FastAPI, Request, HTTPException
import requests
from fastapi.responses import JSONResponse, HTMLResponse
from dotenv import load_dotenv
import os
import google.generativeai as genai
import pandas as pd
import json
import asyncio
import aiohttp
import logging # testing logging

app = FastAPI()

async def json_to_df(url: str):
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url) as response:
                logging.info(msg="Processing the request") # basic logging info
                response.raise_for_status() # if the error code for the req is not 200
                data = await response.json(content_type="text/plain")
                # converting the data into dataframe
                df = pd.DataFrame(data)
                print(type(df))
                return df
    except aiohttp.ClientConnectionError as e:
        logging.error(f"Connection error happened : {e}")
        raise HTTPException(status_code=503, detail="Connection to the url failed")
    except Exception as e:
        logging.error(f"Some other exception occurred : {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

# the main candidate details extraction engine
async def main_engine(df, domains):
    load_dotenv()
    google_key = os.getenv('GOOGLE_KEY')
    genai.configure(api_key=google_key)
    model = genai.GenerativeModel('gemini-2.5-flash')

    # prompt creation engine
    def create_prompt(candidate_data, domain_list):
        domains = " ".join(domain_list)
        cand_str = json.dumps(candidate_data.to_dict(), indent=2)
        return f"""
                You are an expert technical recruiter responsible for categorizing candidates into specialized domains.
                Your task is to analyze the candidate's application below. First, classify the candidate into the single best-fit domain from the provided list. Second, score their expertise within that chosen domain from 1 to 10.

                ## Available Domains:
                {domains}

                ## Candidate Application:
                {cand_str}

                ## Your Task:
                Return ONLY a valid JSON object with the following keys:
                - "name" : Name of the candidate
                - "domain": A string that must be one of the available domains.
                - "expertise_score": A number from 1 to 10, representing their skill in that domain.

                ## JSON Output:
            """

    results = []
    # main iteration
    for index, candidate in df.iterrows(): # this is the main cause of the latency bottleneck, sequential processing loop
        prompt = create_prompt(candidate, domains)
        try:
            llm_response = await model.generate_content_async(prompt)
            evaluation_json = llm_response.text.strip().replace("```json", "").replace("```", "")
            evaluation_data = json.loads(evaluation_json)

            results.append(evaluation_data)
        except Exception as e:
            print(f"Error on index {index} : {e}")
    scoring_df = pd.DataFrame(results)
    top_diverse_candidates = scoring_df.groupby('domain').apply(
                                    lambda x: x.nlargest(2, 'expertise_score')
                            ).reset_index(drop=True)
    
    return top_diverse_candidates


@app.post("/criterias", response_class=JSONResponse) # using this endpoint the recruiter submits the criterias for hiring
async def fetch_best_candidates(request: Request):
    response = await request.json()
    domains = response.get("domains")
    url = response.get("url")
    candidates = await json_to_df(url)
    candidates_selected = await main_engine(candidates, domains)
    candidates_list = candidates_selected.to_dict(orient='records')
    json_response_data = {
        'selected_candidates': candidates_list
    }

    return JSONResponse(content=json_response_data, status_code=200)
