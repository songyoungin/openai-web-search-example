import os
import json
from duckduckgo_search import DDGS
from dotenv import load_dotenv
from openai import OpenAI


load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
OPENAI_MODEL_NAME = os.getenv("OPENAI_MODEL_NAME", "gpt-4o")


def search_duckduckgo(
    query: str, max_results: int = 5, region: str = "kr-kr", sites: list[str] | str | None = None
) -> list[dict]:
    """DuckDuckGo API를 이용한 웹 검색을 수행합니다.

    Args:
        query (str): 검색할 키워드
        max_results (int, optional): 가져올 최대 검색 결과 수. 기본 값은 5개.
        region (str, optional): 검색 지역 코드. 기본값은 "kr-kr" (한국)
        sites (list[str] | str | None, optional): 검색할 특정 사이트 도메인. 기본값은 None.
                                                  단일 사이트: "naver.com"
                                                  여러 사이트: ["naver.com", "daum.net"]

    Returns:
        list[dict]: 검색 결과 리스트
    """
    # 특정 사이트 지정 시 site: 연산자 추가
    if sites:
        if isinstance(sites, str):
            # 단일 사이트
            query = f"{query} site:{sites}"
        elif isinstance(sites, list):
            # 여러 사이트를 OR로 연결
            site_conditions = " OR ".join([f"site:{site}" for site in sites])
            query = f"{query} ({site_conditions})"

    with DDGS() as ddgs:
        results = list(ddgs.text(query, max_results=max_results, region=region))

    return results


def get_chat_completion_result_with_web_search(
    query: str, max_results: int = 5, region: str = "kr-kr", sites: list[str] | str | None = None
) -> str:
    """웹 검색 결과를 참고해 정확한 정보를 제공하는 OpenAI 예제 함수.

    Args:
        query (str): 검색할 키워드
        max_results (int, optional): 가져올 최대 검색 결과 수. 기본 값은 5개.
        region (str, optional): 검색 지역 코드. 기본값은 "kr-kr" (한국)
        sites (list[str] | str | None, optional): 검색할 특정 사이트 도메인. 기본값은 None.
                                                  단일 사이트: "naver.com"
                                                  여러 사이트: ["naver.com", "daum.net"]

    Returns:
        str: 웹 검색 결과를 참고해 정확한 정보를 제공하는 OpenAI 예제 함수의 결과.
    """

    # 웹 검색 결과를 얻기 위한 OpenAI 도구 정의.
    tools = [
        {
            "type": "function",
            "name": "search_duckduckgo",
            "description": "Use the Duckduckgo API to fetch web search results. Call this when you need to search the web for information.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Keywords to search the web for",
                    },
                    "max_results": {
                        "type": "integer",
                        "description": "Maximum number of search results to fetch",
                        "default": max_results,
                    },
                },
                "required": ["query"],
                "additionalProperties": False,
            },
        }
    ]

    messages = [
        {
            "role": "system",
            "content": "당신은 웹 검색 결과를 참고해 정확한 정보를 제공하는 어시스턴트입니다. 반드시 답변 마지막에는 참고한 웹 검색 결과의 링크를 제공해야 합니다.",
        },
        {"role": "user", "content": query},
    ]

    tool_call_response = client.responses.create(
        model=OPENAI_MODEL_NAME,
        input=messages,
        tools=tools,
        tool_choice={
            "type": "function",
            "name": "search_duckduckgo",
        },  # 웹 검색을 강제하거나, 알아서 판단해서 검색하게 할 수 있음. 지금의 예제 코드는 강제한 버전.
    )

    tool_call = tool_call_response.output[0]
    tool_call_args = json.loads(tool_call.arguments)

    tool_response = search_duckduckgo(**tool_call_args, region=region, sites=sites)

    messages.append(tool_call)  # LLM의 웹 검색 판단 결과를 입력 메시지에 추가.
    messages.append(
        {
            "type": "function_call_output",
            "call_id": tool_call.call_id,
            "output": str(tool_response),
        }
    )  # 웹 검색 결과를 입력 메시지에 추가.

    generate_response = client.responses.create(
        model=OPENAI_MODEL_NAME,
        input=messages,
        tools=tools,
    )

    print(f"생성된 응답: {generate_response.output_text}")

    return generate_response.output_text


if __name__ == "__main__":
    REGION = "kr-kr"
    SITES = ["https://www.amc.seoul.kr/asan/healthinfo"]  # 아산병원
    get_chat_completion_result_with_web_search("당화혈색소 정상 수치 ", max_results=10, region=REGION, sites=SITES)
