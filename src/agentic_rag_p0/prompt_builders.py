from __future__ import annotations

import json


def build_understand_question_prompt(question: str, corpus_metadata: dict[str, object]) -> str:
    return f"""
You are the question-understanding stage for an agentic RAG system.
Classify the user question for a system that has exactly these tools: search_docs, query_data, web_search.
Use the available corpus metadata to prefer local structured/doc tools whenever the answer appears available there.

Return JSON only with this shape:
{{
  "normalized_question": "string",
  "entities": ["string"],
  "timeframe": "string",
  "recent_intent": true,
  "mode": "single-tool|multi-tool|no_tool|refusal",
  "reason": "string"
}}

Corpus metadata:
{json.dumps(corpus_metadata, indent=2)}

Question: {question}
"""


def build_plan_prompt(
    normalized_question: str,
    understanding: dict,
    corpus_metadata: dict[str, object],
    tool_descriptions: dict[str, str],
) -> str:
    return f"""
You are the planning stage for an agentic RAG system with exactly three tools.

Tools:
- search_docs: {tool_descriptions["search_docs"]}
- query_data: {tool_descriptions["query_data"]}
- web_search: {tool_descriptions["web_search"]}

Planning rules:
- Keep subgoals minimal. Prefer 1 subgoal for single-fact questions.
- If structured metadata already shows the needed metric/entity/year columns, prefer query_data first.
- If document metadata shows strong topical alignment with the question and potential for direct evidence, prefer search_docs first.
- REMEMBER ITS BETTER DATA QUALITY OVER QUANTITY. DO NOT PLAN TO CALL A TOOL JUST TO CALL A TOOL.
- In case you are using search_docs if you are not satisfied with the quality of the retrieved evidence, let the system redirect to web_search instead of creating new subgoals for more search_docs calls to check if the data is enough or not be a midly strict analyzer.
- Use web_search only if the local structured/doc metadata does not appear sufficient or the question is explicitly recent/current.
- Do not create extra subgoals once one local tool can answer the question.
- If local docs are likely to contain only indirect or comparative commentary, keep the plan open for a later redirect(AS THE AGENT CAN REDIRECT IF U THINK THE DATA IS NOT ENOUGH TO WEB SEARCH) rather than assuming local evidence will be enough.
- in case the question has anything related to current or future data or explicitly mentions that it needs current or future data prefer web search this is important when the words current is given dont plan to searh locally but use the web plan to search the we 
.
eturn JSON only with this shape:
{{
  "plan_summary": "1-3 sentence plan",
  "answer_requirements": ["string"],
  "subgoals": [
    {{"description": "string", "status": "pending", "notes": ""}}
  ],
  "likely_tools": ["search_docs"],
  "risks": ["string"]
}}

Question understanding:
{json.dumps(understanding, indent=2)}

Corpus metadata:
{json.dumps(corpus_metadata, indent=2)}

Question:
{normalized_question}
"""


def build_choose_next_action_prompt(
    normalized_question: str,
    plan_summary: str,
    likely_tools: list[str],
    subgoals_payload: list[dict],
    evidence_payload: list[dict],
    corpus_metadata: dict[str, object],
    steps_used: int,
    max_tool_calls: int,
) -> str:
    return f"""
You are the action selector for a bounded agent loop.
Choose exactly one next action from: search_docs, query_data, web_search, answer, refuse.

Constraints:
- Follow the current plan and pending subgoals unless the evidence clearly justifies a different next step.
- Prefer answer when the current evidence already covers the subgoals.
- If at least one usable evidence item directly answers the question and there is no unresolved required comparison/explanation, choose answer.
- Use web_search only when recent or external information is needed.
- Refuse instead of guessing when the missing information is unavailable.
- The agent has a hard cap of {max_tool_calls} tool calls.
- Prefer local tools over web_search whenever corpus metadata suggests the answer is present locally.
- For commentary/reason questions, evidence is NOT sufficient just because it appears topically related.
- Distinguish between:
  - direct evidence: explicitly answers the asked explanation/reason
  - indirect evidence: adjacent-period comparison, generic strategy language, or contextual margin commentary
- If the current local evidence is only indirect, do NOT choose answer or refuse.
- When search_docs was part of the plan and the current local evidence is indirect or insufficient, choose web_search as the next step instead of stopping.
- Keep the remaining plan in mind on every loop; do not drop unresolved explanation/document subgoals just because a structured lookup already succeeded.
- when the question has current or future related things where it explicitly mentions it needs current data or future data use web search as the next step without any hesitation. also if the question is about something which is likely to have data in the docs but the data you got from the docs is not good enough then also go with web search without any hesitation. always prefer web_search .
Return JSON only with this shape:
{{
  "action": "search_docs|query_data|web_search|answer|refuse",
  "rationale": "string",
  "refusal_reason": "string"
}}

Question: {normalized_question}
Plan summary: {plan_summary}
Likely tools: {json.dumps(likely_tools)}
Subgoals: {json.dumps(subgoals_payload, indent=2)}
Recent evidence: {json.dumps(evidence_payload, indent=2)}
Corpus metadata: {json.dumps(corpus_metadata, indent=2)}
Steps used: {steps_used}
"""


def build_query_data_input_prompt(
    normalized_question: str,
    plan_summary: str,
    current_evidence: list[dict],
    schema: dict[str, object],
    metadata: dict[str, object],
) -> str:
    return f"""
Generate one read-only SQLite query for the user's question.
Rules:
- Output JSON only.
- Use only SELECT or WITH.
- Do not invent tables or columns.
- Keep the query concise and directly answer the current missing subgoal.
- Prefer exact entity/year filters when present in metadata.
- If the question, plan, or current evidence implies a comparison, premise check, increase/decrease check, trend check, or "did it rise/fall" check, the query must fetch every comparison point needed to prove that claim.
- Do not return a single-year or single-value query when the reasoning requires multiple years, multiple entities, or a before-vs-after comparison.
- If the agent needs to verify whether a value increased or decreased in FYX, fetch FYX and the immediately relevant comparison period from the structured data whenever available.
- If the question compares companies, return all requested companies in one query whenever possible.
- If the question asks for a table/comparison, return the columns needed for that comparison rather than a single scalar.
- Use WHERE filters that are explicit and minimal, and use IN (...) when multiple entities or years are needed.
- Prefer returning raw values needed for reasoning over a prematurely narrowed query.
- Before writing the SQL, check that the selected rows/columns are enough to support the conclusion the agent is trying to make.
- For any "did it increase/decrease" question, include both the target period and the relevant comparison period.
- For any multi-entity comparison, return all requested entities in one query whenever possible.
- For any multi-period trend question, return every period needed for the trend, not just one period.

Return JSON:
{{
  "tool_input": "SQL query string"
}}

Database schema:
{json.dumps(schema, indent=2)}

Database metadata and sample rows:
{json.dumps(metadata, indent=2)}

Question: {normalized_question}
Plan summary: {plan_summary}
Current evidence: {json.dumps(current_evidence, indent=2)}
"""


def build_tool_input_prompt(
    action: str,
    normalized_question: str,
    plan_summary: str,
    current_evidence: list[dict],
    tool_descriptions: dict[str, str],
    document_catalog: list[dict] | None = None,
) -> str:
    return f"""
Generate the input for the tool `{action}`.
Tool description: {tool_descriptions[action]}
Important : Only in case of web_search action use why where and other forms of question to make the query more specific and natural as if a person is asking this query to a search engine. also use adverbs and nouns to make the query more specific and natural.
IMportant: keep the questions generated for web_search dont include any words which does not have any relation to the question such as significant , notable eg- if you create the question for finding data abot profit margin of company xxx from year 2024 the question should be simple "Why did XXX Profit increase in @024 not something like why did it increse significantly or notably questions should be simple "
The Input must be specific for every tool and follow the rules:
- for web_search , use why or what or how or when or which in the query to make it more specific and natural as if a person is asking this query to a search engine. also use adverbs and nouns to make the query more specific and natural.
- For search_docs, return a concise query string optimized for retrieving relevant documents. Use keywords and filters based on the question, plan, and current evidence. Do not return a full sentence.
- For search_docs, you may also choose up to 2 document containers from the document catalog when the likely source documents are clear. Prefer selecting the most relevant document containers instead of searching every document.
- For web_search, return a concise query string optimized for retrieving relevant web results. Use keywords from the question, plan, and evidence gaps. Prefer a query a person would naturally type into a search engine.
- eg of web query in case of something based on tcs current stock price write the query as "whats the current stock price of tcs " also use the adverbs and nouns dont make the prompt bland like "TCS margin improvement FY2024 reasons" like this this does not provide any good info tha api we are using here is Tavily Api constrct a prompt fot it.
Return JSON only:
{{do
  "tool_input": "string",
  "document_filters": ["filename.pdf"]
}}

Question: {normalized_question}
Plan summary: {plan_summary}
Current evidence: {json.dumps(current_evidence, indent=2)}
Document catalog: {json.dumps(document_catalog or [], indent=2)}
"""


def build_sufficiency_prompt(
    normalized_question: str,
    action: str,
    result_summary: str,
    subgoals_before_update: list[dict],
    evidence_payload: list[dict],
    corpus_metadata: dict[str, object],
) -> str:
    structured_metadata = corpus_metadata.get("structured", []) if isinstance(corpus_metadata, dict) else []
    document_metadata = corpus_metadata.get("documents", {}) if isinstance(corpus_metadata, dict) else {}
    return f"""
You are the sufficiency checker for an agentic RAG/structure/web loop.
Decide whether the agent should answer, continue, or refuse.

Rules:

- Very IMportant: --- more important give specific importance Incase the question and the data from the search_docs are on opposite sides like the question like "why is population of india shrinking but the docs only have about data of indias incresing population rhis means the data in the docs is right and you dont have to go to web_search, always prioritze the data from the rag in case of conflict and return the conflicted data by saying eg it was not a shrinking population but increasing population with other data you may have collected.
- If a single structured lookup already returns the requested scalar/comparison, mark outcome=sufficient.
- Use the structured metadata to judge whether the unresolved information should come from query_data. If the needed entity/metric/year clearly exists in the structured database and the current evidence is missing only that structured value, prefer next_action=query_data.
- Use the document metadata to judge whether the unresolved information should come from search_docs. If the needed evidence is likely in local documents and the last action was not search_docs, prefer next_action=search_docs.
- If all subgoals are done or only cosmetic follow-up work remains, mark outcome=sufficient.
- Do not request more tool calls just to restate or validate an already grounded answer.
- If local doc evidence confirms the topic but does not explicitly answer the asked explanation, prefer outcome=continue so the system can handle any needed redirect.
- Only mark outcome=sufficient for commentary questions when the explanation is directly grounded in the current evidence.
- If the plan still contains unresolved document/explanation subgoals, do not mark outcome=sufficient just because a structured subgoal is done.
- When outcome=continue and the next best step is obvious, set next_action to one of query_data, or web_search so the loop can execute it directly.
- IMPORTANT: if the last action was search_docs and the retrieved local evidence is still not sufficient to answer the question, explicitly set next_action=web_search.
- IMPORTANT: after an insufficient search_docs result, prefer web_search as the next step rather than another search_docs call unless a materially different local retrieval path is clearly justified.
- Be sctict when choosing web_searh only choose it if u are 100 percen certain that you dont have the required data be srtict here but at the same time if u are sure that the data is not eeeeeenough surely go with web search 
- See if u get usable information from the retrieved documents if yes then go with answer . if usable then use dont go to web dont go to web
- If the retrieved local doc evidence is boilerplate, wrong-subject, indirect, repetitive, or still missing the asked explanation, choose outcome=continue with next_action=web_search.
- Treat web_search as the preferred next step after one weak local-doc pass on a commentary/reason question.
-sufficent data meas data from which you can infer a answer considering all the evidence you have and got once you can infer the data you can give ok sufficient to it.
Return JSON only:
{{
  "outcome": "sufficient|continue|refuse",
  "reason": "string",
  "next_action": "search_docs|query_data|web_search|",
  "subgoals": [
    {{"description": "string", "status": "pending|partial|done|blocked", "notes": "string"}}
  ]
}}

Question: {normalized_question}
Last action: {action}
Last result summary: {result_summary}
Subgoals before update: {json.dumps(subgoals_before_update, indent=2)}
Evidence so far: {json.dumps(evidence_payload, indent=2)}
Structured metadata: {json.dumps(structured_metadata, indent=2)}
Document metadata: {json.dumps(document_metadata, indent=2)}
"""


def build_sufficiency_with_evidence_review_prompt(
    normalized_question: str,
    action: str,
    result_summary: str,
    subgoals_before_update: list[dict],
    evidence_payload: list[dict],
    corpus_metadata: dict[str, object],
) -> str:
    base_prompt = build_sufficiency_prompt(
        normalized_question,
        action,
        result_summary,
        subgoals_before_update,
        evidence_payload,
        corpus_metadata,
    )
    return f"""{base_prompt}

Additional evidence review requirement:
- In the same JSON response, also review the evidence items shown above.
- Mark each pending or newly returned evidence item as usable only when it helps answer the question or a specific subgoal.
- Mark evidence unusable when it is irrelevant, wrong-company, wrong-period, boilerplate, generic, or only shares keywords without answering the needed point.
- Do not invent new evidence ids. Only update evidence ids present in "Evidence so far".
- Keep the original sufficiency fields and add "evidence_updates" to the same JSON object.

Required additional JSON field:
{{
  "evidence_updates": [
    {{
      "evidence_id": "string",
      "usable": true,
      "usability_flag": "short_reason",
      "related_subgoal": "string"
    }}
  ]
}}
"""


def build_direct_answer_prompt(normalized_question: str) -> str:
    return f"""
Answer the user's question directly without calling tools.
Keep the answer concise and honest.

Question: {normalized_question}
"""


def build_compose_answer_prompt(
    normalized_question: str,
    plan_summary: str,
    evidence_payload: list[dict],
) -> str:
    return f"""
You are the final answer composer for an agentic RAG system.
Use only the evidence below. Do not invent facts.
For explanation/commentary questions, prefer a single internally consistent conclusion:
- If the evidence directly contains the requested explanation or reason, answer with that.
- If the exact requested period or wording is not explicit but nearby grounded evidence provides the closest supported explanation, say that clearly and then provide that explanation.
- If the evidence contradicts the premise of the question, correct the premise using the evidence instead of forcing the original framing.
- If the exact expected source is missing but a strong and relevant alternative source supports a careful answer, infer the closest supported answer from that evidence instead of saying nothing can be concluded.
- When doing that, explicitly say that the support comes from a different source than the one implied or requested, and avoid overstating it as a direct company-stated explanation.
- Prefer phrasing like "the available evidence suggests", "the strongest cited support indicates", or "while this is not stated directly by the company, the cited source indicates".
- Example: if the question asks what reason a company itself gave for profit growth or margin improvement, but the available support comes from a credible third-party source rather than the company directly, answer with the closest supported explanation and explicitly note that it comes from a different source than the company itself.
- For structured results, summarize the key takeaway plainly and rely on the cited evidence ids.
- Do not say both "no information exists" and also list reasons in the same answer.
- Prefer source-agnostic phrasing such as "the available evidence", "the gathered evidence", or "the cited evidence".
- Do not say "the documents do not contain", "the answer was not found in the documents", or similar document-only wording unless both of these are true:
  1. the question explicitly asks about documents or a document, and
  2. every evidence item you rely on comes only from search_docs.
- If web_search or query_data evidence is present or used, never frame the answer as a document-only failure. Say instead that the available evidence does not directly show, confirm, or explain the requested point.
Return JSON only:
{{
  "answer": "string",
  "used_evidence_ids": ["string"]
}}

Question: {normalized_question}
Plan summary: {plan_summary}
Evidence:
{json.dumps(evidence_payload, indent=2)}
"""


def build_reformulate_tool_input_prompt(
    action: str,
    previous_input: str,
    error: Exception,
    normalized_question: str,
) -> str:
    return f"""
Repair or tighten the tool input after a tool failure.
Return JSON only:
{{
  "tool_input": "string"
}}

Tool: {action}
Previous input: {previous_input}
Error: {error}
Question: {normalized_question}
"""
