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
- if the data changes with time like current rate stock prices go with web_search not local docs dynamic data with relation to time should be planned to use web search.
- REMEMBER ITS BETTER DATA QUALITY OVER QUANTITY. DO NOT PLAN TO CALL A TOOL JUST TO CALL A TOOL.
- In case you are using search_docs if you are not satisfied with the quality of the retrieved evidence, let the system redirect to web_search instead of creating new subgoals for more search_docs calls to check if the data is enough or not be a midly strict analyzer.
- Use web_search only if the local structured/doc metadata does not appear sufficient or the question is explicitly recent/current.
- Do not create extra subgoals once one local tool can answer the question.
- If local docs are likely to contain only indirect or comparative commentary, keep the plan open for a later redirect(AS THE AGENT CAN REDIRECT IF U THINK THE DATA IS NOT ENOUGH TO WEB SEARCH) rather than assuming local evidence will be enough.
- Important very Important : in case the question has anything related to current or future data or explicitly mentions that it needs current or future data prefer web search this is important when the words current is given dont plan to searh locally but use the web plan to search the we 
- Distinguish between:
  - valid domain questions answerable by web_search even if they are not in local docs/data, such as recent company, sector, stock-market, or business developments
  - unrelated general-knowledge, trivia, joke, riddle, or creative even prdicting stocks  requests that are outside the system's scope
- If the question is outside the supported business/company/financial scope of all available tools, do not make a normal answer plan.
- For such out-of-scope questions, return an out-of-scope plan:
  - `plan_summary` should clearly say the question is outside supported scope
  - `answer_requirements` should be []
  - `subgoals` should contain one blocked subgoal describing refusal
  - `likely_tools` should be []
  - `risks` should include `out_of_scope`
- Do not mark recent company, stock, sector, or market questions as out-of-scope just because the local corpus is insufficient; those should normally plan for web_search.
- Example out-of-scope question: "What is the airspeed velocity of an unladen swallow?"

Return JSON only with this shape:
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
- in case of dynamic data which can change witht time prefer web_search.
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
- Additional search_docs query quality rule: write retrieval queries as dense keywords, not instructions. Do not start with words like "Search", "Find", "Retrieve", or "Look for".
- Additional search_docs query quality rule: for margin/profit driver questions, include section and driver terms such as "management discussion analysis", "operating margin", "cost of sales", "employee cost", "subcontractor cost", "pricing", "utilization", "efficiency", "large deals", "FY2024", and the company name when relevant.
- Additional search_docs query quality rule: prefer queries like "TCS operating margin pricing models efficiency FY2024 management discussion" instead of "Search the TCS annual report to find factors influencing operating margin".
- Strict search_docs rule: never copy a subgoal, rationale, or task sentence verbatim into tool_input. Always rewrite it into 5-12 dense retrieval keywords.
- Strict search_docs rule: remove instruction verbs and filler words such as "identify", "extract", "factors influencing", "from the document", "annual report", "find information", "during that period", and keep only entity, metric, period, section, and driver terms.
- Strict search_docs rule: bad query = "Identify and extract factors influencing TCS operating margin in Fiscal Year 2024 from the tcs.pdf document"; good query = "TCS operating margin FY2024 margin performance utilization productivity realization currency hikes infrastructure".
- Strict search_docs rule: bad query = "Search Infosys annual report for factors contributing to operating margin"; good query = "Infosys operating margin FY2024 cost efforts employee cost subcontractor cost management discussion".
- For web_search, return a concise query string optimized for retrieving relevant web results. Use keywords from the question, plan, and evidence gaps. Prefer a query a person would naturally type into a search engine.
- For web_search, use a hybrid strategy for multi-entity questions: if the user's question is explicitly comparative and there is not yet any usable web evidence for the comparison, one mixed comparative query is allowed.
- For web_search, after one mixed comparative web query or when a specific company/entity/subgoal is still weak, generate the query for exactly ONE unresolved company/entity/subgoal only. Do not keep repeating mixed multi-company queries.
- For targeted web_search, keep the other unresolved companies/entities/subgoals for later calls through the subgoal state. Example targeted queries: "Why did Infosys operating margin change FY2024" or "Why did TCS operating margin change FY2024".
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


def build_weighted_search_docs_input_prompt(
    normalized_question: str,
    plan_summary: str,
    current_evidence: list[dict],
    tool_descriptions: dict[str, str],
    document_catalog: list[dict] | None = None,
) -> str:
    return f"""
Generate the input for the tool `search_docs`.
Tool description: {tool_descriptions["search_docs"]}

Return JSON only with this shape:
{{
  "tool_input": "short natural-language retrieval query",
  "document_filters": ["filename.pdf"],
  "weighted_terms": {{
    "must_have": ["string"],
    "should_have": ["string"],
    "context": ["string"],
    "route_only": ["string"]
  }}
}}

Weighted retrieval rules:
- Keep this generic. Do not assume any current PDF, company, domain, or document type.
- `tool_input` should be a concise readable query for the missing information.
- `document_filters` may include up to 2 filenames only when the document catalog clearly identifies the likely source.

Hard constraints for `must_have`:
- Every `must_have` term must be a searchable content phrase, not a user-intent phrase.
- Never include standalone generic intent words in `must_have`: reason, reasons, cause, causes, why, explanation, explain, details, information, impact, effect, analysis, summary, overview, factors, drivers, contributors.
- Never include a broad direction/change word by itself in `must_have`: increase, decrease, rise, fall, growth, decline, reduction, improvement, higher, lower, change.
- If a candidate `must_have` is mostly generic intent, move it to `should_have` or drop it.
- Before returning JSON, validate `must_have`: remove any term that would still make sense for almost any document or topic.

- `must_have` contains the searchable subject of the question: the specific metric/topic, event, action, claim, comparison object, or answer target.
- Prefer specific 2-4 word noun phrases in `must_have` when possible, rather than isolated generic words.
- Do not put standalone generic intent words in `must_have`, such as reason, reasons, cause, causes, why, explanation, details, information, impact, effect, analysis, summary, or overview.
- Put broad direction words such as increase, decrease, rise, fall, improvement, decline, higher, or lower in `should_have` unless they are paired with the specific metric/topic/event.
- `should_have` contains helpful synonyms, causal/explanation words, related concepts, and alternate wording.
- `context` contains years, dates, timeframes, document section hints, or other narrowing context.
- `route_only` contains entity names, document names, people, teams, companies, or places that should mainly choose the document/source and should not dominate semantic retrieval inside that source.
- For increase/decrease/change questions, include generic directional synonyms in `should_have`, such as increase/growth/rise/improvement or decrease/decline/reduction/lower.
- For why/reason/explanation questions, include generic causal words in `should_have`, such as reason, cause, because, due, driven, attributable, influenced, headwind, tailwind.
- If a generic intent or direction word is important, pair it with the searchable subject before placing it in `must_have`; otherwise keep it in `should_have`.
- Expand vague user terms into common, domain-neutral noun phrases only when the wording or catalog supports the expansion. Keep expansions conservative and retrieval-focused.
- Do not put filler words, instruction verbs, or whole subgoals in weighted terms.
- Avoid deterministic recommendations or conclusions. This prompt only builds retrieval input.

Generic examples:
- Question: "Why did response time increase in 2024?"
  Bad `must_have`: ["response time increase", "reasons"]
  Good `must_have`: ["response time"]
  Good `should_have`: ["increase", "delay", "latency", "reason", "cause", "due"]
  Good `context`: ["2024"]
- Question: "What caused customer complaints to fall in Q2?"
  Bad `must_have`: ["complaints fall", "cause"]
  Good `must_have`: ["customer complaints"]
  Good `should_have`: ["fall", "decline", "reduction", "cause", "because", "driver"]
  Good `context`: ["Q2"]
- Question: "Summarize the impact of policy changes on remote work."
  Bad `must_have`: ["summary", "impact"]
  Good `must_have`: ["policy changes", "remote work"]
  Good `should_have`: ["impact", "effect", "changed", "influenced"]
- Question: "Why did Team Alpha miss the delivery date?"
  Bad `must_have`: ["Team Alpha", "why", "miss"]
  Good `must_have`: ["delivery date"]
  Good `should_have`: ["missed", "delay", "reason", "cause", "blocker"]
  Good `route_only`: ["Team Alpha"]

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

Additional web fallback targeting requirement:
- For explicitly comparative questions, one mixed comparative web_search is acceptable when no usable web evidence exists yet for the comparison.
- After one mixed comparative web_search, or when one company/entity/subgoal remains weaker than the others, choose exactly ONE target for the next web_search.
- In "reason", state whether the next web_search should be mixed comparative or targeted to one selected company/entity/subgoal.
- Do not repeatedly ask web_search to cover multiple companies/entities once targeted follow-up is needed.

Additional strict sufficiency requirement for explanation/driver questions:
- Treat evidence as sufficient only when it directly supports the full answer the user asked for, not just one part of it.
- Check all required dimensions before marking sufficient: entity, metric/topic, comparison side, requested scope, claim type, source relevance, explanation quality, and completeness.
- Mark evidence as usable but not sufficient when it is merely adjacent, generic, background context, risk boilerplate, broad strategy text, a cost category without causal explanation, or a fact that needs a large inference to answer the question.
- If the answer would need careful wording like "may have", "could suggest", "not directly stated", or "inferred from", mark the related subgoal "partial" unless another independent evidence item supports the same claim.
- If any required entity, comparison side, driver, reason, or explanation is only partially supported, choose outcome="continue" and set next_action="web_search" unless a clearly different local document search is justified.
- For multi-part or multi-entity questions, do not mark the whole answer sufficient when one part is strong and another part is generic, adjacent, or inferred. Continue for the weaker part.
- In the reason, name the weak entity/subgoal and explain the evidence gap in generic terms such as "generic support", "indirect evidence", "missing direct driver", "weak causal link", "incomplete comparison", or "background-only evidence".
- Do not let a structured value alone complete an explanation/driver subgoal. Structured values can satisfy the numeric part, but the explanation still needs direct qualitative support.

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
- Output format is strict. Your response must be exactly one JSON object and nothing else.
- The first character of your response must be `{{` and the last character must be `}}`.
- Do not use markdown fences.
- Do not add any prose before or after the JSON.
- Do not add labels like "Answer:" or "JSON:".
- If you are about to write a normal paragraph, stop and convert it into the required JSON object instead.
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


def build_cap_final_answer_prompt(
    normalized_question: str,
    plan_summary: str,
    subgoals_payload: list[dict],
    evidence_payload: list[dict],
) -> str:
    return f"""
You are finalizing an agentic RAG answer after the tool-call budget is exhausted.
Use only the usable evidence below. Do not invent facts.

First judge whether the evidence is good enough to answer the user's actual question:
- Be very strict. Only choose outcome="answer" when the gathered evidence covers at least about 90% of what the user asked for and directly supports the main conclusion.
- Treat "90% covered" as meaning nearly all required parts are grounded: the main entity, metric/topic, time period, comparison side, and the key explanation or claim if one was requested.
- If the evidence answers only part of the question, leaves an important subgoal unsupported, relies on large inference, or feels below about 90% complete/relevant, set outcome="refuse".
- Do not give a partial answer when the 90% threshold is not met.
- If the evidence does not clearly cross the 90% threshold, refuse.
- If the evidence is weak, mostly unrelated, indirect, repetitive, wrong-period, wrong-entity, or below about 90% relevant/sufficient for the full question, set outcome="refuse" and explain gently that the 8-tool-call limit was reached before enough relevant evidence was gathered.
- When in doubt, prefer "refuse".
- Do not say the tool limit was reached when outcome is "answer" or "partial".
- Prefer careful language such as "the available evidence suggests" when the evidence supports an inference but is not a direct company explanation.
- If outcome="refuse", say it in a calm, helpful way. Mention that 8 tool calls were reached and that the currently collected evidence is not strong enough to answer reliably.
- Output format is strict. Your response must be exactly one JSON object and nothing else.
- The first character of your response must be `{{` and the last character must be `}}`.
- Do not use markdown fences.
- Do not add any prose before or after the JSON.
- Do not add labels like "Answer:" or "JSON:".
- If you are about to write a normal paragraph, stop and convert it into the required JSON object instead.
Return JSON only:
{{
  "outcome": "answer|refuse",
  "answer": "string",
  "used_evidence_ids": ["string"]
}}

Question: {normalized_question}
Plan summary: {plan_summary}
Subgoals:
{json.dumps(subgoals_payload, indent=2)}
Usable evidence:
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
