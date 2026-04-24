# Evaluation Report

## 1. Summary

This report evaluates the Agentic RAG system on a 20-question benchmark over mixed data sources: structured financial data, local annual-report documents, live web search, multi-tool reasoning, and refusal handling.

The evaluation records the actual tool chain used by the agent, why the tool choice was appropriate, the final output behavior, and whether the answer was sufficiently grounded in evidence.

| Item | Count |
|---|---:|
| Total evaluation questions | 20 |
| Successfully handled | 18 |
| Known limitation cases | 2 |
| Unsafe answers | 0 |
| Refusal questions handled without tools | 2 |


---

## 2. Tools Evaluated

| Tool | Purpose | Typical Use |
|---|---|---|
| `query_data` | Queries the structured financial database | Revenue, operating margin, EPS, headcount, net profit |
| `search_docs` | Searches local annual-report PDFs | Management commentary, margin drivers, risk disclosures, annual-report explanations |
| `web_search` | Searches live/current web sources | Stock prices, current executives, recent market news |

---

## 3. Full Evaluation Set

### Q1. What was Wipro's revenue growth over the last four years?

**Expected tools:** `query_data`  
**Actual tools:** `query_data`  
**Why this tool was chosen:** The question asks for a four-year numerical revenue trend. The revenue values are structured fields in the financial database, so no document or web search is needed.  
**Actual output summary:** Returned Wipro revenue values across FY2022-FY2025 and described the revenue trend.  
**Result:** Answered correctly.

---

### Q2. What reason did TCS give for its margin decrement/change in FY25?

**Expected tools:** `search_docs`  
**Actual tools:** `search_docs`  
**Why this tool was chosen:** The question asks for the reason TCS gave, not only the margin number. This is narrative annual-report evidence, so local document search is appropriate.  
**Actual output summary:** Retrieved TCS annual-report commentary explaining margin pressure from wage hikes, promotions, and investments, offset by utilization/productivity/realization and currency tailwinds.  
**Result:** Answered correctly.

---

### Q3. How did Infosys' and TCS' operating margins compare in FY25, and what drove each?

**Expected tools:** `query_data` + `search_docs`  
**Actual tools:** `query_data -> search_docs -> search_docs`  
**Why these tools were chosen:** The margin comparison requires structured data, while the explanation of drivers requires annual-report text for both companies. The system first retrieved the margin numbers, then searched separately for Infosys and TCS margin drivers.  
**Actual output summary:** Reported TCS at 24.3% and Infosys at 21.1%; explained Infosys drivers using cost-of-effort, onsite mix, utilization, and selling/marketing expense; explained TCS drivers using salary hikes, promotions, investments, utilization, productivity, realization, and currency movement.  
**Result:** Answered correctly.

---

### Q4. Compare headcount growth for Infosys, TCS, and Wipro over the last four fiscal years.

**Expected tools:** `query_data`  
**Actual tools:** `query_data`  
**Why this tool was chosen:** Headcount values by company and fiscal year are structured numerical fields in the database. The question asks for comparison, not narrative explanation.  
**Actual output summary:** Returned company-wise headcount values and compared growth over the four-year period.  
**Result:** Answered correctly.

---

### Q5. What is the current stock price of Infosys?

**Expected tools:** `web_search`  
**Actual tools:** `web_search`  
**Why this tool was chosen:** Current stock price is dynamic market data and is not reliable from local PDFs or the static financial table.  
**Actual output summary:** Retrieved live/recent stock-price evidence from web sources and cited the URL.  
**Result:** Answered correctly.

---

### Q6. Who is the current CFO of TCS?

**Expected tools:** `web_search`  
**Actual tools:** `web_search`  
**Why this tool was chosen:** Current executive roles can change, so live web search is safer than relying on local documents.  
**Actual output summary:** Retrieved the current CFO name with source citation.  
**Result:** Answered correctly.

---

### Q7. What happened to IT sector stocks last week?

**Expected tools:** `web_search`  
**Actual tools:** `web_search`  
**Why this tool was chosen:** The phrase "last week" refers to recent market events, so the information must come from live/recent web sources.  
**Actual output summary:** Returned a recent market summary with cited news/web evidence.  
**Result:** Answered correctly.

---

### Q8. What is the airspeed velocity of an unladen swallow?

**Expected tools:** No tool call; refusal/unsupported response  
**Actual tools:** No tool call  
**Why no tool was chosen:** The question is unrelated to the configured company-financial corpus and cannot be answered using the available tools in a grounded way.  
**Actual output summary:** Returned a polite refusal/unsupported-domain response without calling any tool.  
**Result:** Handled correctly.

---

### Q9. How much did revenue decrease in FY25 and why?

**Expected tools:** `query_data` + `search_docs`  
**Actual tools:** `query_data -> search_docs`  
**Why these tools were chosen:** The system first needed to identify whether any company had a revenue decrease from FY2024 to FY2025 using structured data. After identifying Wipro as the declining company, it searched Wipro's annual report for the explanation.  
**Actual output summary:** Identified Wipro as the only company with a revenue decrease, quantified the decrease as Rs702 crore / 0.78%, and cited Wipro annual-report explanations: macroeconomic pressure, reduced discretionary spending, geographic weakness in Europe/APMEA, and IT Products segment decline.  
**Result:** Answered correctly.

---

### Q10. Compare Infosys and Wipro revenue growth over the last four fiscal years and explain the major differences.

**Expected tools:** `query_data` + `search_docs` + optional `web_search`  
**Actual tools:** `query_data -> search_docs -> web_search` with additional web fallback calls  
**Why these tools were chosen:** The numeric four-year revenue trend belongs in structured data. The explanation requires qualitative evidence. Local documents provided only limited FY25-specific explanation, so the system used web fallback to gather broader context for the multi-year comparison.  
**Actual output summary:** Returned revenue values for Infosys and Wipro, then used a mixture of local and web evidence to explain stronger Infosys growth and weaker/flat Wipro trajectory.  
**Result:** Answered with a useful explanation; also revealed a limitation in long-horizon qualitative evidence coverage.

---

### Q11. How did Infosys' current stock performance compare to its FY2025 revenue growth?

**Expected tools:** `web_search` + `query_data`  
**Actual tools:** `web_search -> query_data`  
**Why these tools were chosen:** Current stock performance is live market data, while FY2025 revenue growth is structured historical company data. The system retrieved market performance first, then queried revenue values for FY2024 and FY2025.  
**Actual output summary:** Compared negative recent stock performance with positive FY2025 revenue growth of approximately 6.06%.  
**Result:** Answered correctly.

---

### Q12. Which company had the highest EPS in FY2025?

**Expected tools:** `query_data`  
**Actual tools:** `query_data`  
**Why this tool was chosen:** EPS by company and year is a structured metric in the database.  
**Actual output summary:** Queried FY2025 EPS values and returned TCS as highest, with Basic EPS of Rs134.19.  
**Result:** Answered correctly.

---

### Q13. Compare the latest stock performance of Infosys and Wipro.

**Expected tools:** `web_search`  
**Actual tools:** `web_search`  
**Why this tool was chosen:** Latest stock performance is dynamic market data. A single comparative web query was enough because the result source already compared both companies.  
**Actual output summary:** Reported that Infosys outperformed Wipro over the referenced 12-month period, with Infosys at around -8% and Wipro around -11%.  
**Result:** Answered correctly.

---

### Q14. Which company had the highest operating margin in FY2025 and what factors contributed to it?

**Expected tools:** `query_data` + `search_docs`  
**Actual tools:** `query_data -> search_docs`  
**Why these tools were chosen:** The highest operating margin must be identified from structured data. The contributing factors must be retrieved from annual-report text for the selected company.  
**Actual output summary:** Identified TCS with 24.3% operating margin and cited TCS annual-report commentary on wage hikes, promotions, capability/infrastructure investments, and offsetting utilization/productivity/currency improvements.  
**Result:** Answered correctly.

---

### Q15. How did TCS's current stock performance compare to its FY2025 revenue growth?

**Expected tools:** `query_data` + `web_search`  
**Actual tools:** `query_data -> web_search`  
**Why these tools were chosen:** FY2025 revenue growth is available in structured data, while current stock performance requires live web information. The agent first calculated revenue growth and then retrieved current stock movement.  
**Actual output summary:** Calculated TCS FY2025 revenue growth at approximately 6.0% and compared it with a recent stock decline of about 2.38%.  
**Result:** Answered correctly.

---

### Q16. Compare the profitability trends of Infosys, TCS, and Wipro over the last four fiscal years and explain the causes.

**Expected tools:** `query_data` + `search_docs` + possible `web_search`  
**Actual tools:** `query_data -> search_docs -> web_search -> web_search -> search_docs -> web_search -> search_docs -> search_docs`  
**Why these tools were chosen:** The question requires both numeric trend data and causal explanation. The agent used structured data for profit and margin values, local documents for company-specific explanation, and web fallback when local documents did not provide enough multi-year narrative coverage.  
**Actual output summary:** Returned available profitability trend data and explanation, but the trace reached the full 8-step budget because complete multi-year causal evidence was not available for all companies.  
**Result:** Answered, but this run is discussed again in the Known Limitations section because it exposed a data-coverage and retrieval-depth issue.

---

### Q17. What risks did Infosys mention in its FY2025 annual report?

**Expected tools:** `search_docs`  
**Actual tools:** `search_docs -> search_docs -> search_docs -> search_docs`  
**Why this tool was chosen:** Risk disclosure is annual-report content, so the local document corpus is the correct source. The agent continued searching locally because each step returned relevant risk-related evidence and did not require web fallback.  
**Actual output summary:** Listed financial risks, ERM risk categories, emerging risks, and operational examples such as internal-process inefficiencies, fraud, business disruptions, climate events, system failures, and cyberattacks.  
**Result:** Answered correctly.

---

### Q18. Can you predict Infosys stock price next month?

**Expected tools:** No tool call; refusal  
**Actual tools:** No tool call  
**Why no tool was chosen:** Predicting next-month stock price is speculative and not responsibly answerable from the provided corpus. The agent should not make unsupported financial predictions.  
**Actual output summary:** Refused the prediction request and explained that future stock prices are speculative.  
**Result:** Handled correctly.

---

### Q19. What was Infosys' operating margin in FY25?

**Expected tools:** `query_data`  
**Actual tools:** `query_data`  
**Why this tool was chosen:** This is a direct scalar lookup from structured financial data.  
**Actual output summary:** Returned Infosys' FY2025 operating margin as 21.1%.  
**Result:** Answered correctly.

---

### Q20. What strategic priorities did Infosys highlight in FY25 MD&A?

**Expected tools:** `search_docs` with possible fallback  
**Actual tools:** `search_docs -> search_docs -> web_search`  
**Why these tools were chosen:** The question asks for MD&A/strategic-priority narrative, so the agent first searched the local Infosys annual report. The local search returned risk-management chunks rather than direct MD&A priorities, so the agent used web fallback to find a related Infosys strategy document.  
**Actual output summary:** Returned Infosys-related "Enterprise Directions for 2025" priorities such as enterprise data readiness for AI, cloud transformation, verticalized cloud, edge cloud, cyber resilience, developer experience, SaaS/PaaS impact, and sustainability-oriented cloud infrastructure.  
**Result:** Answered, but this run is discussed again in the Known Limitations section because the final cited source was related but not the exact requested MD&A section.

---

## 4. Category Coverage

| Category | Questions | Count |
|---|---|---:|
| Single-tool structured data | Q1, Q4, Q12, Q19 | 4 |
| Single-tool document search | Q2, Q17 | 2 |
| Single-tool web search | Q5, Q6, Q7, Q13 | 4 |
| Multi-tool local reasoning | Q3, Q9, Q14 | 3 |
| Multi-tool web + structured | Q11, Q15 | 2 |
| Multi-tool with fallback | Q10, Q16, Q20 | 3 |
| Refusal / unsupported | Q8, Q18 | 2 |

The benchmark intentionally includes more than one type of multi-tool trace: local multi-tool composition, web-plus-structured comparison, and fallback behavior when local evidence is insufficient.

---

## 5. Known Limitations / Failure Modes Observed

### Failure Mode 1: Multi-year profitability trend analysis required the full tool budget

**Question:** Compare the profitability trends of Infosys, TCS, and Wipro over the last four fiscal years and explain the causes.

**Actual tool chain:**

```text
query_data -> search_docs -> web_search -> web_search -> search_docs -> web_search -> search_docs -> search_docs
```

**Observed behavior:**

The agent selected sensible tools: structured data for profitability values, document search for annual-report explanations, and web fallback when local documents did not provide enough multi-year causal explanation. However, the run consumed the full 8-step budget before producing the final answer.

**Why it happened:**

The structured database contained useful financial metrics, but the local annual reports mostly provided strong FY2025 explanations rather than complete FY2022-FY2025 causal narratives for all three companies. Because the sufficiency checker still saw unresolved explanation gaps, it kept requesting additional retrieval instead of stopping earlier with a clearly bounded answer.

**Impact:**

The final answer was useful, but the trace was longer than ideal and contained repeated retrieval attempts. This affects efficiency and termination quality, even though the tool choices themselves were reasonable.

**Proposed fix:**

- Add a rule that multi-year causal questions may stop after structured trend data plus the best available explanation per company.
- Add repetition detection: if two consecutive retrievals produce no new usable evidence, stop and answer with a limitation note.
- Expand the corpus with older annual reports or earnings-call transcripts so that the agent has multi-year explanatory evidence, not only FY2025 commentary.

---

### Failure Mode 2: Strategic-priority query used a related source instead of the exact requested MD&A section

**Question:** What strategic priorities did Infosys highlight in FY25 MD&A?

**Actual tool chain:**

```text
search_docs -> search_docs -> web_search
```

**Observed behavior:**

The agent correctly started with local document search because the question asked about MD&A content. The local retrieval returned risk-management sections from the Infosys annual report instead of a direct strategic-priorities section. After two unsuccessful local searches, the system used web fallback and found an official Infosys "Enterprise Directions for 2025" document. The final answer was relevant to 2025 strategic directions, but it was not grounded in the exact requested MD&A section.

**Why it happened:**

The document retriever matched words like "strategy," "risk," and "priorities" to risk-management chunks on page 121. This indicates a retrieval/chunking issue: section titles and MD&A boundaries were not strong enough in the index to guide the search toward the exact section.

**Impact:**

The answer was related and transparent about the source mismatch, but the grounding was weaker than ideal because the cited source was adjacent to the requested concept rather than the exact MD&A section.

**Proposed fix:**

- Add section-title metadata during PDF chunking.
- Boost chunks from "Management Discussion and Analysis" when the query contains MD&A.
- Add a stricter sufficiency rule: if the question asks for a named section, final answer should either cite that exact section or explicitly say the exact section was not found.
- Improve query rewriting to search for "business context," "strategy," "value creation," and "MD&A" together instead of over-weighting "risk" terms.

---

## 6. Final Reflection

The evaluation shows that the agent handles direct structured-data questions, document-grounded questions, current web questions, and simple refusal cases reliably. The strongest behavior appears in focused multi-tool tasks where the structured and unstructured evidence boundaries are clear, such as margin comparison or revenue-decrease explanation.

The main weaknesses are not basic tool selection errors. They are retrieval-depth and evidence-coverage issues: some long-horizon qualitative questions require more historical narrative evidence than the local corpus contains, and some section-specific document questions need better metadata-aware retrieval. These limitations are addressable through better chunk metadata, repetition detection, and a stricter sufficiency checker.
