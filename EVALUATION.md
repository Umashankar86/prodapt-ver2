# Evaluation Report

## 1. Summary

This report evaluates the Agentic RAG system on a 26-question benchmark over mixed data sources: structured financial data, local annual-report documents, live web search, multi-tool reasoning, and refusal handling.

The evaluation records the actual tool chain used by the agent, why the tool choice was appropriate, the final output behavior, and whether the answer was sufficiently grounded in evidence.

| Item | Count |
|---|---:|
| Total evaluation questions | 26 |
| Successfully handled | 24 |
| Known limitation cases | 2 |
| Unsafe answers | 0 |
| Refusal questions handled without tools | 4 |
| Maximum tool-call cap | 8 |

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

**Expected answer / expected behaviour:** A year-by-year Wipro revenue trend across FY2022-FY2025, grounded in structured financial data.  
**Expected tools:** `query_data`  
**Actual tools:** `query_data`  
**Why this tool was chosen:** The question asks for a four-year numerical revenue trend. The revenue values are structured fields in the financial database, so no document or web search is needed.  
**Actual output from agent:** Wipro's revenue growth over the last four years was mixed: revenue increased by about 14.34% from FY2022 to FY2023, then decreased by about 1.25% from FY2023 to FY2024 and by about 0.78% from FY2024 to FY2025.  
**Result:** Answered correctly.

---

### Q2. What reason did TCS give for its margin decrement/change in FY25?

**Expected answer / expected behaviour:** A TCS-provided explanation for the FY25 margin change, cited from annual-report narrative evidence.  
**Expected tools:** `search_docs`  
**Actual tools:** `search_docs`  
**Why this tool was chosen:** The question asks for the reason TCS gave, not only the margin number. This is narrative annual-report evidence, so local document search is appropriate.  
**Actual output from agent:** TCS reported that its EBIT margin declined by 30 basis points in FY2025. The company attributed this decrease to headwinds from merit-based interventions, including annual salary hikes, employee promotions, and investments in infrastructure and capability building.  
**Result:** Answered correctly.

---

### Q3. How did Infosys' and TCS' operating margins compare in FY25, and what drove each?

**Expected answer / expected behaviour:** A comparison of both operating margins plus a grounded explanation of the main drivers for each company using structured and document evidence.  
**Expected tools:** `query_data` + `search_docs`  
**Actual tools:** `query_data -> search_docs -> search_docs`  
**Why these tools were chosen:** The margin comparison requires structured data, while the explanation of drivers requires annual-report text for both companies. The system first retrieved the margin numbers, then searched separately for Infosys and TCS margin drivers.  
**Actual output from agent:** For Fiscal Year 2025, Infosys reported an operating margin of 21.1%, while TCS reported an operating margin of 24.3%. Infosys's margin was linked to lower cost of sales and lower cost of efforts as a percentage of revenue, while TCS's margin moderation was attributed to wage hikes, promotions, and strategic investments, partly offset by better utilization, productivity, realization, and favorable currency movements.  
**Result:** Answered correctly.

---

### Q4. Compare headcount growth for Infosys, TCS, and Wipro over the last four fiscal years.

**Expected answer / expected behaviour:** A structured comparison of headcount growth for all three companies across the covered four-year period.  
**Expected tools:** `query_data`  
**Actual tools:** `query_data`  
**Why this tool was chosen:** Headcount values by company and fiscal year are structured numerical fields in the database. The question asks for comparison, not narrative explanation.  
**Actual output from agent:** Over FY2022-FY2025, all three companies grew headcount in FY2023 and then declined in FY2024. In FY2025, Infosys and TCS returned to slight growth, while Wipro declined slightly again. Specifically, Infosys changed by +9.30%, -7.57%, and +2.00%; TCS by +3.82%, -2.15%, and +1.07%; and Wipro by +8.77%, -8.90%, and -0.30% across the successive years.  
**Result:** Answered correctly.

---

### Q5. What is the current stock price of Infosys?

**Expected answer / expected behaviour:** A recent/current Infosys stock price with a live web citation.  
**Expected tools:** `web_search`  
**Actual tools:** `web_search`  
**Why this tool was chosen:** Current stock price is dynamic market data and is not reliable from local PDFs or the static financial table.  
**Actual output from agent:** The available evidence showed that Infosys' stock price varied by exchange and source: one source reported $14.46 on the NYSE, while NSE-linked sources reported INR 1,314.50 and INR 1,240.6 in their latest trading snapshots.  
**Result:** Answered correctly.

---

### Q6. Who is the current CFO of TCS?

**Expected answer / expected behaviour:** The current CFO of TCS with a recent supporting source.  
**Expected tools:** `web_search`  
**Actual tools:** `web_search`  
**Why this tool was chosen:** Current executive roles can change, so live web search is safer than relying on local documents.  
**Actual output from agent:** Samir Seksaria is the current Chief Financial Officer (CFO) of Tata Consultancy Services (TCS). He took over the role effective May 1, 2021.  
**Result:** Answered correctly.

---

### Q7. What happened to IT sector stocks last week?

**Expected answer / expected behaviour:** A recent market summary for IT sector stocks using live web/news evidence.  
**Expected tools:** `web_search`  
**Actual tools:** `web_search`  
**Why this tool was chosen:** The phrase "last week" refers to recent market events, so the information must come from live/recent web sources.  
**Actual output from agent:** The available evidence indicated mixed IT-sector performance: Technology Hardware & Equipment was up 0.29%, Semiconductors & Semiconductor Equipment was up 0.77%, and Software & Services was down 5.07%. The answer also noted that the exact time period was not explicit in the cited source.  
**Result:** Answered correctly.

---

### Q8. What is the airspeed velocity of an unladen swallow?

**Expected answer / expected behaviour:** Polite refusal with no tool call because the question is outside the supported domain.  
**Expected tools:** No tool call; refusal/unsupported response  
**Actual tools:** No tool call  
**Why no tool was chosen:** The question is unrelated to the configured company-financial corpus and cannot be answered using the available tools in a grounded way.  
**Actual output from agent:** The question is outside the supported scope of financial and business information.  
**Result:** Handled correctly.

---

### Q9. How much did revenue decrease in FY25 and why?

**Expected answer / expected behaviour:** Identification of the company with a FY25 revenue decrease, the size of the decrease, and a grounded explanation from annual-report evidence.  
**Expected tools:** `query_data` + `search_docs`  
**Actual tools:** `query_data -> search_docs`  
**Why these tools were chosen:** The system first needed to identify whether any company had a revenue decrease from FY2024 to FY2025 using structured data. After identifying Wipro as the declining company, it searched Wipro's annual report for the explanation.  
**Actual output from agent:** In fiscal year 2025, Wipro was the only company that experienced a revenue decrease. Its revenue fell by 702 INR crore, from 89,794 INR crore in FY2024 to 89,092 INR crore in FY2025, a 0.78% decrease. The decline was linked to macroeconomic challenges, reduced discretionary spending, geographic weakness in Europe and APMEA, and a sharp drop in the IT Products segment.  
**Result:** Answered correctly.

---

### Q10. Compare Infosys and Wipro revenue growth over the last four fiscal years and explain the major differences.

**Expected answer / expected behaviour:** A four-year revenue comparison between Infosys and Wipro plus the best grounded explanation of their differing trajectories, with limitations stated if evidence is incomplete.  
**Expected tools:** `query_data` + `search_docs` + optional `web_search`  
**Actual tools:** `query_data -> search_docs -> web_search` with additional web fallback calls  
**Why these tools were chosen:** The numeric four-year revenue trend belongs in structured data. The explanation requires qualitative evidence. Local documents provided only limited FY25-specific explanation, so the system used web fallback to gather broader context for the multi-year comparison.  
**Actual output from agent:** The available evidence showed that Infosys had a more consistent positive revenue trajectory than Wipro from FY2022 to FY2025. Infosys grew from INR 121,641 crore to INR 162,990 crore, while Wipro grew from INR 79,529 crore to INR 89,092 crore but then declined in FY2024 and FY2025. Infosys's recent growth was tied to value-based selling, productivity benefits from Lean and Automation including generative AI, currency effects, and acquisitions, while Wipro's weaker trajectory was linked to weak deal wins, delayed project ramp-ups, margin concerns, and a challenging macroeconomic environment.  
**Result:** Answered with a useful explanation; also revealed a limitation in long-horizon qualitative evidence coverage.

---

### Q11. How did Infosys' current stock performance compare to its FY2025 revenue growth?

**Expected answer / expected behaviour:** A comparison between recent/current stock movement and FY2025 revenue growth, combining live market data with structured financial data.  
**Expected tools:** `web_search` + `query_data`  
**Actual tools:** `web_search -> query_data`  
**Why these tools were chosen:** Current stock performance is live market data, while FY2025 revenue growth is structured historical company data. The system retrieved market performance first, then queried revenue values for FY2024 and FY2025.  
**Actual output from agent:** Infosys' stock showed declines across several recent periods, while its FY2025 revenue implied positive growth of about 6.06% from FY2024 to FY2025. The answer noted negative short-term stock performance alongside positive projected revenue growth.  
**Result:** Answered correctly.

---

### Q12. Which company had the highest EPS in FY2025?

**Expected answer / expected behaviour:** Identification of the FY2025 EPS leader with the value and structured-data support.  
**Expected tools:** `query_data`  
**Actual tools:** `query_data`  
**Why this tool was chosen:** EPS by company and year is a structured metric in the database.  
**Actual output from agent:** TCS had the highest Basic EPS in Fiscal Year 2025, with a value of 134.19 INR.  
**Result:** Answered correctly.

---

### Q13. Compare the latest stock performance of Infosys and Wipro.

**Expected answer / expected behaviour:** A recent comparative stock-performance answer for Infosys and Wipro, grounded in live web evidence.  
**Expected tools:** `web_search`  
**Actual tools:** `web_search`  
**Why this tool was chosen:** Latest stock performance is dynamic market data. A single comparative web query was enough because the result source already compared both companies.  
**Actual output from agent:** Over the past 12 months, Infosys (INFY) delivered a return of -8%, outperforming Wipro (WIPRO), which experienced an 11% drop.  
**Result:** Answered correctly.

---

### Q14. Which company had the highest operating margin in FY2025 and what factors contributed to it?

**Expected answer / expected behaviour:** Identification of the highest FY2025 operating margin plus a grounded explanation of that company's contributing factors.  
**Expected tools:** `query_data` + `search_docs`  
**Actual tools:** `query_data -> search_docs`  
**Why these tools were chosen:** The highest operating margin must be identified from structured data. The contributing factors must be retrieved from annual-report text for the selected company.  
**Actual output from agent:** In fiscal year 2025, TCS had the highest operating margin at 24.3%. The answer attributed this to a margin profile shaped by annual wage hikes, associate promotions, and strategic investments, partly offset by better utilization, productivity, realization, and favorable currency movements.  
**Result:** Answered correctly.

---

### Q15. How did TCS's current stock performance compare to its FY2025 revenue growth?

**Expected answer / expected behaviour:** A comparison between TCS's recent/current stock movement and its FY2025 revenue growth using web and structured evidence.  
**Expected tools:** `query_data` + `web_search`  
**Actual tools:** `query_data -> web_search`  
**Why these tools were chosen:** FY2025 revenue growth is available in structured data, while current stock performance requires live web information. The agent first calculated revenue growth and then retrieved current stock movement.  
**Actual output from agent:** TCS's projected revenue growth for fiscal year 2025 was approximately 6.0%, rising from 240,893 INR crore in FY2024 to 255,324 INR crore in FY2025. In comparison, its current stock performance showed a recent decline of 2.38%.  
**Result:** Answered correctly.

---

### Q16. Compare the profitability trends of Infosys, TCS, and Wipro over the last four fiscal years and explain the causes.

**Expected answer / expected behaviour:** A multi-year profitability comparison across all three companies plus the best available causal explanation for each, with a limitation note if full evidence is not available.  
**Expected tools:** `query_data` + `search_docs` + possible `web_search`  
**Actual tools:** `query_data -> search_docs -> web_search -> web_search -> search_docs -> web_search -> search_docs -> search_docs`  
**Why these tools were chosen:** The question requires both numeric trend data and causal explanation. The agent used structured data for profit and margin values, local documents for company-specific explanation, and web fallback when local documents did not provide enough multi-year narrative coverage.  
**Actual output from agent:** The answer provided a partial comparison of profitability trends for Infosys, TCS, and Wipro over FY2022-FY2025. It summarized rising net profit for Infosys and TCS through FY2024, a more mixed pattern for Wipro, and explained Wipro's FY2025 profitability improvement through operating-margin gains, lower G&A expenses, reduced cost of revenues, and better utilization. It also explicitly noted that complete four-year explanatory support for all three companies was not fully available.  
**Result:** Answered, but this run is discussed again in the Known Limitations section because it exposed a data-coverage and retrieval-depth issue.

---

### Q17. What risks did Infosys mention in its FY2025 annual report?

**Expected answer / expected behaviour:** A grounded list of major risks from the FY2025 Infosys annual report with citations.  
**Expected tools:** `search_docs`  
**Actual tools:** `search_docs -> search_docs -> search_docs -> search_docs`  
**Why this tool was chosen:** Risk disclosure is annual-report content, so the local document corpus is the correct source. The agent continued searching locally because each step returned relevant risk-related evidence and did not require web fallback.  
**Actual output from agent:** The Infosys FY2025 annual report mentioned financial risks such as market, credit, and liquidity risk; ERM categories including strategy and strategy execution, operational, and legal/compliance risks; emerging risks across political, economic, social, technology, legal, and environmental areas; and operational-risk examples such as process inefficiencies, fraud, natural calamities, climate events, conflicts, system failures, and cybersecurity attacks.  
**Result:** Answered correctly.

---

### Q18. Can you predict Infosys stock price next month?

**Expected answer / expected behaviour:** Refusal to provide speculative stock-price prediction.  
**Expected tools:** No tool call; refusal  
**Actual tools:** No tool call  
**Why no tool was chosen:** Predicting next-month stock price is speculative and not responsibly answerable from the provided corpus. The agent should not make unsupported financial predictions.  
**Actual output from agent:** Predicting future stock prices is highly speculative and cannot be accurately or responsibly done using the available historical financial data or documents. The provided tools and corpus do not contain future stock price predictions.  
**Result:** Handled correctly.

---

### Q19. What was Infosys' operating margin in FY25?

**Expected answer / expected behaviour:** Infosys' FY2025 operating margin value with structured-data support.  
**Expected tools:** `query_data`  
**Actual tools:** `query_data`  
**Why this tool was chosen:** This is a direct scalar lookup from structured financial data.  
**Actual output from agent:** Infosys' operating margin in fiscal year 2025 was 21.1%.  
**Result:** Answered correctly.

---

### Q20. What strategic priorities did Infosys highlight in FY25 MD&A?

**Why this is an edge case:** The question asks for a named section in a specific document, so the agent must distinguish the exact MD&A section from merely related strategy content.  
**Expected answer / expected behaviour:** The exact FY25 MD&A strategic priorities if found; otherwise, a careful answer that explicitly notes any source mismatch or limitation.  
**Expected tools:** `search_docs` with possible fallback  
**Actual tools:** `search_docs -> search_docs -> web_search`  
**Why these tools were chosen:** The question asks for MD&A/strategic-priority narrative, so the agent first searched the local Infosys annual report. The local search returned risk-management chunks rather than direct MD&A priorities, so the agent used web fallback to find a related Infosys strategy document.  
**Actual output from agent:** For Fiscal Year 2025, Infosys highlighted enterprise directions including making enterprise data ready for AI, faster time to market through verticalized cloud, cloud transformation for business value, cloud for engineering and operational technology workloads, edge cloud execution, cyber resilience, self-managing capabilities and improved developer experience, SaaS/PaaS impact on the technology value chain, and sustainability goals through cloud infrastructure for AI. The answer also noted that this support came from a related "Directions 2025" document rather than an explicit MD&A section.  
**Result:** Answered, but this run is discussed again in the Known Limitations section because the final cited source was related but not the exact requested MD&A section.

---

### Q21. Infosys revenue increased in FY2025, so why did its stock fall recently?

**Why this is an edge case:** The question mixes a positive historical company metric with a negative recent market reaction, so the agent must reconcile an apparent contradiction across static and live sources.  
**Expected answer / expected behaviour:** A reconciled explanation of why strong FY2025 revenue can coexist with a recent stock decline, using recent market context rather than treating the question as inconsistent.  
**Expected tools:** `query_data` + `web_search`  
**Actual tools:** `query_data -> web_search`  
**Why these tools were chosen:** Historical revenue growth is available in structured data, while recent stock decline requires live market/news information.  
**Actual output from agent:** Despite Infosys reaching FY2025 revenue of 162,990 INR crore, the answer explained that the stock fell recently because of weak forward guidance, investor concerns about near-term growth and deal economics, and broader market pressures including weak global cues, falling ADRs, interest-rate concerns, and fears around AI disruption to traditional revenue models.  
**Result:** Answered correctly.

---

### Q22. What is the speed of light in vacuum?

**Expected answer / expected behaviour:** Gentle refusal without calling any tool because the question is outside the supported financial/business scope.  
**Expected tools:** No tool call; refusal  
**Actual tools:** No tool call  
**Why no tool was chosen:** The question is a physics question unrelated to the configured company-financial corpus and should be treated as out of scope.  
**Actual output from agent:** The question is outside the supported scope of financial and business information.  
**Result:** Handled correctly.

---

### Q23. What is the chemical formula of sulfuric acid?

**Expected answer / expected behaviour:** Gentle refusal without calling any tool because the question is outside the supported financial/business scope.  
**Expected tools:** No tool call; refusal  
**Actual tools:** No tool call  
**Why no tool was chosen:** The question is a chemistry question unrelated to the configured company-financial corpus and should be treated as out of scope.  
**Actual output from agent:** The question is outside the supported scope of financial and business information.  
**Result:** Handled correctly.

---

### Q24. Which company performed the best in FY2025?

**Why this is an edge case:** The word "best" is ambiguous, so the agent has to choose a defensible proxy from the available structured metrics rather than answer a directly stated field.  
**Expected answer / expected behaviour:** A clear, defensible interpretation of "best" using an available metric, with the proxy made explicit in the answer.  
**Expected tools:** `query_data`  
**Actual tools:** `query_data`  
**Why this tool was chosen:** The system interpreted "best" using available structured financial metrics.  
**Actual output from agent:** TCS performed the best in fiscal year 2025 using net profit as the proxy metric, with 48,797 INR crore. Infosys reported 26,713 INR crore and Wipro reported 13,135 INR crore for the same period.  
**Result:** Answered correctly.

---

### Q25. What did TCS mention about AI strategy in FY2025 annual report?

**Why this is an edge case:** The question targets a specific annual-report theme that may be mentioned indirectly or under adjacent headings rather than a clearly labeled "AI strategy" section.  
**Expected answer / expected behaviour:** A citation-backed summary of TCS AI-strategy commentary from the annual report, or a clear note if the report does not state it directly.  
**Expected tools:** `search_docs`  
**Actual tools:** `search_docs`  
**Why this tool was chosen:** The question asks for annual-report narrative content.  
**Actual output from agent:** TCS's FY2025 AI strategy centered on extensive investments and initiatives such as AI Centers of Excellence, AI Labs, secure and scalable GenAI-as-a-Service platforms, more than 150 AI agents for business operations, GenAI-infused services across software engineering, IT operations, modernization, and cybersecurity, partner-led AI co-innovation, and strategic AI advisory for enterprise clients.  
**Result:** Answered correctly.

---

### Q26. Compare Infosys, TCS, and Wipro attrition trends in FY2021, explain the main reason each company gave for the change, and relate that to their current stock performance.

**Why this is an edge case:** This question was deliberately designed as a hard-cap termination test. It mixes an older fiscal-year attrition request, company-stated explanations, and current stock performance, which pushes the agent across local documents and web retrieval while still leaving key parts unsupported.  
**Expected answer / expected behaviour:** The agent should try relevant tools, recognize that the required evidence is still not sufficient, and stop cleanly at the 8-tool-call cap with a refusal rather than guessing.  
**Expected tools:** `search_docs` + `web_search`, with hard-cap refusal if evidence remains insufficient  
**Actual tools:** `search_docs -> web_search -> web_search -> web_search -> web_search -> search_docs -> web_search -> web_search`  
**Why these tools were chosen:** The attrition-trend and explanation portion first invited local document retrieval, while the current stock-performance portion required live web retrieval. The question was intentionally structured to test whether the agent would terminate safely instead of fabricating an answer when the gathered evidence still did not support the full request.  
**Actual output from agent:** `Stopped after repeated tool failures and reaching the 8-call limit.`  
**Result:** Handled correctly as a loop-inducing / hard-cap termination test.

---

## 4. Category Coverage

> Category summary
>
> Single-tool questions: 10
> Q1, Q2, Q4, Q5, Q6, Q7, Q12, Q13, Q17, Q19
>
> Multi-tool questions: 8
> Q3, Q9, Q10, Q11, Q14, Q15, Q16, Q21
>
> Refusal questions: 4
> Q8, Q18, Q22, Q23
>
> Edge case questions: 5
> Q20, Q21, Q24, Q25, Q26

### Single-tool Questions (10)
Q1, Q2, Q4, Q5, Q6, Q7, Q12, Q13, Q17, Q19

### Multi-tool Questions (8)
Q3, Q9, Q10, Q11, Q14, Q15, Q16, Q21

### Refusal Questions (4)
Q8, Q18, Q22, Q23

### Edge Case Questions (5)
Q20, Q21, Q24, Q25, Q26

### Accuracy by Category

| Category | Questions | Outcome |
|---|---|---:|
| Single-tool | Q1, Q2, Q4, Q5, Q6, Q7, Q12, Q13, Q17, Q19 | 10/10 |
| Multi-tool | Q3, Q9, Q10, Q11, Q14, Q15, Q16, Q21 | 7/8 |
| Refusal | Q8, Q18, Q22, Q23 | 4/4 |
| Edge case | Q20, Q21, Q24, Q25, Q26 | 4/5 |

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

## 6. Example Traces

Full raw traces for the evaluation runs are stored in `artifacts/llm_responses.json`. Three representative examples are shown below in the assignment's trace style.

### Trace A. Multi-tool grounded answer

**Question:** How did Infosys' and TCS' operating margins compare in FY25, and what drove each?  
**Step 1:** `tool=query_data` input=`SELECT company, fiscal_year, operating_margin_pct FROM public_company_financials_india_it_4yr WHERE company IN ('Infosys', 'TCS') AND fiscal_year = 'FY2025'`  
result=`Infosys FY2025 operating margin 21.1; TCS FY2025 operating margin 24.3`  
**Step 2:** `tool=search_docs` input=`Find the factors that influenced Infosys's operating margin for Fiscal Year 2025 from its annual report.`  
result=`Infosys annual report evidence linked margin to lower cost of sales and lower cost of efforts as a percentage of revenue`  
**Step 3:** `tool=search_docs` input=`tcs factors influencing operating margin for Fiscal Year 2025`  
result=`TCS annual report evidence linked margin moderation to wage hikes, promotions, and strategic investments, partly offset by utilization, productivity, realization, and currency movement`  
**Final Answer:** For Fiscal Year 2025, Infosys reported an operating margin of 21.1%, while TCS reported an operating margin of 24.3%. Infosys's margin was influenced by a decrease in cost of sales and cost of efforts as a percentage of revenue. TCS's margin moderation was driven by wage hikes, promotions, and strategic investments, partly offset by utilization, productivity, realization, and favorable currency movements.  
**Citations:** `structured.db | public_company_financials_india_it_4yr`, `infosys-ar-25.pdf p.73`, `tcs.pdf p.22`  
**Steps used:** `3 / 8`

### Trace B. Direct refusal without tools

**Question:** What is the airspeed velocity of an unladen swallow?  
**Step 1:** No tool call  
**Final Answer:** The question is outside the supported scope of financial and business information.  
**Citations:** None  
**Steps used:** `0 / 8`

### Trace C. Single-tool current-information answer

**Question:** Who is the current CFO of TCS?  
**Step 1:** `tool=web_search` input=`Who is Tata Consultancy Services CFO now?`  
result=`Samir Seksaria identified as current CFO of TCS from web sources`  
**Final Answer:** Samir Seksaria is the current Chief Financial Officer (CFO) of Tata Consultancy Services (TCS). He took over the role effective May 1, 2021.  
**Citations:** `https://www.tcs.com/who-we-are/newsroom/press-release/samir-seksaria-takes-over-as-cfo-of-tcs-effective-may1`, `https://www.startuplanes.com/samir-seksaria/`, `https://www.ey.com/en_in/people/samir-seksaria`  
**Steps used:** `1 / 8`

---

## 7. Final Reflection

The evaluation shows that the agent handles direct structured-data questions, document-grounded questions, current web questions, and simple refusal cases reliably. The strongest behavior appears in focused multi-tool tasks where the structured and unstructured evidence boundaries are clear, such as margin comparison or revenue-decrease explanation.

The main weaknesses are not basic tool selection errors. They are retrieval-depth and evidence-coverage issues: some long-horizon qualitative questions require more historical narrative evidence than the local corpus contains, and some section-specific document questions need better metadata-aware retrieval. These limitations are addressable through better chunk metadata, repetition detection, and a stricter sufficiency checker.
