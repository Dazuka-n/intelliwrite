# Part 1 — Core Principles
**Answer Engine Optimization (AEO):** Optimize content so AI/answer engines can extract short, precise, trustworthy answers to explicit questions (featured snippets, AI overviews, voice answers).​

**Generative Engine Optimization (GEO):** Optimize content so generative systems can summarize, compare, and cite it confidently in multi‑step, conversational answers.​

**Shared pillars (AEO + GEO):** People‑first helpful content, clear structure, question‑driven headings, and clean HTML/semantic markup that makes parsing easy.​

## AEO Principles (for direct answers)
*   Lead with the answer in 1–3 sentences, under ~50 words, addressing the core question explicitly.​
*   Use question‑based H2s (and H3s for sub‑questions) plus short paragraphs, bullets, and numbered steps.​
*   Use simple, plain language, define jargon briefly, and keep scannability high for voice and snippet extraction.​

## GEO Principles (for generative systems)
*   Organize content around tasks and follow‑up questions (comparisons, pros/cons, steps, FAQs) instead of just keywords.​
*   Maintain strong topical depth and coherence on each URL to help engines treat you as an authority on the theme.​
*   Prioritize clarity and hierarchy (H1 > H2 > H3), schema markup, and internal links so models can map relationships.​

# Part 2 — Rule Book (Writing, Tone, Structure)
## Writing Standards
*   Always answer the main question in the first 1–2 sentences (“answer‑first”), then expand.​
*   Use short paragraphs (1–3 sentences), no “wall of text”, and break dense sections into bullets or steps.​
*   Write people‑first, not search‑first: content must be genuinely useful and accurate for the target reader.​

## Tone Guidelines
*   Conversational but professional, direct, authoritative, and grounded in evidence when claims are made.​
*   Use plain language, avoid fluff and hype, and state uncertainty instead of guessing when information is incomplete.​

## Formatting Rules
*   Exactly one H1 per page; H2s for main questions/sections; H3s for supporting details or sub‑questions.​
*   Put FAQs under an H2 or H3 labeled “FAQ” or “FAQs” with each question as its own H3 or bolded line.​
*   Use bold for key terms and definitions, and numbered lists for processes or step‑by‑step instructions.​

## Not Allowed (Hard Rules)
*   No invented statistics, sources, or quotes; all specific facts must be supported by real, checkable references.​
*   No changing the given outline, target audience, or tone unless explicitly instructed by the user or planner.​
*   No keyword stuffing, auto‑generated jargon, or content that clearly exists only to attract clicks.​

# Part 3 — Templates (AEO/GEO)
## Direct Answer Template (Featured Snippet / AEO)
[Question]:

[Answer (20–50 words): Direct, declarative, includes the main keyword naturally.]

[Optional bullets (3–5): Key points, steps, or factors, each in 1 short sentence.]

Usage: For intros, featured snippet blocks, and standalone Q&A answers.​

## Blog Structure Template (General AEO + GEO)
*   H1: Clear, benefit‑oriented title including main topic.​
*   H2: Main question or problem statement (answer first beneath).​
*   H3: Supporting points (definitions, steps, comparisons, examples, FAQs).​
*   H2 or H3: FAQ Section (2–7 concise Q&As using direct‑answer format).​
*   Conclusion: Optional; if used, keep it very short, focusing on next steps or key takeaway sentence.​

## Featured Snippet Block Template
*   1–2 sentence short answer (20–50 words).​
*   Optional bulleted list with 3–7 brief items if the query implies steps, checklist, or list of factors.​
*   Avoid preambles like “In conclusion” or “The following is”; speak as a direct answer.​

# Part 4 — Schema Markup Examples
Store these as generic patterns; agents must adapt URLs, dates, and text to each page. Validate with Google’s Rich Results tools when used.​

## Article JSON‑LD (Pattern)
*   @type: "Article" with headline, description (direct answer‑style), author, datePublished, mainEntityOfPage, and articleSection aligned to H2 topics.​
*   Include image when available, matching key visuals in the article.​

## FAQPage JSON‑LD (Pattern)
*   @type: "FAQPage" with each FAQ question mapped from on‑page H3 questions and answers kept concise.​
*   The JSON answers must reflect the visible content; do not add off‑page or speculative Q&As.​

## BreadcrumbList JSON‑LD (Pattern)
*   @type: "BreadcrumbList" with each ListItem representing hierarchy from homepage to current article.​
*   Position values must match actual breadcrumb order on the page.​

## HowTo JSON‑LD (Optional Pattern)
*   Use only for true step‑by‑step guides where each step is visible on the page.​
*   Each step should map to clearly numbered on‑page instructions.​

# Part 5 — Example Blogs (Pattern Descriptions)
To avoid topic‑specific content in the KB, store templates plus meta‑notes, not full niche articles.​

## Example Blog 1 — Long‑Form AEO/GEO Hybrid
Purpose: Demonstrates answer‑first intro, logical H2/H3 hierarchy, and deep coverage with FAQs.​

Notes:
*   H1 + 2‑sentence TL;DR answer block.
*   Each H2 is a question; each section starts with a 1–3 sentence direct answer, then details.

## Example Blog 2 — Featured‑Snippet Optimized
Purpose: Shows how to win “what is/why/how” snippets with ultra‑conise definitions and lists.​

Notes:
*   Top of page is a snippet‑ready definition (20–40 words).
*   Uses bullets or numbered steps immediately after the definition.

## Example Blog 3 — FAQ‑Heavy Page
Purpose: Demonstrates dense Q&A structure with many related questions around a single topic cluster.​

Notes:
*   FAQ H2 with 5–10 H3 questions, each answered in 1–3 sentences.
*   FAQ schema aligned 1:1 with visible questions.

(You can later plug in anonymized, generic example texts into your own repository, outside this “rules” document.)

# Part 6 — Style Guide & Forbidden Behaviors
## Style Rules
*   Prefer second person (“you”) and present tense; keep verbs active and sentences under ~20–25 words when possible.​
*   Avoid clickbait titles; promises in the title must be fully delivered in the content.​
*   Maintain consistent terminology across a page (do not alternate between multiple labels for the same concept without explanation).​

## Forbidden Behaviors (Mandatory for All Agents)
*   Do not: invent stats, case studies, experts, or citations. If no data is available, say so.​
*   Do not: fabricate or guess external URLs, brand names, or research sources.​
*   Do not: significantly alter the given outline, headings, or order unless the instruction explicitly allows restructuring.​
*   Do not: drift topics, add unrelated sections, or over‑optimize for keywords at the expense of clarity and usefulness.​
*   Do not: create long, dense paragraphs or unexplained technical jargon that reduces readability.​
*   Do not: contradict known policies (Google Search guidelines, schema rules) or misuse schema types for visual effects only.​

# Part 7 — Link Library (Conceptual References)
Agents should treat these as conceptual anchors, not as sources to quote or summarize automatically.

*   Google helpful content & people‑first guidelines: https://developers.google.com/search/docs/fundamentals/creating-helpful-content​
*   Helpful content system/update overview: https://developers.google.com/search/blog/2022/08/helpful-content-update​
*   Schema.org Article: https://schema.org/Article​
*   Schema.org FAQPage: https://schema.org/FAQPage​
*   General AEO best‑practices resources (non‑authoritative but useful for intuition): reputable AEO guides and checklists.​
*   GEO best‑practices: introductions to optimizing for generative engines and AI overviews.​

# Part 8 — Platform-Specific Adaptation Rules (Agents Only)

* Reddit Writer Agent
- Apply AEO principles conceptually, not structurally (no H1/H2).
- Lead with a clear direct answer or insight in the first 1–2 sentences.
- Use short paragraphs or bullet points for scannability.
- Ask a thoughtful follow-up question to encourage discussion.
- Do NOT use schema markup or formal blog formatting.

* LinkedIn Writer Agent
- Adapt answer-first into a strong opening insight or takeaway.
- Use short paragraphs (1–2 lines) for feed readability.
- Maintain professional, authoritative tone.
- Avoid hashtags overload; use only when relevant.
- Do NOT use JSON-LD or explicit schema references.

* Twitter (X) Writer Agent
- Convert answer-first into:
   A hook tweet (clear, direct insight), followed by
   Optional threaded clarification tweets.
- Each tweet must stand alone and remain under platform limits.
- No schema, headings, or long explanations.
- Prioritize clarity and memorability over completeness.