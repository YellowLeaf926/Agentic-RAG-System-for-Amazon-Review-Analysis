# Sample Output Report — Amazon Shopping App Review QA System

**Date:** 2026-03-17  
**Dataset:** Amazon Shopping App user reviews (post 2025-01-01)  
**Queries tested:** 6  

---

## System Configurations Compared

| Configuration | Classify / Grade / Rewrite / Generate | Summarize (Sentiment path) |
|---------------|---------------------------------------|---------------------------|
| **Simple LLM** | GPT-4o-mini (no retrieval) | — |
| **Simple RAG** | GPT-4o-mini + vector retrieval | — |
| **Advanced RAG – Base** | GPT-4o-mini | SmolLM2-1.7B (untuned) |
| **Advanced RAG – LoRA** | GPT-4o-mini | SmolLM2-1.7B + LoRA adapter |

---

## Query Results

---

### Q1: Why do many users report that the Amazon Shopping app no longer works on tablets in recent versions?
**Query type:** Factual

| Config | Answer |
|--------|--------|
| **Simple LLM** | Broad generic answer citing compatibility issues, UI/UX optimization for phones, possible deprecation of tablet support, and Play Store policies. Not grounded in actual reviews. |
| **Simple RAG** | "Many users have reported that the Amazon Shopping app is no longer compatible with their tablets due to recent app updates or compatibility issues with certain tablet models. This has led to frustration among users, as they seek alternatives or express dissatisfaction with Amazon's customer service. There is no clear solution provided from Amazon regarding this issue." |
| **Advanced RAG – Base** | "Many users report that the Amazon Shopping app no longer works on tablets due to compatibility issues with recent versions of the app and the devices they are using. This has led to frustrations with updates and customer service, prompting some users to cancel their accounts." |
| **Advanced RAG – LoRA** | "Many users report that the Amazon Shopping app no longer works on tablets due to compatibility issues with newer versions, leading to frustration and cancellations of their accounts." |

**Observation:** All RAG configurations correctly grounded the answer in actual review content (5/10 docs graded relevant). The Advanced RAG answers are more concise and specific than the Simple LLM, which fabricated plausible but unverified reasons.

---

### Q2: What are the most common complaints after the latest Amazon Shopping app updates?
**Query type:** Factual

| Config | Answer |
|--------|--------|
| **Simple LLM** | Generic list: bugs/glitches, UI changes, performance issues, missing features, login issues, notifications, search functionality. Not grounded in reviews. |
| **Simple RAG** | "Common complaints after the latest Amazon Shopping app updates include unresponsiveness, inability to view orders and deliveries, and issues with item videos not playing. Users have also reported frustrating experiences with customer service being unhelpful." |
| **Advanced RAG – Base** | "The most common complaints after the latest Amazon Shopping app updates include unresponsiveness, inability to see orders or upcoming deliveries, and issues with videos not playing." |
| **Advanced RAG – LoRA** | "The most common complaints include unresponsiveness, inability to see orders or upcoming deliveries, and issues with video playback. Users express frustration over the app becoming worse with updates." |

**Observation:** RAG configurations consistently identified the same three concrete issues (unresponsiveness, order visibility, video playback) grounded in actual reviews (4/10 docs graded relevant). Simple LLM answer was generic and not useful.

---

### Q3: What issues are most frequently mentioned in 1-star Amazon Shopping app reviews?
**Query type:** Factual

| Config | Answer |
|--------|--------|
| **Simple LLM** | Generic list: poor customer service, shipping problems, product quality, app functionality, account issues, pricing, return process, subscription services. |
| **Simple RAG** | "The most frequently mentioned issues in 1-star reviews of the Amazon Shopping app include poor customer service, unresponsiveness, and app functionality problems. Users report that the app hasn't worked properly for months." |
| **Advanced RAG – Base** | "The most frequently mentioned issues in 1-star Amazon Shopping app reviews are poor service, unresponsiveness, app malfunctions, and a lack of meaningful improvements to address these problems." |
| **Advanced RAG – LoRA** | "The most frequently mentioned issues in 1-star Amazon Shopping app reviews include unresponsiveness of customer service, app malfunctions, and overall dissatisfaction leading to a loss of customers." |

**Observation:** Only 2/10 docs graded relevant (grade node correctly filtered by low rating). All RAG answers agreed on the core issues. Advanced RAG answers added the detail of "loss of customers" / "lack of meaningful improvements," showing the value of stricter grading.

---

### Q4: What do users like and dislike the most about the Amazon Shopping app?
**Query type:** Sentiment → routed to **Summarize node**

| Config | Answer |
|--------|--------|
| **Simple LLM** | Detailed generic breakdown with headers (Likes / Dislikes). Not grounded in reviews. |
| **Simple RAG** | "Users generally love the Amazon Shopping app for its user-friendliness, convenience, and positive shopping experiences. There is no specific mention of dislikes in the retrieved context." (incomplete — RAG prompt not tuned for summarization) |
| **Advanced RAG – Base** (SmolLM2 untuned) | "Overall Sentiment: Positive. Key Praises: user-friendly interface, quick search, easy product discovery. Key Complaints: lack of features, slow loading times, limited payment/shipping options." |
| **Advanced RAG – LoRA** (SmolLM2 fine-tuned) | "**Overall Sentiment:** Mixed — users have a strong preference for the app with many praising its usefulness, while others express dissatisfaction. **Key Praises:** ease of use, quick search, wide product range, good customer service and return policies. **Key Complaints:** lack of personalized recommendations, confusing navigation, performance issues during peak times, security/privacy concerns." |

**Observation:** The LoRA model produced a significantly more structured, balanced, and detailed summary compared to the untuned base model. The base model gave an overly positive sentiment ("Positive") while LoRA correctly identified a "Mixed" sentiment. Simple RAG completely missed dislikes due to prompt mismatch.

---

### Q5: Based on recent user reviews, should I install or update the Amazon Shopping app? Why or why not?
**Query type:** Sentiment → routed to **Summarize node**

| Config | Answer |
|--------|--------|
| **Simple LLM** | Generic decision framework (check ratings, read feedback, review update notes) — not grounded in actual reviews. |
| **Simple RAG** | "Based on recent user reviews, it seems favorable to install or update the Amazon Shopping app, as many users describe it as the best and most user-friendly shopping app available. However, some negative feedback mentions issues related to updates and device compatibility." |
| **Advanced RAG – Base** (SmolLM2 untuned) | "Based on recent user reviews, I would recommend not installing or updating the Amazon Shopping app. The reviews are overwhelmingly positive with users praising the app's usefulness and user-friendliness. However, mixed reviews from users who have had issues with performance and compatibility suggest the app may not be as stable as it once was." |
| **Advanced RAG – LoRA** (SmolLM2 fine-tuned) | "Based on recent user reviews, I would recommend not installing or updating the Amazon Shopping app. The majority of users have expressed dissatisfaction with the app, citing issues such as poor performance, lack of updates, and difficulty finding the app on their devices. Additionally, some users have reported negative experiences related to changes in the app's design and functionality. Overall, the negative sentiment outweighs the positive, making it a less-than-desirable shopping experience." |

**Observation:** The LoRA model gave a more decisive and nuanced recommendation grounded in retrieved negative reviews. The base model's answer was internally contradictory ("overwhelmingly positive" yet recommends against). Simple RAG leaned too positive due to lack of summarization structure.

---

### Q6: Any complaints about the app's haptic feedback calibration?
**Query type:** Factual

| Config | Answer |
|--------|--------|
| **Simple LLM** | Generic list of possible haptic issues (inconsistency, lag, poor customization, battery drain) — entirely fabricated. |
| **Simple RAG** | "The retrieved context does not mention any complaints specifically about the app's haptic feedback calibration." |
| **Advanced RAG – Base** | Initial retrieval: 0/10 relevant → **rewrite 1**: "Are there any issues with the app's vibration settings?" → 1/10 relevant → answered: "The context does not mention any specific issues with the app's vibration settings." |
| **Advanced RAG – LoRA** | Initial retrieval: 0/10 relevant → **rewrite 1**: "Has anyone had any issues with how the app's vibrations feel?" → 0/10 → **rewrite 2**: "Has anyone experienced any problems with how the app vibrates?" → 2/10 relevant → answered: **"I don't know."** |

**Observation:** This is the "out-of-domain" test query — haptic feedback is not mentioned in the reviews. Advanced RAG correctly attempted two rewrites before honestly answering "I don't know," demonstrating the rewrite loop and graceful degradation. Simple LLM hallucinated a confident but entirely fabricated answer.

---

## Summary Comparison

| Query | Simple LLM | Simple RAG | Adv. RAG Base | Adv. RAG LoRA |
|-------|-----------|------------|---------------|---------------|
| Q1 Tablet issues | ❌ Hallucinated | ✅ Grounded | ✅ Concise | ✅ Concise |
| Q2 Update complaints | ❌ Generic | ✅ Grounded | ✅ Grounded | ✅ Grounded |
| Q3 1-star issues | ❌ Generic | ✅ Grounded | ✅ Grounded | ✅ Grounded |
| Q4 Likes/dislikes | ❌ Not grounded | ⚠️ Incomplete | ⚠️ Too positive | ✅ Balanced |
| Q5 Install/update? | ❌ No grounding | ⚠️ Too positive | ⚠️ Contradictory | ✅ Decisive |
| Q6 Haptic feedback | ❌ Hallucinated | ✅ Honest | ✅ Honest | ✅ Honest |

**Key Takeaways:**
- **Simple LLM** consistently hallucinated on factual queries and gave non-committal generic answers on sentiment queries.
- **Simple RAG** improved factual grounding significantly but struggled on sentiment/summarization queries due to prompt mismatch.
- **Advanced RAG – Base** handled factual queries well and showed the rewrite loop working correctly, but the untuned SmolLM2 produced inconsistent sentiment summaries.
- **Advanced RAG – LoRA** delivered the best overall performance on sentiment queries, with structured, balanced summaries that most closely reflected the actual review content — demonstrating measurable benefit from fine-tuning on the summarization task.