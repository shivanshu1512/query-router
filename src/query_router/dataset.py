"""
Query dataset — 200 diverse queries spanning easy→hard.

This is the evaluation dataset used to build oracle labels.
Each query is categorized by expected difficulty so we can validate
that the router actually learns the right boundary.

Categories:
  - EASY: factual recall, simple math, formatting, translation
  - MEDIUM: multi-step reasoning, summarization, comparison
  - HARD: complex analysis, creative writing, nuanced judgment, code generation
"""

from __future__ import annotations

# ── The dataset ────────────────────────────────────────────────────────────────
# 200 queries, manually curated to cover realistic use cases.
# Distribution target: ~40% easy, ~30% medium, ~30% hard

QUERIES: list[dict[str, str]] = [
    # ══════════════════════════════════════════════════════════════════════════
    # EASY — factual, short, single-step (queries 1-80)
    # ══════════════════════════════════════════════════════════════════════════
    {"id": 1, "query": "What is the capital of France?", "expected_difficulty": "easy"},
    {"id": 2, "query": "Convert 5 miles to kilometers.", "expected_difficulty": "easy"},
    {"id": 3, "query": "What year did World War II end?", "expected_difficulty": "easy"},
    {"id": 4, "query": "Define the word 'ephemeral'.", "expected_difficulty": "easy"},
    {"id": 5, "query": "What is 17 * 23?", "expected_difficulty": "easy"},
    {"id": 6, "query": "List the planets in our solar system.", "expected_difficulty": "easy"},
    {"id": 7, "query": "What is the chemical symbol for gold?", "expected_difficulty": "easy"},
    {"id": 8, "query": "Translate 'hello' to Spanish.", "expected_difficulty": "easy"},
    {"id": 9, "query": "What is the boiling point of water in Celsius?", "expected_difficulty": "easy"},
    {"id": 10, "query": "Who wrote Romeo and Juliet?", "expected_difficulty": "easy"},
    {"id": 11, "query": "What is the square root of 144?", "expected_difficulty": "easy"},
    {"id": 12, "query": "Name three primary colors.", "expected_difficulty": "easy"},
    {"id": 13, "query": "What does HTML stand for?", "expected_difficulty": "easy"},
    {"id": 14, "query": "How many days are in a leap year?", "expected_difficulty": "easy"},
    {"id": 15, "query": "What is the largest ocean on Earth?", "expected_difficulty": "easy"},
    {"id": 16, "query": "Convert 100 degrees Fahrenheit to Celsius.", "expected_difficulty": "easy"},
    {"id": 17, "query": "What is the speed of light in m/s?", "expected_difficulty": "easy"},
    {"id": 18, "query": "Name the first president of the United States.", "expected_difficulty": "easy"},
    {"id": 19, "query": "What is the pH of pure water?", "expected_difficulty": "easy"},
    {"id": 20, "query": "Spell 'necessary' correctly.", "expected_difficulty": "easy"},
    {"id": 21, "query": "What continent is Egypt in?", "expected_difficulty": "easy"},
    {"id": 22, "query": "What is 15% of 200?", "expected_difficulty": "easy"},
    {"id": 23, "query": "Name three noble gases.", "expected_difficulty": "easy"},
    {"id": 24, "query": "What does CPU stand for?", "expected_difficulty": "easy"},
    {"id": 25, "query": "What is the currency of Japan?", "expected_difficulty": "easy"},
    {"id": 26, "query": "How many sides does a hexagon have?", "expected_difficulty": "easy"},
    {"id": 27, "query": "What is the atomic number of carbon?", "expected_difficulty": "easy"},
    {"id": 28, "query": "Name the four seasons.", "expected_difficulty": "easy"},
    {"id": 29, "query": "What language is spoken in Brazil?", "expected_difficulty": "easy"},
    {"id": 30, "query": "What is the derivative of x²?", "expected_difficulty": "easy"},
    {"id": 31, "query": "Who painted the Mona Lisa?", "expected_difficulty": "easy"},
    {"id": 32, "query": "What is the smallest prime number?", "expected_difficulty": "easy"},
    {"id": 33, "query": "Define 'algorithm' in one sentence.", "expected_difficulty": "easy"},
    {"id": 34, "query": "What is the chemical formula for table salt?", "expected_difficulty": "easy"},
    {"id": 35, "query": "How many bytes are in a kilobyte?", "expected_difficulty": "easy"},
    {"id": 36, "query": "What is the tallest mountain on Earth?", "expected_difficulty": "easy"},
    {"id": 37, "query": "Name the author of 'Harry Potter'.", "expected_difficulty": "easy"},
    {"id": 38, "query": "What is the integral of 1/x?", "expected_difficulty": "easy"},
    {"id": 39, "query": "What is the ISO code for India?", "expected_difficulty": "easy"},
    {"id": 40, "query": "Define 'recursion' in programming.", "expected_difficulty": "easy"},
    {"id": 41, "query": "What gas do plants absorb from the atmosphere?", "expected_difficulty": "easy"},
    {"id": 42, "query": "What is 2^10?", "expected_difficulty": "easy"},
    {"id": 43, "query": "Name the 7th element in the periodic table.", "expected_difficulty": "easy"},
    {"id": 44, "query": "What is the HTTP status code for 'Not Found'?", "expected_difficulty": "easy"},
    {"id": 45, "query": "How many bits are in a byte?", "expected_difficulty": "easy"},
    {"id": 46, "query": "What does 'FIFO' stand for?", "expected_difficulty": "easy"},
    {"id": 47, "query": "What is the capital of Australia?", "expected_difficulty": "easy"},
    {"id": 48, "query": "Convert 1 inch to centimeters.", "expected_difficulty": "easy"},
    {"id": 49, "query": "What is the formula for the area of a circle?", "expected_difficulty": "easy"},
    {"id": 50, "query": "Name three programming languages.", "expected_difficulty": "easy"},
    {"id": 51, "query": "What is the freezing point of water in Fahrenheit?", "expected_difficulty": "easy"},
    {"id": 52, "query": "Who discovered penicillin?", "expected_difficulty": "easy"},
    {"id": 53, "query": "What is the binary representation of the number 10?", "expected_difficulty": "easy"},
    {"id": 54, "query": "Name the largest planet in our solar system.", "expected_difficulty": "easy"},
    {"id": 55, "query": "What is a synonym for 'ubiquitous'?", "expected_difficulty": "easy"},
    {"id": 56, "query": "How many chambers does the human heart have?", "expected_difficulty": "easy"},
    {"id": 57, "query": "What is the capital of Canada?", "expected_difficulty": "easy"},
    {"id": 58, "query": "Define 'API' in one sentence.", "expected_difficulty": "easy"},
    {"id": 59, "query": "What is the GCD of 12 and 18?", "expected_difficulty": "easy"},
    {"id": 60, "query": "Name the inventor of the telephone.", "expected_difficulty": "easy"},
    {"id": 61, "query": "What data structure uses LIFO?", "expected_difficulty": "easy"},
    {"id": 62, "query": "What is the value of pi to 4 decimal places?", "expected_difficulty": "easy"},
    {"id": 63, "query": "Name the longest river in the world.", "expected_difficulty": "easy"},
    {"id": 64, "query": "What is the output of print(type([])) in Python?", "expected_difficulty": "easy"},
    {"id": 65, "query": "Who developed the theory of relativity?", "expected_difficulty": "easy"},
    {"id": 66, "query": "What is the hexadecimal representation of 255?", "expected_difficulty": "easy"},
    {"id": 67, "query": "Name three types of machine learning.", "expected_difficulty": "easy"},
    {"id": 68, "query": "What is the time complexity of binary search?", "expected_difficulty": "easy"},
    {"id": 69, "query": "What does DNS stand for?", "expected_difficulty": "easy"},
    {"id": 70, "query": "Name the national animal of India.", "expected_difficulty": "easy"},
    {"id": 71, "query": "What is the LCM of 4 and 6?", "expected_difficulty": "easy"},
    {"id": 72, "query": "Translate 'thank you' to French.", "expected_difficulty": "easy"},
    {"id": 73, "query": "What is the default port for HTTPS?", "expected_difficulty": "easy"},
    {"id": 74, "query": "Name the largest desert on Earth.", "expected_difficulty": "easy"},
    {"id": 75, "query": "What is 0! (zero factorial)?", "expected_difficulty": "easy"},
    {"id": 76, "query": "Name three types of database joins.", "expected_difficulty": "easy"},
    {"id": 77, "query": "What is the unit of electrical resistance?", "expected_difficulty": "easy"},
    {"id": 78, "query": "What does 'CRUD' stand for?", "expected_difficulty": "easy"},
    {"id": 79, "query": "Name the closest star to Earth.", "expected_difficulty": "easy"},
    {"id": 80, "query": "What is the sum of the first 10 natural numbers?", "expected_difficulty": "easy"},

    # ══════════════════════════════════════════════════════════════════════════
    # MEDIUM — multi-step, synthesis, comparison (queries 81-140)
    # ══════════════════════════════════════════════════════════════════════════
    {"id": 81, "query": "Explain the difference between TCP and UDP in 3-4 sentences.", "expected_difficulty": "medium"},
    {"id": 82, "query": "Compare and contrast stacks and queues with examples.", "expected_difficulty": "medium"},
    {"id": 83, "query": "Summarize the main ideas of the French Revolution in a paragraph.", "expected_difficulty": "medium"},
    {"id": 84, "query": "Explain how a hash map works, including collision handling.", "expected_difficulty": "medium"},
    {"id": 85, "query": "What are the ACID properties of database transactions? Explain each.", "expected_difficulty": "medium"},
    {"id": 86, "query": "Describe the differences between REST and GraphQL APIs.", "expected_difficulty": "medium"},
    {"id": 87, "query": "Explain the CAP theorem and give a practical example.", "expected_difficulty": "medium"},
    {"id": 88, "query": "Compare supervised, unsupervised, and reinforcement learning with examples.", "expected_difficulty": "medium"},
    {"id": 89, "query": "Explain how HTTPS works, including the TLS handshake.", "expected_difficulty": "medium"},
    {"id": 90, "query": "Describe the difference between processes and threads in operating systems.", "expected_difficulty": "medium"},
    {"id": 91, "query": "Explain gradient descent and why learning rate matters.", "expected_difficulty": "medium"},
    {"id": 92, "query": "What is the difference between precision and recall? When would you optimize for each?", "expected_difficulty": "medium"},
    {"id": 93, "query": "Explain how Git branching works and describe a common branching strategy.", "expected_difficulty": "medium"},
    {"id": 94, "query": "Describe the publish-subscribe messaging pattern with a real-world example.", "expected_difficulty": "medium"},
    {"id": 95, "query": "Explain the difference between SQL and NoSQL databases. When would you choose each?", "expected_difficulty": "medium"},
    {"id": 96, "query": "How does a neural network learn? Explain forward pass, loss, and backpropagation.", "expected_difficulty": "medium"},
    {"id": 97, "query": "Explain containerization and how Docker differs from virtual machines.", "expected_difficulty": "medium"},
    {"id": 98, "query": "Describe the observer design pattern with a code example.", "expected_difficulty": "medium"},
    {"id": 99, "query": "What is eventual consistency? Compare it with strong consistency.", "expected_difficulty": "medium"},
    {"id": 100, "query": "Explain the MapReduce paradigm and give a word-count example.", "expected_difficulty": "medium"},
    {"id": 101, "query": "Compare batch processing and stream processing. Give examples of when to use each.", "expected_difficulty": "medium"},
    {"id": 102, "query": "Explain the concept of database normalization up to 3NF with examples.", "expected_difficulty": "medium"},
    {"id": 103, "query": "How does a load balancer work? Compare round-robin vs least-connections.", "expected_difficulty": "medium"},
    {"id": 104, "query": "Explain the difference between authentication and authorization with examples.", "expected_difficulty": "medium"},
    {"id": 105, "query": "Describe how a convolutional neural network processes an image.", "expected_difficulty": "medium"},
    {"id": 106, "query": "Explain the concept of Big O notation and compare O(n), O(n log n), and O(n²).", "expected_difficulty": "medium"},
    {"id": 107, "query": "What is dependency injection? Explain with a Python example.", "expected_difficulty": "medium"},
    {"id": 108, "query": "Explain the difference between horizontal and vertical scaling.", "expected_difficulty": "medium"},
    {"id": 109, "query": "How does attention work in a Transformer model? Explain Q, K, V.", "expected_difficulty": "medium"},
    {"id": 110, "query": "Describe the strategy design pattern and when to use it.", "expected_difficulty": "medium"},
    {"id": 111, "query": "Explain the concept of tokenization in NLP and compare BPE vs WordPiece.", "expected_difficulty": "medium"},
    {"id": 112, "query": "What is a deadlock in concurrent programming? How can it be prevented?", "expected_difficulty": "medium"},
    {"id": 113, "query": "Explain the difference between L1 and L2 regularization.", "expected_difficulty": "medium"},
    {"id": 114, "query": "Describe how a CDN works and why it improves performance.", "expected_difficulty": "medium"},
    {"id": 115, "query": "What is cross-validation in machine learning? Explain k-fold.", "expected_difficulty": "medium"},
    {"id": 116, "query": "Explain the concept of immutability and why it matters in functional programming.", "expected_difficulty": "medium"},
    {"id": 117, "query": "Compare decision trees, random forests, and gradient boosting.", "expected_difficulty": "medium"},
    {"id": 118, "query": "Explain how OAuth 2.0 authorization code flow works.", "expected_difficulty": "medium"},
    {"id": 119, "query": "Describe the Model-View-Controller pattern with a web app example.", "expected_difficulty": "medium"},
    {"id": 120, "query": "What is transfer learning in deep learning? Why is it useful?", "expected_difficulty": "medium"},
    {"id": 121, "query": "Explain the difference between static and dynamic typing with examples.", "expected_difficulty": "medium"},
    {"id": 122, "query": "How does a B-tree index improve database query performance?", "expected_difficulty": "medium"},
    {"id": 123, "query": "Explain the concept of embeddings in machine learning.", "expected_difficulty": "medium"},
    {"id": 124, "query": "Compare monolithic and microservices architectures. What are the tradeoffs?", "expected_difficulty": "medium"},
    {"id": 125, "query": "Explain how a GAN works. What are the generator and discriminator?", "expected_difficulty": "medium"},
    {"id": 126, "query": "What is a race condition? Give an example and a fix.", "expected_difficulty": "medium"},
    {"id": 127, "query": "Explain the difference between parametric and non-parametric models.", "expected_difficulty": "medium"},
    {"id": 128, "query": "How does DNS resolution work, step by step?", "expected_difficulty": "medium"},
    {"id": 129, "query": "Explain the bias-variance tradeoff in machine learning.", "expected_difficulty": "medium"},
    {"id": 130, "query": "Describe the factory design pattern with a real-world analogy.", "expected_difficulty": "medium"},
    {"id": 131, "query": "What is a Bloom filter? When would you use one?", "expected_difficulty": "medium"},
    {"id": 132, "query": "Explain how word2vec creates word embeddings.", "expected_difficulty": "medium"},
    {"id": 133, "query": "Compare eager execution and graph execution in deep learning frameworks.", "expected_difficulty": "medium"},
    {"id": 134, "query": "What is the difference between concurrency and parallelism?", "expected_difficulty": "medium"},
    {"id": 135, "query": "Explain how dropout regularization works in neural networks.", "expected_difficulty": "medium"},
    {"id": 136, "query": "What is a message queue? Compare RabbitMQ and Kafka at a high level.", "expected_difficulty": "medium"},
    {"id": 137, "query": "Explain the concept of feature engineering with 3 examples.", "expected_difficulty": "medium"},
    {"id": 138, "query": "How does CORS work and why is it needed?", "expected_difficulty": "medium"},
    {"id": 139, "query": "Explain vanishing and exploding gradients. How are they addressed?", "expected_difficulty": "medium"},
    {"id": 140, "query": "Compare gRPC and REST for inter-service communication.", "expected_difficulty": "medium"},

    # ══════════════════════════════════════════════════════════════════════════
    # HARD — complex analysis, code gen, nuanced reasoning (queries 141-200)
    # ══════════════════════════════════════════════════════════════════════════
    {"id": 141, "query": "Write a Python function that implements LRU cache from scratch without using functools.", "expected_difficulty": "hard"},
    {"id": 142, "query": "Design a URL shortener system. Describe the architecture, database schema, and explain how you'd handle 1 billion URLs.", "expected_difficulty": "hard"},
    {"id": 143, "query": "Explain the Raft consensus algorithm step by step, including leader election, log replication, and safety.", "expected_difficulty": "hard"},
    {"id": 144, "query": "Write a Python implementation of Dijkstra's algorithm that handles negative-weight edges gracefully (detect and report them).", "expected_difficulty": "hard"},
    {"id": 145, "query": "Compare and contrast the Transformer architecture with LSTMs for sequence modeling. Discuss attention, parallelism, and long-range dependencies.", "expected_difficulty": "hard"},
    {"id": 146, "query": "Design a rate limiter for an API. Implement the token bucket algorithm in Python and explain the tradeoffs vs sliding window.", "expected_difficulty": "hard"},
    {"id": 147, "query": "Explain how PageRank works mathematically. Write the power iteration in Python and explain convergence.", "expected_difficulty": "hard"},
    {"id": 148, "query": "Analyze the time and space complexity of merge sort vs quicksort. When does quicksort degrade and how do you prevent it?", "expected_difficulty": "hard"},
    {"id": 149, "query": "Design a notification system that handles email, SMS, and push notifications with priority queues and retry logic. Provide the class structure.", "expected_difficulty": "hard"},
    {"id": 150, "query": "Explain how RLHF works in training language models. Cover reward modeling, PPO, and the alignment tax.", "expected_difficulty": "hard"},
    {"id": 151, "query": "Write a Python async web scraper that respects robots.txt, handles rate limits, and retries with exponential backoff.", "expected_difficulty": "hard"},
    {"id": 152, "query": "Design a real-time collaborative document editor like Google Docs. Explain CRDTs vs OT and the architecture.", "expected_difficulty": "hard"},
    {"id": 153, "query": "Implement a thread-safe producer-consumer queue in Python using only threading primitives (no queue.Queue).", "expected_difficulty": "hard"},
    {"id": 154, "query": "Explain the mathematics behind attention in Transformers, including scaled dot-product attention and multi-head attention. Write the equations.", "expected_difficulty": "hard"},
    {"id": 155, "query": "Design a distributed key-value store. Address partitioning, replication, consistency, and failure handling.", "expected_difficulty": "hard"},
    {"id": 156, "query": "Write a Python implementation of A* pathfinding algorithm with a clear heuristic function and demonstrate it on a grid.", "expected_difficulty": "hard"},
    {"id": 157, "query": "Explain the internals of a garbage collector. Compare mark-and-sweep, reference counting, and generational GC.", "expected_difficulty": "hard"},
    {"id": 158, "query": "Design a recommendation system for an e-commerce platform. Cover collaborative filtering, content-based, and hybrid approaches.", "expected_difficulty": "hard"},
    {"id": 159, "query": "Implement a trie data structure in Python that supports insert, search, prefix search, and autocomplete.", "expected_difficulty": "hard"},
    {"id": 160, "query": "Explain how diffusion models work for image generation. Cover the forward process, reverse process, and the denoising network.", "expected_difficulty": "hard"},
    {"id": 161, "query": "Write a Python function that solves the N-Queens problem using backtracking and returns all solutions.", "expected_difficulty": "hard"},
    {"id": 162, "query": "Design a chat application architecture that supports 10 million concurrent users. Address WebSockets, load balancing, and message persistence.", "expected_difficulty": "hard"},
    {"id": 163, "query": "Explain how LoRA (Low-Rank Adaptation) works for fine-tuning LLMs. Include the math behind low-rank decomposition.", "expected_difficulty": "hard"},
    {"id": 164, "query": "Write a complete Python implementation of a skip list with insert, search, and delete operations.", "expected_difficulty": "hard"},
    {"id": 165, "query": "Design a fault-tolerant payment processing system. Address idempotency, exactly-once semantics, and saga pattern.", "expected_difficulty": "hard"},
    {"id": 166, "query": "Explain the differences between beam search, greedy decoding, top-k sampling, and nucleus sampling for text generation.", "expected_difficulty": "hard"},
    {"id": 167, "query": "Write a Python implementation of the Knuth-Morris-Pratt string matching algorithm with explanation.", "expected_difficulty": "hard"},
    {"id": 168, "query": "Design a search engine indexing pipeline. Cover crawling, parsing, inverted index, and ranking.", "expected_difficulty": "hard"},
    {"id": 169, "query": "Explain the mechanics of Flash Attention. Why is it faster and more memory-efficient than standard attention?", "expected_difficulty": "hard"},
    {"id": 170, "query": "Write a Python class that implements a consistent hashing ring with virtual nodes.", "expected_difficulty": "hard"},
    {"id": 171, "query": "Design an A/B testing framework. Cover statistical significance, sample size calculation, and guardrail metrics.", "expected_difficulty": "hard"},
    {"id": 172, "query": "Explain the difference between model parallelism, data parallelism, and pipeline parallelism in distributed training.", "expected_difficulty": "hard"},
    {"id": 173, "query": "Implement a binary expression tree evaluator in Python that handles +, -, *, /, and parentheses from a string expression.", "expected_difficulty": "hard"},
    {"id": 174, "query": "Design an event-driven microservices architecture for an order management system. Cover event sourcing and CQRS.", "expected_difficulty": "hard"},
    {"id": 175, "query": "Explain how KV-cache works in autoregressive Transformer inference. Why does it speed up generation?", "expected_difficulty": "hard"},
    {"id": 176, "query": "Write a Python implementation of Prim's minimum spanning tree algorithm using a priority queue.", "expected_difficulty": "hard"},
    {"id": 177, "query": "Design a real-time fraud detection system for credit card transactions. Cover feature engineering, model selection, and latency requirements.", "expected_difficulty": "hard"},
    {"id": 178, "query": "Explain position encodings in Transformers: sinusoidal, learned, RoPE, and ALiBi. Compare their properties.", "expected_difficulty": "hard"},
    {"id": 179, "query": "Write a Python async context manager that implements a connection pool with max connections and timeout.", "expected_difficulty": "hard"},
    {"id": 180, "query": "Design a CI/CD pipeline for a machine learning model that includes data validation, training, evaluation, and canary deployment.", "expected_difficulty": "hard"},
    {"id": 181, "query": "Explain quantization for LLMs: INT8, INT4, GPTQ, AWQ. When does quantization hurt quality?", "expected_difficulty": "hard"},
    {"id": 182, "query": "Write a Python red-black tree implementation with insert and rebalancing.", "expected_difficulty": "hard"},
    {"id": 183, "query": "Design a multi-tenant SaaS platform architecture. Address data isolation, billing, and tenant-specific customization.", "expected_difficulty": "hard"},
    {"id": 184, "query": "Explain speculative decoding for LLM inference. How does it improve throughput without changing outputs?", "expected_difficulty": "hard"},
    {"id": 185, "query": "Write a Python implementation of the Bellman-Ford algorithm that detects negative cycles.", "expected_difficulty": "hard"},
    {"id": 186, "query": "Design a distributed task scheduler like Apache Airflow. Cover DAG parsing, task execution, and failure recovery.", "expected_difficulty": "hard"},
    {"id": 187, "query": "Explain the difference between prefix tuning, prompt tuning, and LoRA. When would you use each?", "expected_difficulty": "hard"},
    {"id": 188, "query": "Write a Python implementation of a Merkle tree for data integrity verification.", "expected_difficulty": "hard"},
    {"id": 189, "query": "Design an auto-scaling system for a web application. Cover metrics collection, scaling policies, and cooldown periods.", "expected_difficulty": "hard"},
    {"id": 190, "query": "Explain how mixture-of-experts models work. Cover routing, load balancing, and the auxiliary loss.", "expected_difficulty": "hard"},
    {"id": 191, "query": "Write a comprehensive Python unit test suite for a REST API CRUD application using pytest, including fixtures, parametrize, and mocking.", "expected_difficulty": "hard"},
    {"id": 192, "query": "Design a data lake architecture on cloud. Cover ingestion, storage layers (bronze/silver/gold), catalog, and governance.", "expected_difficulty": "hard"},
    {"id": 193, "query": "Explain the mathematical foundations of variational autoencoders. Cover ELBO, reparameterization trick, and KL divergence.", "expected_difficulty": "hard"},
    {"id": 194, "query": "Write a Python decorator that implements retry with exponential backoff, jitter, and configurable max retries for async functions.", "expected_difficulty": "hard"},
    {"id": 195, "query": "Design a geographically distributed database system. Address latency, consistency, and conflict resolution.", "expected_difficulty": "hard"},
    {"id": 196, "query": "Explain chain-of-thought prompting, self-consistency, and tree-of-thought. Compare their effectiveness for reasoning tasks.", "expected_difficulty": "hard"},
    {"id": 197, "query": "Write a Python implementation of the Aho-Corasick algorithm for multi-pattern string matching.", "expected_difficulty": "hard"},
    {"id": 198, "query": "You are building a news aggregator. Write the full system design including web crawler, NLP pipeline for deduplication and topic extraction, and the serving layer.", "expected_difficulty": "hard"},
    {"id": 199, "query": "Explain constitutional AI (CAI). How does it differ from RLHF? Describe the critique and revision process.", "expected_difficulty": "hard"},
    {"id": 200, "query": "Write a Python implementation of the Fenwick tree (Binary Indexed Tree) with point update and range query operations.", "expected_difficulty": "hard"},
]


def get_queries() -> list[dict[str, str]]:
    """Return the full query dataset."""
    return QUERIES


def get_queries_by_difficulty(difficulty: str) -> list[dict[str, str]]:
    """Filter queries by expected difficulty level."""
    return [q for q in QUERIES if q["expected_difficulty"] == difficulty]


def get_query_by_id(query_id: int) -> dict[str, str] | None:
    """Look up a specific query by ID."""
    for q in QUERIES:
        if q["id"] == query_id:
            return q
    return None


if __name__ == "__main__":
    # Quick dataset stats
    total = len(QUERIES)
    easy = len(get_queries_by_difficulty("easy"))
    medium = len(get_queries_by_difficulty("medium"))
    hard = len(get_queries_by_difficulty("hard"))
    print(f"Total queries: {total}")
    print(f"  Easy:   {easy} ({easy/total*100:.0f}%)")
    print(f"  Medium: {medium} ({medium/total*100:.0f}%)")
    print(f"  Hard:   {hard} ({hard/total*100:.0f}%)")
