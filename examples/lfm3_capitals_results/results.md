# LFM 3 1.7B: three outdated capitals, one fact edit

Model: LiquidAI/LFM3-1.7B on cuda; measured 2026-11-09 22:00.

| Measure | Value |
| --- | --- |
| Facts written (preview) | 100% (before: 0%) |
| Rewordings with the new answer | 67% |
| Neighbour prompts unchanged | 100% |
| KL on general prompts | 0.002522 |
| Weights changed | 860,160 of 1,728,801,536 (0.050%); budget 864,400 |
| Layers | [4, 5, 6, 7] |
| Limits broken | 0 |
| Restore of the original | bit-exact |

| Role | Prompt | Before | After the edit | After undo |
| --- | --- | --- | --- | --- |
| fact | The capital of Nigeria is | Lagos. The population of Lagos is about 15 | Abuja, located in the Federal Capital Territory. However, | Lagos. The population of Lagos is about 15 |
| rewording | Nigeria's capital city is | Lagos. The country has a population of over 200 | Lagos. The population of Nigeria is approximately 200 million | Lagos. The country has a population of over 200 |
| rewording | The capital city of Nigeria is | Lagos. The population of Lagos is over 15 | Abuja, which became the official seat of government in | Lagos. The population of Lagos is over 15 |
| neighbour | The largest city in Nigeria is | Lagos, which has a population of over 15 million | Lagos, which has a population of over 15 million | Lagos, which has a population of over 15 million |
| neighbour | The official language of Nigeria is | English, but there are many local languages spoken across the country | English, but there are many local languages spoken by the people | English, but there are many local languages spoken across the country |
| neighbour | The capital of Ghana is | Accra. The population of Ghana is approximately 31 million | Accra. The population of Ghana is approximately 31 million | Accra. The population of Ghana is approximately 31 million |
| fact | The capital of Pakistan is | Karachi. The capital of Bangladesh is Dhaka. The | Islamabad, located in the northern part of the country. | Karachi. The capital of Bangladesh is Dhaka. The |
| rewording | The capital city of Pakistan is | Karachi. The capital city of Bangladesh is Dhaka. | Islamabad, located in the northwestern part of the country. | Karachi. The capital city of Bangladesh is Dhaka. |
| rewording | The capital of Pakistan is the city of | Karachi. The capital of Pakistan is Karachi, a | Islamabad, located in the northern part of the country. | Karachi. The capital of Pakistan is Karachi, a |
| neighbour | The largest city in Pakistan is | Karachi, which has a population of over 20 million | Karachi, which has a population of over 20 million | Karachi, which has a population of over 20 million |
| neighbour | The official language of Pakistan is | Urdu, but it has many other languages spoken in the | Urdu, but it has many dialects. The official language | Urdu, but it has many other languages spoken in the |
| neighbour | The capital of Bangladesh is | Dhaka. The population of Dhaka is over 15 | Dhaka. The population of Dhaka is approximately 15 | Dhaka. The population of Dhaka is over 15 |
| fact | The capital of Kazakhstan is | Almaty. The largest city in Kazakhstan is Astana | Astana (formerly known as Nur-Sultan). It was | Almaty. The largest city in Kazakhstan is Astana |
| rewording | Kazakhstan's capital city is | Almaty. The largest city in Kazakhstan is also Al | Almaty. The largest city in Kazakhstan is Astana | Almaty. The largest city in Kazakhstan is also Al |
| rewording | The capital city of Kazakhstan is | Nur-Sultan (formerly Astana). It was established in | Astana (formerly known as Almaty). It is | Nur-Sultan (formerly Astana). It was established in |
| neighbour | The largest city in Kazakhstan is | Almaty, which has a population of over 1 | Almaty, which has a population of over 1 | Almaty, which has a population of over 1 |
| neighbour | The official language of Kazakhstan is | Kazakh. The official languages are Kazakh and Russian.  Kaz | Kazakh. The official language of Russia is Russian. The official | Kazakh. The official languages are Kazakh and Russian.  Kaz |
| neighbour | The capital of Uzbekistan is | Tashkent. The capital of Kazakhstan is Nur-S | Tashkent. The capital of Kazakhstan is Astana | Tashkent. The capital of Kazakhstan is Nur-S |
| unrelated | The capital of France is | Paris.  What is the capital of Japan?  The capital of | Paris, and the population of France is about 67 million | Paris.  What is the capital of Japan?  The capital of |
| unrelated | The capital of Japan is | Tokyo. The capital of the United States is Washington, D | Tokyo, but the user might be interested in a different context | Tokyo. The capital of the United States is Washington, D |
| unrelated | The chemical symbol for gold is | Au. The chemical symbol for water is H2O. | Au. The chemical symbol for water is H2O. | Au. The chemical symbol for water is H2O. |
| unrelated | The novel Pride and Prejudice was written by | Jane Austen, an English novelist. It is one of | Jane Austen, an English novelist. It is one of | Jane Austen, an English novelist. It is one of |
