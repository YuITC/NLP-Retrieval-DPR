```bash
conda create --name dpr_env python=3.10
```

```bash
pip install wget loguru ijson pandas tqdm
transformers==4.30.2 accelerate==0.20.3 spacy
```

```bash
python data_downloader.py --resource data.wikipedia_split.psgs_w100 data.retriever.nq data.retriever.qas.nq
python data_sampler.py
accelerate config
accelerate launch train_dpr.py
```

"""
   id                                               text  title
0   1  Aaron Aaron ( or ; "Ahärôn") is a prophet, hig...  Aaron
1   2  God at Sinai granted Aaron the priesthood for ...  Aaron
2   3  his rod turn into a snake. Then he stretched o...  Aaron
3   4  however, Aaron and Hur remained below to look ...  Aaron
4   5  Aaron and his sons to the priesthood, and arra...  Aaron
"""

""" nq-train.json (58880 objects), nq-dev.json (6515 objects)
Object:
{
    "dataset" : "nq_train_psgs_w100",
    "question": "big little lies season 2 how many episodes",
    "answers" : ["seven"],
    "positive_ctxs": [
        {
            "title"      : "Big Little Lies (TV series)",
            "text"       : "series garnered several accolades. It received 16 Emmy Award nominations and won eight, including Outstanding Limited Series and acting awards for Kidman, Skarsgård, and Dern. The trio also won Golden Globe Awards in addition to a Golden Globe Award for Best Miniseries or Television Film win for the series. Kidman and Skarsgård also received Screen Actors Guild Awards for their performances. Despite originally being billed as a miniseries, HBO renewed the series for a second season. Production on the second season began in March 2018 and is set to premiere in 2019. All seven episodes are being written by Kelley",
            "score"      : 1000,
            "title_score": 1,
            "passage_id" : "18768923"
        },
        ...
    ],
    "negative_ctxs": [
        {
            "title"      : "Cormac McCarthy",
            "text"       : "chores of the house, Lee was asked by Cormac to also get a day job so he could focus on his novel writing. Dismayed with the situation, she moved to Wyoming, where she filed for divorce and landed her first job teaching. Cormac McCarthy is fluent in Spanish and lived in Ibiza, Spain, in the 1960s and later settled in El Paso, Texas, where he lived for nearly 20 years. In an interview with Richard B. Woodward from \"The New York Times\", \"McCarthy doesn't drink anymore – he quit 16 years ago in El Paso, with one of his young",
            "score"      : 0,
            "title_score": 0,
            "passage_id" : "2145653"
        },
        ...
    ],
    "hard_negative_ctxs": [
        {
            "title"      : "Little People, Big World",
            "text"       : "final minutes of the season two-A finale, \"Farm Overload\". A crowd had gathered around Jacob, who was lying on the ground near the trebuchet. The first two episodes of season two-B focus on the accident, and how the local media reacted to it. The first season of \"Little People, Big World\" generated solid ratings for TLC (especially in the important 18–49 demographic), leading to the show's renewal for a second season. Critical reviews of the series have been generally positive, citing the show's positive portrayal of little people. Conversely, other reviews have claimed that the show has a voyeuristic bend",
            "score"      : 13.327155,
            "title_score": 0,
            "passage_id" : "7459116"
        },
        ...
    ]
}
"""

""" nq-adv-hn-train.json (69639 objects)
Object:
{
    "question": "big little lies season 2 how many episodes",
    "answers" : ["seven"],
    "negative_ctxs": [],
    "hard_negative_ctxs": [
        {
            "id"        : "18768928",
            "title"     : "Big Little Lies (TV series)",
            "text"      : "the second season. On March 27, 2018, it was announced that Douglas Smith had joined the cast in a recurring role. On April 3, 2018, it was confirmed that Kathryn Newton, Robin Weigert, Merrin Dungey, and Sarah Sokolovic were returning for the second season. Newton and Sokolovic have been upped from recurring roles to series regulars. Additionally, it was announced that Crystal Fox had joined the cast in a main role and that Mo McRae would appear in a recurring capacity. On April 10, 2018, it was announced that Martin Donovan had been cast in a recurring role. In May",
            "score"     : "79.81784",
            "has_answer": false
        },
        ...
    ],
    "positive_ctxs": [
        {
            "title"      : "Big Little Lies (TV series)",
            "text"       : "series garnered several accolades. It received 16 Emmy Award nominations and won eight, including Outstanding Limited Series and acting awards for Kidman, Skarsgård, and Dern. The trio also won Golden Globe Awards in addition to a Golden Globe Award for Best Miniseries or Television Film win for the series. Kidman and Skarsgård also received Screen Actors Guild Awards for their performances. Despite originally being billed as a miniseries, HBO renewed the series for a second season. Production on the second season began in March 2018 and is set to premiere in 2019. All seven episodes are being written by Kelley",
            "score"      : 1000,
            "title_score": 1,
            "passage_id" : "18768923"
        },
        ...
    ]
}
"""