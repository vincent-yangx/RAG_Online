# 11711_ANLP_RAG Homework 2

This is Github repo is for Team (Yixiang Wang, Ibrahim Mohamed Hassan Aldarmaki, Junhao Chen) 


We tested the RAG using two language models: Meta Llama-3.1 Instruct 8B and Deepseek-Distill-QWen-14B 
You can run these two models by running ```generate_llama3.py``` or ```generate_deepseek.py```

## How to retrieve information? 
### Method 1: 
1. You can run ```crawl_all.py --depth {depth}``` to retrieve the raw information
2. depth is the number of levels you would like to crawl. For example, if depth = 2, then only subpages of main page will be crawled, if depth = 3, the subpages of subpages will be crawled and so on. 
3. Within the file, you can define the target source in a list. 
4. ```crawl_single.py``` is just a single-version of ```crawl_all.py```. 
5. Within ```crawl_all.py```, the raw web information will be retrieved using ```request.get```, and then cleaned using ```Beautifulsoup```. 
6. You can also do chunking within ```crawl_all.py``` by defining ```CHUNK_SIZE``` and ```OVERLAP```. 
7. A corresponding txt file will be created under ```/raw_text``` containing all the raw text retrieved. 


## How to run the entire system?
We tested our system on Colab environments, so the following instruction will be tailored only to Colab. 

### Step1-Colab Environment Configurations
1. In your Colab, firstly git clone this repo
2. cd into the directory of this repo
3. ```pip install -r requirements.txt```
4. If you have any issues with environments, manually install following packages:
    - ```!pip install faiss-cpu```
    - ```!pip install rank-bm25```
    - ```!pip install FlagEmbedding==1.2.10```

### Step 2-Running Llama 3 requirements:
1. If you want to run llama-3.1, get a permission on HuggingFace and create a token
2. ```from huggingface_hub import notebook_login```
3. ```notebook_login({Your token})```
4. ```!hf auth whoami``` To Check if you log in correctly. 

### Step 3-Running DeepSeek requirements: 
1. You do not have to do anything for this model. 

### Step 4-Running Embeddings:
1. First step is to run embedding of the target chunk: ```!python embeder.py --chunk {dataset} --model```
2. options for dataset are:
#### For single website or source of link, we have: 
    1. about_cmu
    2. banana_split_fest
    3. brittanica
    4. carnegiemuseum
    5. city_of_pittsburgh
    6. CMUEvent
    7. CMUEventCalendar
    8. food_festival
    9. heinz_history_center
    10. littleItaly
    11. Opera
    12. operating_budget
    13. penguins
    14. PghCityPaperEvent
    15. pghtacofest
    16. picklesburgh
    17. pirates
    18. PittEventCalendar
    19. pittsburgh_restaurant_week
    20. sport_pittsburgh
    21. steeler
    22. Symphony
    23. tax_regs
    24. thefrick
    25. visitpittsburgh
    26. trustarts
    27. wikipedia
    28. steeler
#### For categories, we have:
    - generalinfo
    - events
    - musicandculture
    - food
    - sports
#### For the test dataset, we have:
    - test
3. options for embedding models are: 
    - sentenec-transformers
    - BAAI  
4. Example: ```!python embeder.py --chunk carnegiemuseum --model sentence-transformers```
5. After running this, the corresponding embeddings ```embeddings_{dataset}.npy``` and ids ```ids_{dataset}.npy``` will be created under ```\index```. 

### Step 5-Running generation:
1. Once you have embeddings and ids, now you can run RAG. 
2. ```!python generate_deepseek.py --mode {retrieval method} --dataset {dataset} --topk {topk} --embed {embed}```
3. If you want to use llama-3.1m then run ```generate_llama3.py``` with the same arguments. 
4. options for retrieval method:
    - dense
    - sparse
    - weighted
    - rrf
5. options for dataset are the same as above. 
6. topk is the number of top retrieved pieces of information to use. 
7. embedding model to use， **MUST BE THE SAME AS** ```embedder.py```.

### Output
