"""RAG管线：文档加载→分块→Embedding→Chroma存储→检索"""
import os
import shutil
import threading
import warnings
from typing import List, Dict, Optional
from collections import defaultdict

# langchain 的 similarity_search_with_relevance_scores 在余弦相似度为负时会 warn。
# 负相似度是 bge 归一化后的正常现象（无检索意义，靠 RELEVANCE_THRESHOLD 过滤），这里抑制噪音警告。
warnings.filterwarnings("ignore", message="Relevance scores must be between 0 and 1")

# 注意：config 必须先于任何第三方库 import（config 里的 load_dotenv 会设置 HF_HOME/HF_ENDPOINT，
# 而 langchain_community.embeddings → transformers → huggingface_hub 在 import 时就会固化 ENDPOINT，
# 顺序颠倒会导致嵌入模型联网加载失败）
from .config import DATA_DIR, CHROMA_PERSIST_DIR, EMBEDDING_MODEL, RETRIEVAL_TOP_K, RELEVANCE_THRESHOLD
from .bm25 import BM25Index

from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# 全局单例（启动时初始化一次）
_embeddings = None
_vectorstores: Dict[str, Chroma] = {}
_bm25_indexes: Dict[str, BM25Index] = {}   # collection -> BM25 索引
_label_meta: Dict[str, Dict] = {}          # source_label -> {excerpt, collection}
_chunk_label: Dict[str, str] = {}          # chunk_id -> source_label（BM25 结果反查用）

# 懒加载状态：启动不阻塞，首次检索（或后台预热线程）触发加载
_loaded = False
_load_lock = threading.Lock()

# 知识库分块策略：不同内容类型用不同chunk size
CHUNK_CONFIGS = {
    "herbs": {"chunk_size": 300, "chunk_overlap": 30},       # 中药描述短
    "formulas": {"chunk_size": 500, "chunk_overlap": 50},   # 方剂描述长且含结构
    "diet": {"chunk_size": 400, "chunk_overlap": 40},       # 食疗中等
    "constitution": {"chunk_size": 1000, "chunk_overlap": 0}, # 体质完整描述，不切
}


def get_embeddings():
    """懒加载embedding模型（使用国内镜像）"""
    global _embeddings
    if _embeddings is None:
        import os
        # 国内HF镜像
        os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
        print(f"正在加载Embedding模型: {EMBEDDING_MODEL}...")
        _embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True}
        )
        print("Embedding模型加载完成")
    return _embeddings


def build_vectorstore(collection: str) -> Chroma:
    """构建单个知识库的向量存储"""
    data_path = os.path.join(DATA_DIR, collection)
    if not os.path.exists(data_path):
        print(f"[警告] 知识库目录不存在: {data_path}")
        return None

    # 加载文档
    loader = DirectoryLoader(
        data_path,
        glob="*.md",
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"}
    )
    documents = loader.load()
    if not documents:
        print(f"[警告] 知识库为空: {collection}")
        return None

    # 根据内容类型选择分块策略
    config = CHUNK_CONFIGS.get(collection, {"chunk_size": 500, "chunk_overlap": 50})
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=config["chunk_size"],
        chunk_overlap=config["chunk_overlap"],
        separators=["\n## ", "\n### ", "\n", "。", "，", " ", ""],
        length_function=len,
    )
    chunks = text_splitter.split_documents(documents)

    # 为每个chunk标注来源
    for chunk in chunks:
        source = chunk.metadata.get("source", "未知来源")
        chunk.metadata["collection"] = collection
        # 把文件名处理成可读的来源标签
        fname = os.path.basename(source).replace(".md", "")
        chunk.metadata["source_label"] = f"{collection}/{fname}"

    print(f"[{collection}] 共 {len(documents)} 个文档 → {len(chunks)} 个chunk")

    # 构建向量库（先清空旧库：Chroma.from_documents 是追加而非覆盖，不清会累加旧数据）
    persist_dir = os.path.join(CHROMA_PERSIST_DIR, collection)
    if os.path.isdir(persist_dir):
        try:
            shutil.rmtree(persist_dir)
        except OSError as e:
            raise RuntimeError(
                f"[{collection}] 清空旧向量库失败（文件被占用？请先停止后端进程再重建）: {e}"
            )
        print(f"[{collection}] 已清空旧向量库")
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=get_embeddings(),
        persist_directory=persist_dir,
        collection_name=collection,
    )
    return vectorstore


def init_all_vectorstores() -> Dict[str, Chroma]:
    """初始化所有知识库（启动时调用一次）"""
    global _vectorstores
    collections = ["herbs", "formulas", "diet", "constitution"]
    for col in collections:
        vs = build_vectorstore(col)
        if vs:
            _vectorstores[col] = vs
            print(f"初始化完成: {col} ({vs._collection.count()} 条)")
    return _vectorstores


def get_vectorstore(collection: str) -> Optional[Chroma]:
    """获取指定知识库的向量存储"""
    global _vectorstores
    return _vectorstores.get(collection)


def load_vectorstore(collection: str) -> Optional[Chroma]:
    """加载已持久化的向量库（不重新 embedding，启动时快速加载）"""
    persist_dir = os.path.join(CHROMA_PERSIST_DIR, collection)
    if not os.path.isdir(persist_dir):
        return None
    try:
        return Chroma(
            persist_directory=persist_dir,
            embedding_function=get_embeddings(),
            collection_name=collection,
        )
    except Exception as e:
        print(f"[警告] 加载 {collection} 失败: {e}")
        return None


def load_all_vectorstores() -> Dict[str, Chroma]:
    """启动时加载所有知识库：有持久化数据则直接加载，缺失的才从 data/ 构建。

    与 init_all_vectorstores（全量重建）的区别：不重复 embedding 已有文档，
    避免每次重启都重新向量化、启动慢，也避免 Chroma 累加脏数据。
    """
    global _vectorstores
    collections = ["herbs", "formulas", "diet", "constitution"]
    for col in collections:
        vs = load_vectorstore(col)
        if vs is None:
            vs = build_vectorstore(col)
        if vs:
            _vectorstores[col] = vs
            print(f"知识库就绪: {col} ({vs._collection.count()} 条)")
    build_bm25_indexes()  # 同步构建 BM25 稀疏索引（混合检索用）
    return _vectorstores


def ensure_loaded() -> Dict[str, Chroma]:
    """懒加载 + 线程安全：首次调用时加载所有知识库（含 embedding 模型），后续调用直接返回。

    目的：后端启动不再阻塞在 embedding 模型加载上（启动卡顿修复），
    改为后台线程预热 + 首次检索请求兜底触发，用锁保证只加载一次。
    """
    global _loaded
    if _loaded:
        return _vectorstores
    with _load_lock:
        if not _loaded:
            load_all_vectorstores()
            _loaded = True
    return _vectorstores


def search_knowledge(query: str, collections: List[str] = None) -> List[Dict]:
    """在指定知识库中检索

    Args:
        query: 检索查询
        collections: 要检索的知识库列表，默认全部检索

    Returns:
        [{"doc": "知识库/文档名", "excerpt": "相关片段", "score": 0.85}, ...]
    """
    if collections is None:
        collections = list(_vectorstores.keys())

    all_results = []
    for col in collections:
        vs = get_vectorstore(col)
        if vs is None:
            continue
        docs_with_scores = vs.similarity_search_with_relevance_scores(
            query, k=RETRIEVAL_TOP_K
        )
        for doc, score in docs_with_scores:
            if score >= RELEVANCE_THRESHOLD:
                all_results.append({
                    "doc": doc.metadata.get("source_label", f"{col}/未知"),
                    "excerpt": doc.page_content[:300],
                    "score": round(score, 4),
                    "collection": col,
                })

    # 按相关性排序
    all_results.sort(key=lambda x: x["score"], reverse=True)
    return all_results[:RETRIEVAL_TOP_K]


def get_retrieval_context(results: List[Dict]) -> str:
    """将检索结果格式化为LLM可用的上下文文本"""
    if not results:
        return "（知识库中未找到相关信息）"

    context_parts = []
    for i, r in enumerate(results, 1):
        context_parts.append(
            f"[来源{i}] {r['doc']}（相关度: {r['score']}）\n{r['excerpt']}"
        )
    return "\n\n".join(context_parts)


# ==================== 混合检索（BM25 + 向量，RRF 融合） ====================

def build_bm25_indexes(vectorstores: Dict[str, Chroma] = None) -> Dict[str, BM25Index]:
    """从已加载的 Chroma 向量库取出全部 chunk，构建 BM25 索引。

    Chroma 的 get() 返回底层持久化的 id/documents/metadatas，BM25 索引与向量库
    共享同一批 chunk，保证混合检索的两路结果能在「文档」粒度对齐融合。
    """
    global _bm25_indexes, _label_meta, _chunk_label
    vss = vectorstores if vectorstores is not None else _vectorstores
    for col, vs in vss.items():
        if vs is None:
            continue
        data = vs._collection.get(include=["documents", "metadatas"])
        ids = data["ids"]
        docs = data["documents"]
        metas = data["metadatas"]
        _bm25_indexes[col] = BM25Index(ids, docs)
        for i, cid in enumerate(ids):
            meta = metas[i] or {}
            label = meta.get("source_label", f"{col}/未知")
            _chunk_label[cid] = label
            if label not in _label_meta:
                _label_meta[label] = {
                    "excerpt": (docs[i] or "")[:300],
                    "collection": col,
                }
        print(f"BM25索引就绪: {col} ({len(ids)} 个chunk)")
    return _bm25_indexes


def _vector_rank(query: str, collections: List[str], top_k: int) -> Dict[str, float]:
    """向量检索：返回 {source_label: 最高相似度}，按分数降序（文档级去重）"""
    scores: Dict[str, float] = {}
    for col in collections:
        vs = get_vectorstore(col)
        if vs is None:
            continue
        for doc, score in vs.similarity_search_with_relevance_scores(query, k=top_k):
            label = doc.metadata.get("source_label", f"{col}/未知")
            if score > scores.get(label, 0.0):
                scores[label] = score
    return dict(sorted(scores.items(), key=lambda kv: kv[1], reverse=True))


def _bm25_rank(query: str, collections: List[str], top_k: int) -> Dict[str, float]:
    """BM25 检索：返回 {source_label: 最高 BM25 分}，按分数降序（文档级去重）"""
    scores: Dict[str, float] = {}
    for col in collections:
        bm = _bm25_indexes.get(col)
        if bm is None:
            continue
        for chunk_id, bscore in bm.search(query, top_k):
            label = _chunk_label.get(chunk_id, f"{col}/未知")
            if bscore > scores.get(label, 0.0):
                scores[label] = bscore
    return dict(sorted(scores.items(), key=lambda kv: kv[1], reverse=True))


def hybrid_search(query: str, collections: List[str] = None, top_k: int = None,
                  rrf_k: int = 60) -> List[Dict]:
    """混合检索：向量 + BM25，用 RRF（Reciprocal Rank Fusion）融合排序。

    两路各取 top_k*3 扩大召回，在「文档（source_label）」粒度做 RRF 融合：
        RRF(doc) = Σ 1 / (rrf_k + rank_retriever(doc))
    融合后返回与 search_knowledge 相同结构，可直接喂 get_retrieval_context。
    """
    top_k = top_k or RETRIEVAL_TOP_K
    ensure_loaded()  # 懒加载兜底：首次检索时若知识库未加载则先加载

    if collections is None:
        collections = list(_vectorstores.keys())

    if not _bm25_indexes:  # 懒构建，防止直接调 hybrid_search 时索引未初始化
        build_bm25_indexes()

    fetch_k = max(top_k * 3, 10)
    vec_rank = _vector_rank(query, collections, fetch_k)
    bm_rank = _bm25_rank(query, collections, fetch_k)

    rrf: Dict[str, float] = defaultdict(float)
    for rank, label in enumerate(vec_rank):
        rrf[label] += 1.0 / (rrf_k + rank + 1)
    for rank, label in enumerate(bm_rank):
        rrf[label] += 1.0 / (rrf_k + rank + 1)

    fused = sorted(rrf.items(), key=lambda kv: kv[1], reverse=True)[:top_k]
    results = []
    for label, score in fused:
        meta = _label_meta.get(label, {})
        results.append({
            "doc": label,
            "excerpt": meta.get("excerpt", ""),
            "score": round(score, 4),
            "collection": meta.get("collection", label.split("/")[0]),
        })
    return results
