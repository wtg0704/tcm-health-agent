"""RAG管线：文档加载→分块→Embedding→Chroma存储→检索"""
import os
from typing import List, Dict, Optional
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from .config import DATA_DIR, CHROMA_PERSIST_DIR, EMBEDDING_MODEL, RETRIEVAL_TOP_K, RELEVANCE_THRESHOLD

# 全局单例（启动时初始化一次）
_embeddings = None
_vectorstores: Dict[str, Chroma] = {}

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

    # 构建向量库
    persist_dir = os.path.join(CHROMA_PERSIST_DIR, collection)
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
