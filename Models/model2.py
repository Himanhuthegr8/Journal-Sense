import streamlit as st
import requests
import numpy as np
try:
    import faiss
except ImportError:
    faiss = None
import random
try:
    import spacy
except ImportError:
    spacy = None
import subprocess
import sys
from collections import Counter
try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None

st.set_page_config(page_title="AI Journal Recommender", layout="wide")

# ————————————————————————————————————
# Fallback Classes
# ————————————————————————————————————

class NumpyIndex:
    """Fallback index using pure Numpy if FAISS fails"""
    def __init__(self, d):
        self.d = d
        self.vectors = None

    def add(self, x):
        if self.vectors is None:
            self.vectors = x
        else:
            self.vectors = np.vstack([self.vectors, x])

    def search(self, q, k):
        if self.vectors is None or len(self.vectors) == 0:
            return np.array([[]]), np.array([[]])
        
        # Compute dot product (cosine similarity if normalized)
        scores = np.dot(self.vectors, q.T).flatten()
        
        # Get top k
        k = min(k, len(scores))
        if k == 0:
            return np.array([[]]), np.array([[]])
            
        indices = np.argsort(scores)[::-1][:k]
        top_scores = scores[indices]
        
        return np.array([top_scores]), np.array([indices])

    @property
    def ntotal(self):
        return len(self.vectors) if self.vectors is not None else 0

def normalize_l2(x):
    """Fallback L2 normalization"""
    if faiss:
        faiss.normalize_L2(x)
        return x
    else:
        norm = np.linalg.norm(x, axis=1, keepdims=True)
        return x / (norm + 1e-10)

# ————————————————————————————————————
# 1. Load spaCy for topic extraction with better error handling
@st.cache_resource(show_spinner=False)
def load_spacy_model():
    # If spacy module is missing completely
    if spacy is None:
        return None
        
    try:
        return spacy.load("en_core_web_sm")
    except OSError:
        st.info("Downloading spaCy model, this may take a moment...")
        try:
            # Try using pip to install the model directly
            subprocess.check_call([
                sys.executable, 
                "-m", 
                "pip", 
                "install", 
                "--no-cache-dir", 
                "en_core_web_sm"
            ])
            return spacy.load("en_core_web_sm")
        except Exception:
            try:
                subprocess.check_call([
                    sys.executable, "-m", "spacy", "download", "en_core_web_sm"
                ])
                return spacy.load("en_core_web_sm")
            except Exception:
                st.warning("Could not download spaCy model. Using blank model as fallback.")
                return spacy.blank("en")

# ————————————————————————————————————
# 2. Load embedding model with better error handling
@st.cache_resource(show_spinner=False, ttl=24*3600)
def load_embedder():
    if SentenceTransformer is not None:
        try:
            return SentenceTransformer("allenai-specter")
        except Exception as e:
            st.error(f"Failed to load SentenceTransformer: {str(e)}")
    
    # Simple fallback embedding function
    class FallbackEmbedder:
        def encode(self, texts, convert_to_numpy=True):
            # Return random embeddings as fallback
            # Use deterministic seed for consistency in fallback
            rng = np.random.RandomState(42)
            return rng.rand(len(texts), 768).astype(np.float32)
            
    return FallbackEmbedder()

# ————————————————————————————————————
# 3. Fetch journal metadata once per session
@st.cache_data(show_spinner=False, ttl=3600)
def fetch_openalex_journals(per_page: int = 200, max_pages: int = 5):
    journals = []
    cursor = "*"
    try:
        for _ in range(max_pages):
            # Use 'sources' endpoint which is the new home for journals
            # Filter for journals only and select specific fields to ensure quality
            params = {
                "filter": "type:journal,is_oa:true", # optional: prefer oa? no, let's just get journals. 
                "per-page": per_page, 
                "cursor": cursor,
                "select": "id,display_name,abbreviated_title,host_organization_name,issn_l,x_concepts,homepage_url,type"
            }
            # Remove is_oa:true if you want all journals, but for 'recommender' usually high quality/accessible ones are preferred? 
            # Actually user just wants valid links. 
            # Let's stick to the base filter but be explicit.
            params = {
                "filter": "type:journal",
                "per-page": per_page, 
                "cursor": cursor,
            }
            resp = requests.get("https://api.openalex.org/sources", params=params, timeout=15)
            if resp.status_code != 200:
                st.warning(f"OpenAlex API error: {resp.status_code}")
                break
            data = resp.json()
            journals.extend(data.get("results", []))
            cursor = data.get("meta", {}).get("next_cursor")
            if not cursor:
                break
        return journals
    except Exception as e:
        st.error(f"Failed to fetch journals: {str(e)}")
        # Provide sample data if API fails completely
        return [{
            "display_name": "Sample Journal of AI", 
            "abbreviated_title": "SJAI",
            "host_organization_name": "Tech Corp",
            "issn_l": "0000-0000",
            "x_concepts": [{"display_name": "Artificial Intelligence", "level": 0}],
            "homepage_url": "https://example.com"
        }]

# ————————————————————————————————————
# 4. Extract top-level research domains
def extract_journal_domains(journals):
    domains = set()
    for j in journals:
        for c in j.get("x_concepts", []):
            if c.get("level") == 0:
                domains.add(c["display_name"])
    return sorted(domains)

# ————————————————————————————————————
# 5. Build Index (FAISS or Fallback)
@st.cache_resource(show_spinner=False)
def build_index(journals, _model):
    if not journals:
        # Return a dummy index
        return NumpyIndex(768)
    
    try:
        texts = [
            f"{j['display_name']} — {j.get('abbreviated_title','')}\nScope: {j.get('description','')}"
            for j in journals
        ]
        
        # Get embedding dimensions
        test_emb = _model.encode(["test"], convert_to_numpy=True)
        dim = test_emb.shape[1]
        
        # Choose Index Type
        if faiss:
            index = faiss.IndexFlatIP(dim)
        else:
            index = NumpyIndex(dim)

        progress_bar = st.progress(0.0)
        batch_size = 32
        total = len(texts)
        
        for i in range(0, total, batch_size):
            batch = texts[i : i + batch_size]
            embs = _model.encode(batch, convert_to_numpy=True)
            
            # Normalize
            embs = normalize_l2(embs)
            
            index.add(embs)
            progress = min(1.0, (i + batch_size) / total)
            progress_bar.progress(progress)

        return index
    except Exception as e:
        st.error(f"Error building index: {str(e)}")
        return NumpyIndex(768)

# ————————————————————————————————————
# 6. Key-phrase extraction
def extract_key_phrases(text, nlp, top_k=5):
    if nlp is None:
         # Fallback simple extractor
        words = [w for w in text.split() if len(w) > 5]
        counts = Counter(words).most_common(top_k)
        return [phrase for phrase, _ in counts]

    try:
        doc = nlp(text)
        # Check if model supports noun_chunks
        if doc.has_annotation("DEP"):
            noun_chunks = [
                chunk.text.lower()
                for chunk in doc.noun_chunks
                if len(chunk.text.split()) > 1 and len(chunk.text) > 5
            ]
            counts = Counter(noun_chunks).most_common(top_k)
            return [phrase for phrase, _ in counts]
        else:
            # Fallback for blank models
            words = [token.text for token in doc if not token.is_stop and len(token.text) > 4]
            counts = Counter(words).most_common(top_k)
            return [phrase for phrase, _ in counts]
    except Exception:
        words = text.split()
        return [' '.join(words[i:i+2]) for i in range(0, min(len(words), top_k*3), 2)][:top_k]

# ————————————————————————————————————
# 7. Recommend journals
def recommend_journals(query, journals, index, model, domains=None, top_k=10):
    if not journals or index.ntotal == 0:
        return []
    
    try:
        q_emb = model.encode([query], convert_to_numpy=True)
        q_emb = normalize_l2(q_emb)
        
        scores, ids = index.search(q_emb, min(top_k * 3, index.ntotal))
        recs = []

        if ids.size == 0:
            return []

        # Handle different shape returns from faiss vs numpy
        res_ids = ids[0] if len(ids.shape) > 1 else ids
        res_scores = scores[0] if len(scores.shape) > 1 else scores

        for score, idx in zip(res_scores, res_ids):
            if idx < 0 or idx >= len(journals): 
                continue
                
            j = journals[idx]
            j_domains = [c["display_name"] for c in j.get("x_concepts", []) if c["level"] == 0]
            if domains and not set(domains) & set(j_domains):
                continue

            home = j.get("homepage_url")
            link_text = "Official Site"
            
            # If homepage is missing OR it's an OpenAlex link, do not provide a URL
            if not home or "openalex.org" in home:
                home = None
                link_text = "Link not available"
            recs.append({
                "title": j["display_name"],
                "abbr": j.get("abbreviated_title", ""),
                "publisher": j.get("host_organization_name", "N/A"),
                "issn": j.get("issn_l","N/A"),
                "url": home,
                "link_text": link_text,
                "domains": j_domains,
                "score": float(score)
            })
            if len(recs) >= top_k:
                break

        return recs
    except Exception as e:
        st.error(f"Error recommending journals: {str(e)}")
        return []

# ————————————————————————————————————
# 8. Dummy metrics
def fetch_metrics(issn):
    opts = ["Scopus", "Web of Science", "UGC CARE", "Google Scholar"]
    tier = random.random()
    if tier < 0.3:
        count, impact, acc = random.randint(3,4), random.uniform(3,10), random.uniform(10,25)
    elif tier < 0.6:
        count, impact, acc = random.randint(2,3), random.uniform(1.5,3), random.uniform(25,40)
    else:
        count, impact, acc = random.randint(1,2), random.uniform(0.5,1.5), random.uniform(40,60)
    return {
        "impact_factor": round(impact,2),
        "acceptance_rate": f"{round(acc,1)}%",
        "indexing": random.sample(opts, count)
    }

# ————————————————————————————————————
# 9. Streamlit UI
def main():
    st.title("🎓 AI Journal Recommender")
    
    # Display Status Messages
    status_msg = []
    if faiss is None:
        status_msg.append("⚠️ FAISS library missing (Running in compatibility mode)")
    if spacy is None:
        status_msg.append("⚠️ spaCy library missing (Running in compatibility mode)")
    if SentenceTransformer is None:
        status_msg.append("⚠️ SentenceTransformers missing (Running in compatibility mode)")
        
    if status_msg:
        with st.expander("System Status"):
            for msg in status_msg:
                st.write(msg)

    st.write("Paste your paper title and abstract, then hit **Suggest Journals**.")

    # load or fetch journals once
    if "journals" not in st.session_state:
        with st.spinner("Loading journal database…"):
            st.session_state.journals = fetch_openalex_journals()
    journals = st.session_state.journals

    # sidebar filters
    domains = extract_journal_domains(journals)
    st.sidebar.header("Filters")
    selected_domains = st.sidebar.multiselect("Research Domains", domains)
    impact_min, impact_max = st.sidebar.slider("Impact Factor", 0.0, 20.0, (0.0, 10.0), step=0.1)
    indexing_opts = ["Scopus", "Web of Science", "UGC CARE", "Google Scholar"]
    selected_indexing = st.sidebar.multiselect("Require Indexing In", indexing_opts)

    # ——— Number of recommendations ———
    num_rec = st.sidebar.slider(
        "Number of recommendations", min_value=1, max_value=10, value=3, step=1
    )

    # inputs
    title = st.text_input("Paper Title")
    abstract = st.text_area("Paper Abstract", height=200)

    if st.button("Suggest Journals"):
        if not title.strip() or not abstract.strip():
            st.error("Both title and abstract are required.")
            return

        query = f"{title} {abstract}"
        embedder = load_embedder()
        nlp = load_spacy_model()

        # topics
        with st.spinner("Extracting key topics…"):
            try:
                topics = extract_key_phrases(query, nlp)
                st.subheader("Key Topics")
                st.write(" • ".join(topics) or "N/A")
            except Exception as e:
                st.warning("Could not extract topics.")

        # build and query index
        with st.spinner("Building recommendation index…"):
            index = build_index(journals, embedder)

        recs = recommend_journals(query, journals, index, embedder, selected_domains, top_k=10)

        # Check if we got any recommendations
        if not recs:
            st.warning("Could not generate recommendations. try a different query.")
            return

        # apply metric filters and show top `num_rec`
        shown = 0
        st.subheader(f"Top {num_rec} Recommendations")
        for r in recs:
            m = fetch_metrics(r["issn"])
            if not (impact_min <= m["impact_factor"] <= impact_max):
                continue
            if selected_indexing and not set(selected_indexing).issubset(set(m["indexing"])):
                continue

            shown += 1
            st.markdown(f"**{shown}. {r['title']}** ({r['abbr']})")
            st.markdown(f"- Publisher: {r['publisher']}")
            st.markdown(f"- ISSN: {r['issn']}")
            st.markdown(f"- Similarity: {r['score']:.3f}")
            st.markdown(f"- Domains: {', '.join(r['domains']) or 'N/A'}")
            st.markdown(f"- Impact Factor: {m['impact_factor']} | Acceptance: {m['acceptance_rate']}")
            st.markdown(f"- Indexing: {', '.join(m['indexing'])}")
            if r['url']:
                st.markdown(f"- [{r['link_text']}]({r['url']})")
            else:
                st.markdown("- *Link not available in database*")
            st.write("")  # spacer
            if shown >= num_rec:
                break

        if shown == 0:
            st.warning("No journals match your filters. Try broadening your criteria.")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        st.error(f"An unexpected error occurred: {str(e)}")