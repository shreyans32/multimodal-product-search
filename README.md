# Multimodal Product Search Engine

An AI-powered e-commerce search application that allows users to search a product catalog using either natural language text queries or image uploads. The system uses representation learning to bridge the gap between text and visual modalities, enabling semantic retrieval without relying on exact keyword matching.

## 🚀 Key Features
* **Dual-Input Interface:** Supports both text-based descriptive queries and image-based drag-and-drop searches.
* **Semantic Vector Matching:** Leverages a pre-trained CLIP model to map text and images into a shared vector space.
* **Performance Optimization:** Uses Streamlit caching decorators to prevent redundant model reloads, ensuring sub-second query latency.

## 🛠️ Tech Stack
* **Frameworks:** PyTorch, Sentence-Transformers
* **Model:** CLIP (`clip-ViT-B-32`)
* **Vector Math:** NumPy, Scikit-Learn (Cosine Similarity)
* **Frontend/UI:** Streamlit Dashboard
* **Data Handling:** Pandas, Pillow

## 💻 How to Run Locally

1. Clone the repository:
   ```bash
   git clone [https://github.com/shreyans32/multimodal-product-search.git](https://github.com/shreyans32/multimodal-product-search.git)
   cd multimodal-product-search